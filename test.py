from functools import partial
import os
import time
import yaml
import torch

from argparse import ArgumentParser, Namespace
from monai.data import decollate_batch
from monai.handlers import from_engine
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureTyped,
    Activationsd,
    Invertd,
    AsDiscreted,
    SaveImaged,
    ToDeviced,
    Orientationd,
)
from torch.amp import autocast

from utils.logger import get_root_logger
from networks.utils.misc import model_from_cfg
from data.multi_modal import get_loaders


def _make_unet_predictor(model, modality_tensor: torch.Tensor | None):
    """Wrap UNet forward so sliding_window_inference can call predictor(x) only.
    It broadcasts a single-item modality tensor to current window batch size.
    """
    def _predictor(x: torch.Tensor) -> torch.Tensor:
        if modality_tensor is None:
            return model(x)
        b = x.shape[0]
        styles = modality_tensor
        if isinstance(styles, torch.Tensor):
            styles = styles.view(-1)
            if styles.numel() == 1:
                styles = styles.repeat(b)
            elif styles.shape[0] != b:
                repeat = (b + styles.shape[0] - 1) // styles.shape[0]
                styles = styles.repeat(repeat)[:b]
        return model(x, styles)

    return _predictor


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_root_logger(log_file=os.path.join(cfg.experiment, "test.log"), file_mode="w")

    logger.info("=> Building model ...")
    model = model_from_cfg(cfg).to(device=device)
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    weight_path = os.path.join(cfg.experiment, "model", cfg.weight)
    logger.info(f"=> Loading weight at {weight_path} ...")
    load_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(load_dict["state_dict"])
    load_epoch = load_dict.get("epoch", "n/a")
    best_metric = load_dict.get("best", load_dict.get("dice", float("nan")))
    try:
        best_str = f"{best_metric:.4f}"
    except (TypeError, ValueError):
        best_str = str(best_metric)
    logger.info(
        f"=> Loaded weight at {weight_path} (epoch {load_epoch}, best {best_str})"
    )

    logger.info("=> Building dataloader for testing ...")
    cfg.test_mode = True
    test_loader, preprocessing = get_loaders(cfg)
    logger.info(f"Totally {len(test_loader)} samples in test loader.")

    postprocessing = Compose(
        [
            EnsureTyped(keys=["pred", "label"]),
            Activationsd(keys="pred", softmax=True),
            ToDeviced(keys=["pred", "label", "image"], device="cpu"),
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=os.path.join(cfg.experiment, "result"),
                output_ext=".mhd",
                output_postfix="pred",
                resample=False,
                print_log=False,
            ),
            AsDiscreted(keys=["pred", "label"], to_onehot=cfg.out_channels),
            Orientationd(keys=["pred", "label"], axcodes="RAS"),
        ]
    )

    dice_metric = DiceMetric(include_background=cfg.include_background, reduction="none")
    all_modalities: list[int] = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            start = time.time()

            # move tensors to device
            if torch.cuda.is_available():
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)

            # prepare predictor that handles modality broadcasting for UNet
            modality_tensor = batch.get("modality", None)
            predictor = _make_unet_predictor(model, modality_tensor)

            model_inferer = partial(
                sliding_window_inference,
                roi_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
                sw_batch_size=cfg.sw_batch_size,
                predictor=predictor,
                overlap=cfg.infer_overlap,
            )

            with autocast(enabled=cfg.amp, device_type="cuda"):
                batch["pred"] = model_inferer(batch["image"]) if model_inferer is not None else predictor(batch["image"]) 

            batch = [postprocessing(i) for i in decollate_batch(batch)]
            y_pred, y = from_engine(["pred", "label"])(batch)

            # accumulate dice and corresponding modality ids per sample
            dice_metric(y_pred=y_pred, y=y)
            bt_mod = modality_tensor
            if isinstance(bt_mod, torch.Tensor):
                all_modalities.extend(bt_mod.view(-1).detach().cpu().tolist())
            else:
                # if no modality provided, mark as -1
                all_modalities.extend([-1] * len(y_pred))
            logger.info(f"Test {idx + 1}/{len(test_loader)} [{time.time() - start:.4f}s]")

    # summarize per-modality, per-class dice
    buffer = dice_metric.get_buffer()
    if isinstance(buffer, torch.Tensor):
        buffer = buffer.detach().cpu()  # shape: [N, C]
    else:
        logger.warning("Dice buffer is empty; nothing to report.")
        return

    if len(all_modalities) != buffer.shape[0]:
        logger.warning(
            f"Modality count ({len(all_modalities)}) != samples in dice buffer ({buffer.shape[0]})."
        )

    unique_modalities = sorted(set(all_modalities))
    modality_names = getattr(cfg, "modalities", None)
    class_names = getattr(cfg, "classes", None)
    include_bg = cfg.include_background

    def _get_class_label(ci: int) -> str:
        if class_names is None:
            return str(ci)
        base_idx = ci + (1 if not include_bg else 0)
        if 0 <= base_idx < len(class_names):
            return str(class_names[base_idx])
        return str(ci)

    for m in unique_modalities:
        idxs = [i for i, v in enumerate(all_modalities) if v == m and i < buffer.shape[0]]
        if not idxs:
            continue
        vals = buffer[idxs]  # [n_i, C]
        logger.info(
            f"Per-modality Dice ({modality_names[m] if modality_names is not None and m >= 0 and m < len(modality_names) else m}):"
        )
        for c in range(vals.shape[1]):
            vc = vals[:, c]
            not_nan = ~torch.isnan(vc)
            mean_c = torch.where(not_nan.any(), vc[not_nan].mean(), torch.tensor(float('nan'))).item()
            logger.info(f"  - Class { _get_class_label(c) }: {mean_c:.4f}")

    mean_dice = torch.nanmean(buffer).item()
    logger.info(f"Overall Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="UNet Testing")
    parser.add_argument("--experiment", type=str, help="Path to the experiment directory")
    parser.add_argument("--weight", type=str, help="Checkpoint filename under experiment/model/")
    args = parser.parse_args()

    with open(os.path.join(args.experiment, "cfg.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    cfg = Namespace(**cfg)
    cfg.experiment = args.experiment
    cfg.weight = args.weight
    cfg.distributed = False

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    main(cfg)
