import os
import torch
import yaml
import torch.distributed as dist
from tensorboardX import SummaryWriter
from utils.launch import launch
from utils.logger import get_root_logger
from utils import comm
from functools import partial
from argparse import ArgumentParser, Namespace
from torch.cuda.amp import GradScaler
from data.multi_modal import get_loaders
from utils.trainer import train_epoch, val_epoch
from monai.inferers import sliding_window_inference
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.training_utils import loss_from_cfg, optimizer_from_cfg, scheduler_from_cfg
from monai.transforms import AsDiscrete
from monai.metrics.meandice import DiceMetric
from monai.metrics import GeneralizedDiceScore, Cumulative
from networks.utils.misc import copy_model_state, model_from_cfg
import logging


def build_writer(save_path, logger):
    logger.info("=> Building writer ...")
    writer = SummaryWriter(save_path) if comm.is_main_process() else None
    logger.info(f"Tensorboard writer logging dir: {save_path}")
    return writer


def save_checkpoint(model, epoch, logdir, filename="model.pt", best=0, optimizer=None, scheduler=None, scaler=None,
                    logger=None):
    state_dict = model.state_dict() if not comm.get_world_size() > 1 else model.module.state_dict()
    save_dict = {"epoch": epoch, "best": best, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            save_dict["scheduler"] = scheduler.state_dict()
        except Exception:
            save_dict["scheduler"] = None
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()
    filename = os.path.join(logdir, "model", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(save_dict, filename)
    if logger is not None:
        logger.info(f"Saving checkpoint {filename}")


def _find_checkpoint_path(resume_arg):
    if resume_arg is None:
        return None
    if os.path.isfile(resume_arg):
        return resume_arg
    candidate = os.path.join(resume_arg, "model", "latest.pt")
    if os.path.isfile(candidate):
        return candidate
    candidate_best = os.path.join(resume_arg, "model", "best.pt")
    if os.path.isfile(candidate_best):
        return candidate_best
    return None


def load_checkpoint_for_resume(model, optimizer, scheduler, scaler, resume_arg, logger):
    ckpt_path = _find_checkpoint_path(resume_arg)
    if ckpt_path is None:
        logger.warning(f"No checkpoint found for resume at '{resume_arg}'. Starting from scratch.")
        return {"start_epoch": 1, "best": -float("inf"), "epochs_no_improve": 0, "best_epoch": 0}
    logger.info(f"=> Resuming training from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    state = ckpt.get("state_dict", ckpt)
    try:
        if isinstance(model, DDP):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
    except Exception as e:
        logger.error(f"Failed to load state_dict into model: {e}")
        raise

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}")

    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            logger.warning(f"Could not load scaler state: {e}")

    start_epoch = ckpt.get("epoch", 0) + 1
    best = ckpt.get("best", -float("inf"))
    epochs_no_improve = ckpt.get("epochs_no_improve", 0)
    best_epoch = ckpt.get("best_epoch", 0)

    logger.info(
        f"=> Resume successful. start_epoch={start_epoch}, best={best}, epochs_no_improve={epochs_no_improve}, best_epoch={best_epoch}")
    return {"start_epoch": start_epoch, "best": best, "epochs_no_improve": epochs_no_improve, "best_epoch": best_epoch}


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_logdir = os.path.join(cfg.default_root_dir, cfg.study_name, cfg.trial_name)
    os.makedirs(model_logdir, exist_ok=True)

    if comm.is_main_process():
        with open(os.path.join(model_logdir, "cfg.yaml"), "w") as f:
            yaml.dump(cfg.__dict__, f)

    logger = get_root_logger(
        log_file=os.path.join(model_logdir, "train.log"),
        file_mode="a" if hasattr(cfg, 'resume') and cfg.resume else "w",
    )
    writer = build_writer(model_logdir, logger)

    logger.info("=> Building model ...")
    model_cfg = {k: v for k, v in cfg.__dict__.items() if k != "img_size" or cfg.model != "ME-UNet"}
    model = model_from_cfg(Namespace(**model_cfg)).to(device=device)
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if hasattr(cfg, 'pretrained') and cfg.pretrained:
        logger.info(f"=> Loading pretrained model: {cfg.pretrained} ...")
        pretrained = torch.load(cfg.pretrained, map_location=device)
        if 'state_dict' in pretrained:
            pretrained = pretrained['state_dict']
        copy_model_state(model, pretrained, filter_func=None)

    if comm.get_world_size() > 1:
        model = DDP(model, device_ids=[comm.get_local_rank()], output_device=comm.get_local_rank())

    if getattr(cfg, 'use_mutation_loss', False):
        from utils.losses import mutation_loss
        criterion = mutation_loss(cfg.out_channels, lambda_consistency=0.5)
        logger.info("Using MUTATION loss with multiple decoder outputs.")
    else:
        criterion = loss_from_cfg(cfg)

    optimizer = optimizer_from_cfg(cfg, model)
    scheduler = scheduler_from_cfg(cfg, optimizer)
    scaler = GradScaler() if getattr(cfg, 'amp', False) else None

    start_epoch = 1
    best = -float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    if hasattr(cfg, 'resume') and cfg.resume:
        resume_info = load_checkpoint_for_resume(model, optimizer, scheduler, scaler, cfg.resume, logger)
        start_epoch = resume_info.get("start_epoch", 1)
        best = resume_info.get("best", -float("inf"))
        epochs_no_improve = resume_info.get("epochs_no_improve", 0)
        best_epoch = resume_info.get("best_epoch", 0)

    train_loader, val_loader = get_loaders(cfg)

    metrics = (
        DiceMetric(include_background=cfg.include_background, reduction='mean_batch'),
        GeneralizedDiceScore(reduction="mean_batch", include_background=cfg.include_background),
    )
    modalities = Cumulative()

    post_label = AsDiscrete(to_onehot=cfg.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.out_channels)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
        sw_batch_size=getattr(cfg, 'sw_batch_size', 1),
        predictor=model,
        overlap=getattr(cfg, 'infer_overlap', 0.5),
    )

    patience = getattr(cfg, 'early_stop_patience', 20)
    min_delta = getattr(cfg, 'early_stop_min_delta', 0.0005)
    min_epochs = getattr(cfg, 'early_stop_min_epochs', 2000)
    max_epochs_cap = cfg.epochs

    logger.info(f"Training starts from epoch {start_epoch}, max epoch {cfg.epochs}, best={best}")

    worst_dice_label = None

    for epoch in range(start_epoch, cfg.epochs + 1):

        logger.info(f"Epoch [{epoch}/{cfg.epochs}] Starting...")

        train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            amp=getattr(cfg, 'amp', False),
            logger=logger,
            out_channels=cfg.out_channels,
            worst_dice_label=worst_dice_label,
            organ_id_dict=cfg.classes,
            current_epoch=epoch
        )

        try:
            if hasattr(scheduler, "step"):
                scheduler.step()
        except Exception as e:
            logger.warning(f"Scheduler step failed: {e}")

        if epoch <= 2000:
            should_validate = (epoch % 50 == 0)
        else:
            should_validate = (epoch % 5 == 0)

        if should_validate:

            val_loss = val_epoch(
                model,
                val_loader,
                criterion,
                post_label=post_label,
                post_pred=post_pred,
                model_inferer=model_inferer,
                logger=logger,
                metrics=metrics,
                modalities=modalities,
                out_channels=cfg.out_channels,
            )

            logger.info(f"Epoch [{epoch}] Validation Loss: {val_loss:.4f}")

            for metric in metrics:
                try:
                    logger.info(f"Epoch [{epoch}] Metric {metric.__class__.__name__}: {metric.aggregate()}")
                except Exception:
                    logger.info(f"Epoch [{epoch}] Metric {metric.__class__.__name__}: (aggregate failed)")

            if epoch > 500:
                try:
                    dice_scores = metrics[0].aggregate()
                    if dice_scores is not None:
                        dice_scores = dice_scores.detach().cpu().numpy()
                        expected_len = len(cfg.classes) - 1 if cfg.include_background is False else len(cfg.classes)

                        if not cfg.include_background:
                            if dice_scores.shape[0] == expected_len:
                                offset = 1
                            elif dice_scores.shape[0] == len(cfg.classes):
                                dice_scores = dice_scores[1:]
                                offset = 1
                            else:
                                logger.warning(
                                    f"[Epoch {epoch}] Dice scores length ({dice_scores.shape[0]}) "
                                    f"unexpected for {len(cfg.classes)} total classes."
                                )
                                offset = 1
                        else:
                            offset = 0

                        worst_dice_label = int(dice_scores.argmin()) + offset
                        logger.info(f"Epoch {epoch}: Worst Dice label = {worst_dice_label}")
                        logger.info(f"Dice scores shape: {dice_scores.shape}, values: {dice_scores}")
                    else:
                        logger.warning(
                            f"[Epoch {epoch}] Could not compute worst_dice_label: "
                            f"the data to aggregate must be PyTorch Tensor, got <class 'NoneType'>."
                        )
                except Exception as e:
                    logger.warning(
                        f"[Epoch {epoch}] Could not compute worst_dice_label: {e}"
                    )

            if comm.is_main_process():

                try:
                    current = metrics[0].aggregate().mean().item()
                except Exception:
                    current = -float("inf")

                improved = current > (best + min_delta)

                save_checkpoint(
                    model, epoch, model_logdir, filename="latest.pt", best=best,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler, logger=logger
                )

                if improved:
                    best = current
                    best_epoch = epoch
                    epochs_no_improve = 0
                    save_checkpoint(
                        model, epoch, model_logdir, filename="best.pt", best=best,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler, logger=logger
                    )
                    logger.info(f"New best metric: {best:.6f} at epoch {best_epoch}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement this validation. epochs_no_improve = {epochs_no_improve}")

                try:
                    latest_ckpt_path = os.path.join(model_logdir, "model", "latest.pt")
                    latest = torch.load(latest_ckpt_path, map_location=device)
                    latest["epochs_no_improve"] = epochs_no_improve
                    latest["best_epoch"] = best_epoch
                    torch.save(latest, latest_ckpt_path)
                except Exception as e:
                    logger.warning(f"Could not update latest.pt with epochs_no_improve: {e}")

            for metric in metrics:
                try:
                    metric.reset()
                except Exception:
                    pass

            if epoch >= min_epochs and epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best Dice {best:.5f} at epoch {best_epoch}."
                )
                break

        if epoch >= max_epochs_cap:
            logger.info(
                f"Reached max_epochs_cap {max_epochs_cap}. Best Dice {best:.5f} at epoch {best_epoch}."
            )
            break

    if comm.is_main_process() and writer is not None:
        writer.close()
    if comm.get_world_size() > 1:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == '__main__':
    parser = ArgumentParser(description="Load configuration file.")
    parser.add_argument("--config", type=str, help="Path to the configuration file (YAML).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs per machine.")
    parser.add_argument("--num_machines", type=int, default=1, help="Number of machines.")
    parser.add_argument("--machine_rank", type=int, default=0, help="Local rank for distributed training.")
    parser.add_argument("--dist_url", type=str, default="auto", help="URL used to set up distributed training.")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment or checkpoint to resume.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Namespace(**cfg_dict)

    if not hasattr(cfg, 'test_mode'):
        cfg.test_mode = False

    cfg.resume = args.resume

    launch(main, cfg, num_gpus_per_machine=args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url)