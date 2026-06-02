import time
import torch
import logging
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage
from monai.data.meta_tensor import MetaTensor
from torch.amp import autocast
from utils.image_randomizer import apply_moa

logger = logging.getLogger(__name__)

def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        scaler,
        amp=True,
        iters_to_accumulate=1,
        logger=None,
        labeled_classes=None,
        out_channels=8,
        worst_dice_label=None,        # worst performing organ label (int)
        organ_id_dict=None,           # class dictionary from config, e.g. {0: BG, 1: spleen, ...}
        current_epoch=1,              # current epoch number

):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    run_loss = CumulativeAverage()

    # Add consistency loss function
    consistency_criterion = torch.nn.MSELoss()
    lambda_consistency = 0.1  # consistency loss weight

    optimizer.zero_grad(set_to_none=True)
    for idx, batch in enumerate(loader):
        start_time = time.time()
        if torch.cuda.is_available():
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda(non_blocking=True)

        if "label" in batch and isinstance(batch["label"], MetaTensor):
            batch["label"] = batch["label"].as_tensor()

        if "label" in batch and batch["label"].max().item() >= out_channels:
            logger.warning(f"Batch {idx}: Label contains values >= out_channels {out_channels}")
            batch["label"] = torch.clamp(batch["label"], 0, out_channels - 1)

        # with autocast(enabled=amp, device_type=device):
        #     output = model(batch["image"], batch.get("modality", None))
        #     if output.shape[1] != out_channels:
        #         logger.error(f"Output channels mismatch: expected {out_channels}, got {output.shape[1]}")
        #         raise ValueError("Output channels mismatch")
        #
        #     loss = criterion(output, batch["label"]) / iters_to_accumulate
        #     run_loss.append(loss)
        # if amp:
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     loss.backward()
        #     optimizer.step()
        #
        # optimizer.zero_grad(set_to_none=True)
        #
        # if logger:
        #     logger.info(f"Train: [{idx + 1}/{len(loader)}] Batch {(time.time() - start_time):.4f} loss: {loss.item():.4f}")

        with autocast(enabled=amp, device_type=device):
            # Get original images and labels
            images = batch["image"]
            labels = batch["label"]

            # Store all inputs and original outputs
            all_inputs = []
            all_outputs = []

            for i in range(images.shape[0]):
                # === Define augmentation strategy ===
                fixed_labels = [3]  # default: augment left kidney
                candidate_labels = [k for k in organ_id_dict.keys() if k != 0 and k != 3]
                # after 500 epochs, choose worst_label
                if current_epoch > 500 and worst_dice_label is not None:
                    fixed_labels = [worst_dice_label]
                    candidate_labels = [k for k in organ_id_dict.keys() if k != 0 and k != worst_dice_label]

                # Generate augmented versions (including original image)
                # For images with only background, automatically skips organ-specific augmentation
                aug_images = apply_moa(
                    images[i].cpu().numpy(),  # ensure processing on CPU
                    labels[i].cpu().numpy(),
                    fixed_labels=fixed_labels,
                    candidate_labels=candidate_labels
                )

                # Convert to tensor and store
                for aug_img in aug_images:
                    input_tensor = torch.tensor(aug_img, dtype=torch.float32).unsqueeze(0).to(device)
                    all_inputs.append(input_tensor)

            # Batch process all inputs (original + augmented)
            all_inputs = torch.cat(all_inputs, dim=0)
            all_outputs = model(all_inputs, batch.get("modality", None))

            # Separate original and augmented outputs
            num_versions = len(aug_images)  # number of versions per sample
            batch_size = images.shape[0]

            # Reshape outputs: [original1, aug1_1, aug1_2, ..., original2, aug2_1, ...]
            outputs_reshaped = all_outputs.view(batch_size, num_versions, *all_outputs.shape[1:])

            # Compute supervised loss (using only original image outputs)
            original_outputs = outputs_reshaped[:, 0]
            supervised_loss = criterion(original_outputs, labels)

            # Compute consistency loss (between all augmented versions and original)
            consistency_loss = 0.0
            for i in range(1, num_versions):
                # Compute MSE between each augmented version and original output
                aug_output = outputs_reshaped[:, i]
                consistency_loss += consistency_criterion(aug_output, original_outputs.detach())

            # Average consistency loss
            consistency_loss /= (num_versions - 1)

            # Total loss = supervised loss + consistency loss
            total_loss = supervised_loss + lambda_consistency * consistency_loss
            total_loss = total_loss / iters_to_accumulate
            run_loss.append(total_loss)

        if amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if logger:
            logger.info(f"Train: [{idx + 1}/{len(loader)}] Batch {(time.time() - start_time):.4f} "
                        f"loss: {total_loss.item():.4f} "
                        f"(sup: {supervised_loss.item():.4f}, "
                        f"cons: {consistency_loss.item():.4f})")

    epoch_loss = run_loss.aggregate().item()
    run_loss.reset()
    return epoch_loss

def val_epoch(
        model,
        loader,
        criterion,
        post_label,
        post_pred,
        metrics,
        model_inferer=None,
        amp=True,
        logger=None,
        modalities=None,
        labeled_classes=None,
        out_channels=8,
):
    device = next(model.parameters()).device
    model.eval()
    run_loss = CumulativeAverage()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            start_time = time.time()
            if torch.cuda.is_available():
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)

            if "label" in batch and isinstance(batch["label"], MetaTensor):
                batch["label"] = batch["label"].as_tensor()

            if "label" in batch and batch["label"].max().item() >= out_channels:
                logger.warning(f"Val batch {idx}: Label contains values >= out_channels {out_channels}")
                batch["label"] = torch.clamp(batch["label"], 0, out_channels - 1)

            with autocast(enabled=amp, device_type="cuda" if device.type == "cuda" else "cpu"):
                if model_inferer is not None:
                    output = model_inferer(batch["image"], modalities=batch.get("modality", None))
                else:
                    output = model(batch["image"], modalities=batch.get("modality", None))

            output = output.to(device)

            if output.shape[1] != out_channels:
                logger.error(f"Output channels mismatch: expected {out_channels}, got {output.shape[1]}")
                raise ValueError("Output channels mismatch")

            loss = criterion(output, batch["label"])
            run_loss.append(loss)

            val_labels_list = decollate_batch(batch["label"])
            val_labels_convert = [post_label(t) for t in val_labels_list]
            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(t) for t in val_outputs_list]

            for metric in metrics:
                metric(y_pred=val_output_convert, y=val_labels_convert)

            if logger:
                logger.info(f"Val: [{idx + 1}/{len(loader)}] Loss {loss:.4f} [{time.time() - start_time:.4f}s]")

    epoch_loss = run_loss.aggregate().item()
    run_loss.reset()
    return epoch_loss