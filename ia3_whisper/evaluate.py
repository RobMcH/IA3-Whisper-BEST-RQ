"""Evaluation script for testing fine-tuned IA3Whisper models."""
from __future__ import annotations

import torch

from ia3_whisper.dataset import get_dataloader
from ia3_whisper.log import get_logger, log_metrics
from ia3_whisper.masking import BestRQMasking
from ia3_whisper.model import IA3AudioEncoder, IA3Whisper
from ia3_whisper.utils import compute_cross_entropy_loss, get_compute_dtype

logger = get_logger(__name__)


def evaluate(
    model: IA3Whisper,
    best_rq: BestRQMasking,
    batch_size: int,
    device: str,
    use_wandb: bool,
    validation_dataset: str,
    use_mixed_precision: bool,
) -> float:
    """Evaluate a given model by BEST-RQ loss on a held-out dataset.

    :param model: A trained IA3Whisper model to evaluate.
    :param best_rq: A BestRQMasking object used to obtain the masked features and corresponding targets.
    :param batch_size: The batch size used for evaluation.
    :param device: The device to load the model onto.
    :param use_wandb: Whether to log metrics to wandb.
    :param validation_dataset: Which LibriSpeech data split to use for evaluation.
    :param use_mixed_precision: Whether to enable mixed precision inference.
    :return: The computed validation loss.
    """
    logger.info("Evaluating model on %s", validation_dataset)
    dataloader = get_dataloader(validation_dataset, batch_size, False, device)
    dtype: torch.dtype = get_compute_dtype(use_mixed_precision)
    validation_loss = 0.0
    for batch in dataloader:
        batch = best_rq.get_targets_and_features(batch)
        with torch.autocast(device_type=device, dtype=dtype):
            loss = evaluate_step(batch, model.encoder)
        validation_loss += loss
    validation_loss /= float(len(dataloader.dataset))  # type: ignore
    log_metrics(0, 0, {"loss": validation_loss}, use_wandb, logger)
    return validation_loss


def evaluate_step(
    batch: dict[str, torch.Tensor],
    model: IA3AudioEncoder,
) -> float:
    """Perform one BEST-RQ evaluation step.

    :param batch: A dictionary holding the batch. Must have keys 'in_feats' and 'targets'.
    :param model: The model to use for the forward pass.
    :return: A float holding the computed loss.
    """
    model.eval()
    with torch.no_grad():
        _, logits = model(batch["in_feats"])
        # Cast logits to full precision.
        logits = logits[:, batch["mask"]].float()
        loss = compute_cross_entropy_loss(logits, batch["targets"])
    return loss.float().detach().cpu().item()
