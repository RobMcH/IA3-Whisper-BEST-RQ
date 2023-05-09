"""Contains the main logic to train a Whisper-like model using IA3 and BEST-RQ."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch.cuda
import wandb
from whisper import _MODELS

from ia3_whisper.dataset import get_dataloader
from ia3_whisper.log import get_logger
from ia3_whisper.masking import BestRQMasking
from ia3_whisper.model import IA3AudioEncoder, IA3Whisper
from ia3_whisper.utils import (
    compute_cross_entropy_loss,
    get_ia3_model,
    get_optimizer,
    set_seed,
    upload_to_wandb,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments and wrap them in an argparse.Namespace.

    :return: An argparse.Namespace object holding the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model hyper-parameters.
    parser.add_argument(
        "--model_name",
        type=str,
        default="tiny",
        choices=_MODELS.keys(),
        help="The pre-trained Whisper checkpoint to load.",
    )
    # BEST-RQ hyper-parameters.
    parser.add_argument(
        "--num_targets",
        type=int,
        default=8192,
        help="The number of targets per codebook.",
    )
    parser.add_argument(
        "--num_codebooks", type=int, default=1, help="The number of codebooks."
    )
    parser.add_argument(
        "--codebook_dim",
        type=int,
        default=16,
        help="The dimension of the codebook vectors.",
    )
    parser.add_argument(
        "--masking_prob",
        type=float,
        default=0.01,
        help="The probability of masking a span of tokens.",
    )
    parser.add_argument(
        "--temporal_reduction",
        type=int,
        default=2,
        help="The fold of temporal downsampling performed by the encoder.",
    )
    # Optimizer/training hyper-parameters.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="The batch size used for BEST-RQ training.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="The number of epochs to train for."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.004,
        help="The peak learning rate to use by the Transformer learning rate scheduler.",
    )
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="The number of batches to accumulate before updating the parameters.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=25000,
        help="The number of warmup steps to use by the Transformer learning rate scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to use for all random number generators.",
    )
    # Compute device.
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def train(
    model: IA3Whisper,
    best_rq: BestRQMasking,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch_size: int,
    accumulate_gradients: int,
    num_epochs: int,
    device: str,
    use_wandb: bool,
    output_path: Path,
) -> None:
    """Run BEST-RQ training on the encoder of the given IA3Whisper model.

    :param model: A (pre-trained) IA3Whisper model. Only the IA3 weights of the AudioEncoder will be fine-tuned.
    :param best_rq: A BestRQMasking object used to obtain the masked features and corresponding targets.
    :param optimizer: An optimizer to update the weights with.
    :param lr_scheduler: A learning rate scheduler.
    :param batch_size: The batch size used for training.
    :param accumulate_gradients: An integer denoting the number of batches used per gradient update.
    :param num_epochs: The number of epochs to train for.
    :param device: The device to load the model onto.
    :param use_wandb: Whether to log metrics to wandb.
    :param output_path: The output path where to store the trained IA3 weights.
    """
    dataloader = get_dataloader("train-clean-100", batch_size, True, device)
    for epoch in range(1, num_epochs + 1):
        for i, batch in enumerate(dataloader):
            loss, metrics = train_step(batch, model.encoder, best_rq)
            metrics["lr"] = lr_scheduler.get_last_lr()[0]
            update_weights(i, accumulate_gradients, loss, optimizer, lr_scheduler)
            log_metrics(i, epoch, metrics, use_wandb)
        # Store epoch IA3 weights and upload to wandb.
        path = Path(f"{output_path}_epoch_{epoch}").with_suffix(".pt")
        model.save_ia3_encoder(path)
        upload_to_wandb(use_wandb, path)
    # Store final IA3 weights and upload to wandb.
    output_path = output_path.with_suffix(".pt")
    logger.info("Saving trained IA3 weights to %s.", str(output_path))
    model.save_ia3_encoder(output_path)
    upload_to_wandb(use_wandb, output_path)


def train_step(
    batch: dict[str, torch.Tensor],
    model: IA3AudioEncoder,
    best_rq: BestRQMasking,
) -> tuple[torch.Tensor, dict]:
    """Perform one BEST-RQ training step.

    :param batch: A dictionary holding the batch. Must have key 'in_feats'.
    :param model: The model to use for the forward pass.
    :param best_rq: A BestRQMasking object to obtain the masked features and targets.
    :return: A tuple containing:
        * loss: The computed loss tensor.
        * metrics: A dictionary holding a float representation of the loss ('loss'), the number of unique targets in
         the batch ('unique_targets'), and the number of total targets in the batch ('targets').
    """
    batch = best_rq.get_targets_and_features(batch)
    _, logits = model(batch["in_feats"])
    logits = logits[:, batch["mask"]]
    loss = compute_cross_entropy_loss(logits, batch["targets"])

    unique_targets = batch["targets"].unique().nelement()
    return loss, {
        "loss": loss.cpu().item(),
        "unique_targets": unique_targets,
        "targets": batch["targets"].nelement(),
    }


def update_weights(
    batch_idx: int,
    accumulate_gradients: int,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> None:
    """Update the model weights attached to the given optimizer.

    Performs gradient accumulation if accumulate_gradients > 1.

    :param batch_idx: The index of the batch within the epoch.
    :param accumulate_gradients: Integer specifying how many batches gradients are being accumulated (averaged) over.
    :param loss: The computed loss.
    :param optimizer: The optimizer to perform the weight update with.
    :param lr_scheduler: An LR scheduler; updates the learning rate every accumulate_gradients steps.
    """
    loss = loss / accumulate_gradients
    loss.backward()
    if (batch_idx + 1) % accumulate_gradients == 0:
        logger.debug(
            "Updating weights after accumulating gradients over %d batches.",
            accumulate_gradients,
        )
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


def log_metrics(batch_idx: int, epoch: int, metrics: dict, use_wandb: bool) -> None:
    """Log the given metrics using the logger as well as wandb.

    Only logs using the logger every 50 batches to not pollute the logs.

    :param batch_idx: The index of the batch within the epoch.
    :param epoch: The epoch of the training.
    :param metrics: A dictionary holding the metrics to log. Keys 'loss', 'unique_targets' and 'targets' will be
     logged using the logger. Uses -1.0 as a default value when a key is not present.
    :param use_wandb: Whether to log the metrics to wandb.
    """
    if use_wandb:
        wandb.log(metrics)
    if batch_idx % 50 == 1:
        logger.info(
            "Epoch %d - Batch %d - Loss %.5f - #Unique targets %d / %d",
            epoch,
            batch_idx,
            metrics.get("loss", -1.0),
            metrics.get("unique_targets", -1.0),
            metrics.get("targets", -1.0),
        )


def main():
    """Train a Whisper checkpoint using IA3 and BEST-RQ."""
    args = parse_args()
    set_seed(args.seed)
    logger.info("Loading model checkpoint %s", args.model_name)
    model = get_ia3_model(
        args.model_name, args.device, args.num_targets, args.num_codebooks
    )
    optimizer, lr_scheduler = get_optimizer(
        model.parameters(),
        warmup_end_lr=args.learning_rate,
        warmup_init_lr=args.learning_rate / 10,
        warmup_steps=args.warmup_steps,
    )
    best_rq = BestRQMasking(
        num_targets=args.num_targets,
        num_codebooks=args.num_codebooks,
        emb_dim=80,
        codebook_dim=args.codebook_dim,
        masking_prob=args.masking_prob,
        temporal_reduction=args.temporal_reduction,
        device=args.device,
        seed=args.seed,
    )
    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        wandb.init(project="ia3whisper", config=vars(args))
    output_path = Path(f"{args.model_name}_IA3_{int(time.time())}")
    train(
        model,
        best_rq,
        optimizer,
        lr_scheduler,
        args.batch_size,
        args.accumulate_gradients,
        args.num_epochs,
        args.device,
        use_wandb,
        output_path,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
