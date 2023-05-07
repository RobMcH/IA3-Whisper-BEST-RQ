from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch.cuda
from whisper import _MODELS

from ia3_whisper.dataset import get_dataloader
from ia3_whisper.log import get_logger
from ia3_whisper.masking import BestRQMasking
from ia3_whisper.model import IA3AudioEncoder, IA3Whisper
from ia3_whisper.utils import compute_cross_entropy_loss, get_ia3_model, get_optimizer

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
    output_path: Path,
):
    dataloader = get_dataloader("test-clean", batch_size, True, device)
    for epoch in range(1, num_epochs + 1):
        for i, batch in enumerate(dataloader):
            loss, metrics = train_step(batch, model.encoder, best_rq)
            update_weights(i, accumulate_gradients, loss, optimizer, lr_scheduler)
            log_metrics(i, epoch, metrics)

    logger.info("Saving trained IA3 weights to %s.", str(output_path))
    model.save_ia3_encoder(output_path)


def train_step(
    batch: dict[str, torch.Tensor],
    model: IA3AudioEncoder,
    best_rq: BestRQMasking,
) -> tuple[torch.Tensor, dict]:
    batch = best_rq.get_targets_and_features(batch)
    _, logits = model(batch["in_feats"])
    logits = logits[0, batch["mask"]]
    loss = compute_cross_entropy_loss(logits, batch["targets"])

    unique_targets = batch["targets"].unique().size(0)
    return loss, {
        "loss": loss.cpu().item(),
        "unique_targets": unique_targets,
        "targets": batch["targets"].size(0),
    }


def update_weights(
    batch_idx: int,
    accumulate_gradients: int,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> None:
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


def log_metrics(batch_idx: int, epoch: int, metrics: dict) -> None:
    if batch_idx % 50 == 1:
        logger.info(
            "Epoch %d - Batch %d - Loss %.5f - #Unique targets %d / %d",
            epoch,
            batch_idx,
            metrics["loss"],
            metrics["unique_targets"],
            metrics["targets"],
        )


def main():
    args = parse_args()
    logger.info("Loading model checkpoint %s", args.model_name)
    model = get_ia3_model(args.model_name, args.device, args.num_targets)
    optimizer, lr_scheduler = get_optimizer(
        model.parameters(),
        warmup_end_lr=args.learning_rate,
        warmup_init_lr=args.learning_rate / 10,
        warmup_steps=args.warmup_steps,
    )
    best_rq = BestRQMasking(
        num_targets=args.num_targets,
        emb_dim=80,
        codebook_dim=args.codebook_dim,
        masking_prob=args.masking_prob,
        temporal_reduction=args.temporal_reduction,
        device=args.device,
        seed=args.seed,
    )

    output_path = Path(f"{args.model_name}_IA3_{int(time.time())}.pt")
    train(
        model,
        best_rq,
        optimizer,
        lr_scheduler,
        args.batch_size,
        args.accumulate_gradients,
        args.num_epochs,
        args.device,
        output_path,
    )


if __name__ == "__main__":
    main()
