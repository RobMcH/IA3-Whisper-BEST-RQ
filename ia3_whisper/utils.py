"""Contains a collection of utility functions."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import wandb

from ia3_whisper import IA3Whisper, load_model


def compute_cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compute the cross entropy loss between the given logits and targets.

    :param logits: The computed logits. Shape: (num_codebooks, num_masked, num_targets).
    :param target: The target tensor. Shape: (num_codebooks, num_masked).
    :return: The computed loss tensor.
    """
    if logits.shape[:-1] != target.shape:
        raise ValueError(
            f"Logits and targets need to have matching shapes in the leading dimensions."
            f" Got logits: {logits.shape} and target: {target.shape}."
        )
    num_targets = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(
        input=logits.reshape(-1, num_targets), target=target.reshape(-1)
    )
    return loss


def get_ia3_model(
    model_name: str, device: str, num_targets: int, num_codebooks: int
) -> IA3Whisper:
    """Load an IA3 Whisper model, freeze its weights, add codebook classifiers and unfreeze the IA3 weights.

    :param model_name: The name of the pre-trained Whisper model to load.
    :param device: The device to put the model on.
    :param num_targets: The number of targets per codebooks.
    :param num_codebooks: The numbber of codebooks/classifiers to use.
    :return: The loaded model.
    """
    model = load_model(model_name, device)
    model.freeze()
    model.add_codebook_classifiers(num_codebooks, num_targets)
    model.unfreeze_ia3()
    return model


def get_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    warmup_init_lr: float,
    warmup_end_lr: float,
    warmup_steps: int,
    use_lr_scheduler: bool = True,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Get an Adam optimizer and Transformer learning rate scheduler.

    If use_lr_scheduler is set to False, a dummy scheduler is returned which provides a constant learning rate.

    :param parameters: The parameters of the model to optimize.
    :param warmup_init_lr: The initial warmup learning rate.
    :param warmup_end_lr: The final warmup learning rate.
    :param warmup_steps: The number of warmup steps.
    :param use_lr_scheduler: Whether to return a true or dummy learning rate scheduler.
    :return: A tuple containing the optimizer and the learning rate scheduler.
    """
    warmup_init_lr = min(warmup_init_lr, warmup_end_lr)
    linear_lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
    decay_factor = warmup_end_lr * warmup_steps**2
    if use_lr_scheduler:
        optimizer = torch.optim.Adam(parameters, warmup_init_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: lr_update(
                num_updates=x,
                warmup_steps=warmup_steps,
                warmup_init_lr=warmup_init_lr,
                lr_step=linear_lr_step,
                decay_factor=decay_factor,
            ),
        )
    else:
        optimizer = torch.optim.Adam(parameters, warmup_end_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )

    return optimizer, lr_scheduler


def lr_update(
    num_updates: int,
    warmup_steps: int,
    warmup_init_lr: float,
    lr_step: float,
    decay_factor: float,
) -> float:
    """Implement an InverseSquareRootSchedule as in fairseq.

    C.f. https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py#L32

    :param num_updates: The number of performed updates.
    :param warmup_steps: The number of warmup steps.
    :param warmup_init_lr: The initial learning rate at the beginning of warmup.
    :param lr_step: The step size when increasing the learning rate.
    :param decay_factor: The factor for decaying the learnign rate after warmum.
    :return: The calculated learning rate at a given step.
    """
    if num_updates < warmup_steps:
        lr = warmup_init_lr + num_updates * lr_step
    else:
        lr = decay_factor * num_updates**2
    if warmup_init_lr > 0:
        return lr / warmup_init_lr
    return 0.0


def upload_to_wandb(use_wandb: bool, path: Path) -> None:
    """Upload the weights found at path to wandb.

    :param use_wandb: Whether to upload to wandb.
    :param path: The path to the weights to upload.
    """
    if use_wandb:
        # Upload IA3 weights to wandb.
        artifact = wandb.Artifact("ia3_encoder_weights", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)


def set_seed(seed: int) -> None:
    """Set the random seeds of the various libraries responsible for randomness.

    :param seed: The seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_compute_dtype(use_mixed_precision: bool) -> torch.dtype:
    """Get the compute dtype depending on whether to use mixed precision and the underlying hardware.

    :param use_mixed_precision: Whether to use half precision compute dtypes.
    :return: A torch.dtype.
    """
    if not use_mixed_precision:
        return torch.get_default_dtype()
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
