from __future__ import annotations

from typing import Iterator

import torch

from ia3_whisper import IA3Whisper, load_model


def compute_cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Computes the cross entropy loss between the given logits and targets.

    :param logits: The computed logits. Shape: (num_masked, num_targets).
    :param target: The target tensor. Shape: (num_masked, 1).
    :return: The computed loss tensor.
    """
    return torch.nn.functional.cross_entropy(input=logits, target=target.squeeze())


def get_ia3_model(
    model_name: str, device: str, num_targets: int, num_codebooks: int
) -> IA3Whisper:
    """Loads an IA3 Whisper model, freezes its weights, adds codebook classifiers and unfreezes the IA3 weights.

    :param model_name: The name of the pre-trained Whisper model to load.
    :param device: The device to put the model on.
    :param num_targets: The number of targets per codebooks.
    :param num_codebooks: The numbber of codebooks/classifiers to use.
    :return: The loaded model.
    """
    model = load_model(model_name, device)
    model.freeze()
    model.add_codebook_classifiers(num_codebooks, num_targets)
    model.unfreeze_encoder_ia3()
    return model


def get_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    warmup_init_lr: float,
    warmup_end_lr: float,
    warmup_steps: int,
    use_lr_scheduler: bool = True,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
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
    """Implements an InverseSquareRootSchedule as in fairseq.

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
