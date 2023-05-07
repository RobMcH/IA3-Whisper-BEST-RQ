import torch


def compute_cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Computes the cross entropy loss between the given logits and targets.

    :param logits: The computed logits. Shape: (num_masked, num_targets).
    :param target: The target tensor. Shape: (num_masked, 1).
    :return: The computed loss tensor.
    """
    return torch.nn.functional.cross_entropy(input=logits, target=target.squeeze())
