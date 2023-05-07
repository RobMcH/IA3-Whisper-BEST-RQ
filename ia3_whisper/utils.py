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


def get_ia3_model(model_name: str, device: str, num_targets: int) -> IA3Whisper:
    """Loads an IA3 Whisper model, freezes its weights, adds codebook classifiers and unfreezes the IA3 weights.

    :param model_name: The name of the pre-trained Whisper model to load.
    :param device: The device to put the model on.
    :param num_targets: The number of targets per codebooks.
    :return: The loaded model.
    """
    model = load_model(model_name, device)
    model.freeze()
    model.add_codebook_classifiers(1, num_targets)
    model.unfreeze_encoder_ia3()
    return model
