"""Implement IA3-based training of Whisper-like models using BEST-RQ."""

from __future__ import annotations

import io
from pathlib import Path

import torch
from whisper import (
    _ALIGNMENT_HEADS,
    _MODELS,
    ModelDimensions,
    _download,
    available_models,
)

from ia3_whisper.model import IA3Whisper


def load_model(
    name: str,
    device: str | torch.device | None = None,
    download_root: Path | str | None = None,
    in_memory: bool = False,
) -> IA3Whisper:
    """Load a Whisper ASR model.

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : str | torch.device | None
        the PyTorch device to put the model into
    download_root: Path | str | None
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    :return:
    -------
    model : Whisper
        The Whisper ASR model instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = Path("~").expanduser() / ".cache"
        download_root = default / "whisper"

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif Path(name).is_file():
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    ia3_model = IA3Whisper(dims)
    ia3_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if alignment_heads is not None:
        ia3_model.set_alignment_heads(alignment_heads)

    return ia3_model.to(device)
