"""Contains the implementations of the IA3-adapted Whisper models."""


from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Generator

import torch
import whisper.model
from torch import nn
from whisper.model import (
    AudioEncoder,
    ModelDimensions,
    MultiHeadAttention,
    ResidualAttentionBlock,
    Whisper,
)

from ia3_whisper.log import get_logger

logger = get_logger(__name__)


class IA3MultiHeadAttention(MultiHeadAttention):
    """Implements an IA3-adapted MHA as described in https://arxiv.org/abs/2205.05638."""

    def __init__(self, n_state: int, n_head: int) -> None:
        """(IA)^3-adapted MultiHeadAttention calculation. Learn to rescale the activations for fine-tuning.

        :param n_state: The hidden size of the layers.
        :param n_head: The number of attention heads.
        """
        super().__init__(n_state=n_state, n_head=n_head)
        # Initialise parameters such that (IA)^3 will perform identity.
        self.key_weights = nn.Parameter(torch.ones((n_state,)))
        self.value_weights = nn.Parameter(torch.ones((n_state,)))
        self.key_biases = nn.Parameter(torch.zeros((n_state,)))
        self.value_biases = nn.Parameter(torch.zeros((n_state,)))

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate (IA)^3-adapted QKV attention with learned activation scaling.

        :param q: A tensor holding the queries. Shape: (batch_size, n_ctx, n_state_audio)
        :param k: A tensor holding the keys. Shape: (batch_size, n_ctx, n_state_audio)
        :param v: A tensor holding the values. Shape: (batch_size, n_ctx, n_state_audio)
        :param mask: [Optional] A tensor holding a mask for the attention calculation.
        :return: The attention-weighted value vectors. Shape: (batch_size, n_ctx, n_state_audio)
        """
        # (IA)^3 changes
        k = (self.key_weights * k).to(k.dtype) + self.key_biases.to(k.dtype)
        v = (self.value_weights * v).to(v.dtype) + self.value_biases.to(v.dtype)
        return super().qkv_attention(q, k, v, mask)


class IA3MLPRescaling(nn.Module):
    """Implements an IA3-rescaled MLP for the use in IA3-adapted models."""

    def __init__(self, n_state: int, *args, **kwargs):
        """Initialize the class.

        :param n_state: The hidden dimension of the network.
        """
        super().__init__(*args, **kwargs)
        self.mlp_weights = nn.Parameter(torch.ones((n_state,)))
        self.mlp_biases = nn.Parameter(torch.zeros((n_state,)))

    def forward(self, feats_in: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass through the IA3 rescaled MLP.

        :param feats_in: The input features of the network. Shape: (batch_size, n_ctx, n_state_audio)

        :return: The computed output tensor.
        """
        return (self.mlp_weights * feats_in).to(feats_in.dtype) + self.mlp_biases.to(
            feats_in.dtype
        )


class IA3ResidualAttentionBlock(ResidualAttentionBlock):
    """Implements an IA3 ResidualAttentionBlock using MHA with learnable activation rescaling."""

    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False
    ) -> None:
        """Initialize an (IA)^3-adapted ResidualAttentionBlock.

        :param n_state: The hidden size of the layers.
        :param n_head: The number of attention heads.
        :param cross_attention: Whether to use cross attention.
        """
        super().__init__(
            n_state=n_state, n_head=n_head, cross_attention=cross_attention
        )
        self.attn = IA3MultiHeadAttention(n_state, n_head)
        self.cross_attn = (
            IA3MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.mlp: nn.Sequential = nn.Sequential(
            *[*self.mlp[:-1], IA3MLPRescaling(self.mlp[0].out_features), self.mlp[-1]]
        )


class CodebookClassifier(torch.nn.Module):
    """Implement the CodebookClassifier for mapping masked input features to quantized codebook labels."""

    def __init__(
        self,
        num_codebooks: int,
        num_targets: int,
        n_audio_state: int,
        device: str,
        *args,
        **kwargs,
    ):
        """Initialize a number of codebook classifiers for BEST-RQ training of the encoder.

        :param num_codebooks: The number of codebook classifiers to add.
        :param num_targets: The number of targets (classes) per classifier.
        :param n_audio_state: The hidden size of audio features within the model.
        :param device: The device to initialize the classifiers on.
        """
        super().__init__(*args, **kwargs)
        self.codebook_classifier = torch.nn.Linear(
            n_audio_state, num_targets * num_codebooks, device=device
        )
        self.num_codebooks = num_codebooks
        self.num_targets = num_targets

    def forward(
        self,
        x: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Compute a forward pass through the codebook classifiers.

        :param x: Tensor holding the mel spectrogram of the audio. Shape = (batch_size, n_ctx, n_state_audio)
        :param target_mask: Tensor holding a mask to extract the target tokens.
        :return:
            x: The input features. Shape: (batch_size, n_ctx, n_state_audio)
            logits: The logits obtained by applying the codebook classifiers to the encoded features.
             Shape: (num_codebooks, num_masked, num_targets).
        """
        x_inp = x[target_mask]  # Shape: (num_masked, n_state_audio)
        num_masked = x_inp.shape[0]
        logits = self.codebook_classifier(x_inp).reshape(
            num_masked, self.num_codebooks, self.num_targets
        )
        logits = logits.permute(
            1, 0, 2
        )  # Shape: (num_codebooks, num_masked, num_targets).
        return x, logits


class IA3AudioEncoder(AudioEncoder):
    """Implements an IA3-adapted Whisper AudioEncoder with learnable activation rescaling."""

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ) -> None:
        """Initialize an (IA)^3-adapted AudioEncoder implementation.

        :param n_mels: The number of mel spectrograms.
        :param n_ctx: The context size.
        :param n_state: The hidden size of the layers.
        :param n_head: The number of attention heads.
        :param n_layer: The number of ResidualAttentionBlocks layers.
        """
        super().__init__(
            n_mels=n_mels, n_ctx=n_ctx, n_state=n_state, n_head=n_head, n_layer=n_layer
        )
        self.blocks = nn.ModuleList(
            [IA3ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.codebook_classifiers: None | CodebookClassifier = None

    def add_codebook_classifiers(
        self, num_codebooks: int, num_targets: int, n_audio_state: int, device: str
    ) -> None:
        """Add a number of codebook classifiers for BEST-RQ training of the encoder to the model.

        :param num_codebooks: The number of codebook classifiers to add.
        :param num_targets: The number of targets (classes) per classifier.
        :param n_audio_state: The hidden size of audio features within the model.
        :param device: The device to initialize the classifiers on.
        """
        if num_codebooks <= 0:
            raise ValueError(
                f"Number of codebooks must be greater than 0. Got {num_codebooks}"
            )
        if num_targets <= 0:
            raise ValueError(
                f"Number of targets must be greater than 0. Got {num_targets}"
            )

        self.codebook_classifiers = CodebookClassifier(
            num_codebooks, num_targets, n_audio_state, device
        )

    def forward(
        self,
        x: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Compute a forward pass through the encoder.

        If the encoder.codebook_classifiers have been added, additionally computes the logits of the classifiers.

        :param x: Tensor holding the mel spectrogram of the audio. Shape = (batch_size, n_mels, n_ctx)
        :param target_mask: An optional target mask for BEST-RQ training to select the target tokens.
        :return:
            x: The computed encoded features. Shape: (batch_size, n_ctx, n_state_audio)
            logits: [Optional] The logits obtained by applying the codebook classifiers to the encoded features.
             Shape: (num_codebooks, num_masked, num_targets).
        """
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        n_ctx = x.shape[1]
        # Add positional embeddings; truncated to x's context length.
        x = (x + self.positional_embedding[:n_ctx]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return (
            x
            if self.codebook_classifiers is None
            else self.codebook_classifiers(x, target_mask)
        )


class IA3TextDecoder(whisper.model.TextDecoder):
    """Implements an (IA)^3-adapted TextDecoder with learnable activation rescaling."""

    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """Initialize an (IA)^3-adapted TextDecoder implementation.

        :param n_vocab: The size of the vocabulary.
        :param n_ctx: The maximum sequence length.
        :param n_state: The hidden feature size.
        :param n_head: The number of attention heads.
        :param n_layer: The number of decoder layers.
        """
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

        self.blocks = nn.ModuleList(
            [
                IA3ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )


class IA3Whisper(Whisper):
    """Implements an IA3-adapted Whisper model with IA3-weights injected into the encoder and decoder stacks."""

    def __init__(self, dims: ModelDimensions) -> None:
        """Initialize an (IA)^3-adapted Whisper implementation.

        :param dims: A container holding the model hyper-parameters.
        """
        super().__init__(dims=dims)
        self.encoder = IA3AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = IA3TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def freeze(self) -> None:
        """Freeze all model parameters."""
        self.requires_grad_(False)

    def unfreeze_ia3(self, stack_prefix: str = "encoder.") -> None:
        """Unfreeze the added IA3 parameters in the encoder.

        :param stack_prefix: A string encoding a prefix within the state dict, to filter the weights for the
         encoder/decoder stacks.
        """
        for name, child in self._get_ia3_parameters(stack_prefix):
            child.requires_grad_(True)
            logger.debug("Unfreezing parameters of %s", name)

    def save_ia3_state_dict(self, path: Path, stack_prefix: str = "encoder.") -> None:
        """Write the IA3 parameter state dict to the specified path.

        :param stack_prefix: A string encoding a prefix within the state dict, to filter the weights for the
         encoder/decoder stacks.

        :param path: The path to write the state dict to.
        """
        ia3_state_dict = self._get_ia3_state_dict(stack_prefix)
        torch.save(ia3_state_dict, path)

    def load_ia3_state_dict(self, path: Path) -> None:
        """Load an IA3 state dict from disk and updates the corresponding model weights.

        :param path: The path from where to load the weights from.
        """
        ia3_state_dict = torch.load(path)
        self.load_state_dict(ia3_state_dict, strict=False)

    def add_codebook_classifiers(self, num_codebooks: int, num_targets: int) -> None:
        """Add a number of codebook classifiers for BEST-RQ training of the encoder to the model.

        :param num_codebooks: The number of codebook classifiers to add.
        :param num_targets: The number of targets (classes) per classifier.
        """
        self.encoder.add_codebook_classifiers(
            num_codebooks, num_targets, self.dims.n_audio_state, self.device
        )

    def _get_ia3_state_dict(self, stack_prefix: str = "encoder.") -> OrderedDict:
        """Get the state dict of only the IA3 parameters.

        :param stack_prefix: A string encoding a prefix within the state dict, to filter the weights for the
         encoder/decoder stacks.

        :return: A PyTorch state dict of the IA3 encoder parameters.
        """
        ia3_state_dict = OrderedDict()
        for name, child in self._get_ia3_parameters(stack_prefix):
            ia3_state_dict[name] = self.state_dict()[name]
        return ia3_state_dict

    def _get_ia3_parameters(
        self, stack_prefix: str = "encoder."
    ) -> Generator[tuple[str, torch.nn.Parameter], None, None]:
        """Return a generator over the IA3 parameters of the model.

        The given prefix allows to only get the IA3 parameters for a given stack (encoder/decoder).

        :param stack_prefix: A string encoding a prefix within the state dict, to filter the weights for the
         encoder/decoder stacks.

        :return: A generator yielding tuples of names and parameters.
        """
        for name, child in self.named_parameters():  # type: ignore
            if name.startswith(f"{stack_prefix}") and (
                name.endswith("_weights") or name.endswith("_biases")
            ):
                yield name, child
