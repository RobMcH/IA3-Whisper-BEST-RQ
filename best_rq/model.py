from __future__ import annotations

from typing import Iterable

import torch
from log import get_logger
from torch import nn
from whisper.model import (
    AudioEncoder,
    ModelDimensions,
    MultiHeadAttention,
    ResidualAttentionBlock,
    Whisper,
)

logger = get_logger(__name__)


class IA3MultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_state: int, n_head: int) -> None:
        """(IA)^3-adapted MultiHeadAttention calculation. Learn to rescale the activations for fine-tuning.

        :param n_state: The hidden size of the layers.
        :param n_head: The number of attention heads.
        """
        super().__init__(n_state=n_state, n_head=n_head)
        # Initialise parameters with zeros -> (IA)^3 will perform identity.
        self.key_weights = nn.Parameter(torch.zeros((n_state,)))
        self.value_weights = nn.Parameter(torch.zeros((n_state,)))
        self.key_biases = nn.Parameter(torch.zeros((n_state,)))
        self.value_biases = nn.Parameter(torch.zeros((n_state,)))

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculates (IA)^3-adapted QKV attention with learned activation scaling.

        :param q: A tensor holding the queries. Shape: (batch_size, n_mels, n_ctx)
        :param k: A tensor holding the keys. Shape: (batch_size, n_mels, n_ctx)
        :param v: A tensor holding the values. Shape: (batch_size, n_mels, n_ctx)
        :param mask: [Optional] A tensor holding a mask for the attention calculation.
        :return: The attention-weighted value vectors. Shape: (batch_size, n_mels, n_ctx)
        """
        # (IA)^3 changes
        k = k + (self.key_weights * k).to(k.dtype) + self.key_biases.to(k.dtype)
        v = v + (self.value_weights * v).to(v.dtype) + self.value_biases.to(v.dtype)
        return super().qkv_attention(q, k, v, mask)


class IA3MLPRescaling(nn.Module):
    def __init__(self, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_weights = nn.Parameter(torch.zeros((n_state,)))
        self.mlp_biases = nn.Parameter(torch.zeros((n_state,)))

    def forward(self, feats_in: torch.Tensor) -> torch.Tensor:
        return (
            feats_in
            + (self.mlp_weights * feats_in).to(feats_in.dtype)
            + self.mlp_biases.to(feats_in.dtype)
        )


class IA3ResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False
    ) -> None:
        """Initializes an (IA)^3-adapted ResidualAttentionBlock.

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


class IA3AudioEncoder(AudioEncoder):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ) -> None:
        """Initializes an (IA)^3-adapted AudioEncoder implementation.

        :param n_mels: The number of mel spectrograms.
        :param n_ctx: The context size.
        :param n_state: The hidden size of the layers.
        :param n_head: The number of attention heads.
        :param n_layer: The number of ResidualAttentionBlocks layers.
        """
        super().__init__(
            n_mels=n_mels, n_ctx=n_ctx, n_state=n_state, n_head=n_head, n_layer=n_layer
        )
        self.blocks: Iterable[IA3ResidualAttentionBlock] = nn.ModuleList(
            [IA3ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )


class IA3Whisper(Whisper):
    def __init__(self, dims: ModelDimensions) -> None:
        """Initializes an (IA)^3-adapted Whisper implementation.

        :param dims: A container holding the model hyperparameters.
        """
        super().__init__(dims=dims)
        self.encoder = IA3AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )

    def freeze(self) -> None:
        """Freezes all model parameters."""
        self.requires_grad_(False)

    def unfreeze_encoder_ia3(self) -> None:
        """Unfreezes the added IA3 parameters in the encoder."""
        for block in self.encoder.blocks:
            for layer in (block.attn, block.mlp):
                for name, child in layer.named_parameters():  # type: ignore
                    if name.endswith("_weights") or name.endswith("_biases"):
                        child.requires_grad_(True)
                        logger.debug("Unfreezing layer %s", name)
