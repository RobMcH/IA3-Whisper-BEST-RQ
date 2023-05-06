from __future__ import annotations

import faiss

# Enables PyTorch CUDA inputs/outputs for FAISS.
import faiss.contrib.torch_utils
import torch

WHISPER_FRAME_LENGTH = 25


class BestRQMasking:
    def __init__(
        self,
        num_targets: int,
        emb_dim: int,
        codebook_dim: int,
        masking_prob: float,
        masking_length: int = 400,
        device: str = "cpu",
        seed: int = 0,
        metric: int = faiss.METRIC_INNER_PRODUCT,
    ) -> None:
        """Implements the Best-RQ masking strategy. Allows for both original Best-RQ and Google USM-style.

        :param num_targets: The number of quantization targets per codebook.
        :param emb_dim: The dimension of the speech input features.
        :param codebook_dim: The dimension of the codebook vectors.
        :param masking_prob: The probability of masking an input.
        :param masking_length: The length of the masks in ms.
        :param device: The device to initialize parameters on. Defaults to "CPU".
        :param seed: The seed used to initialize the RNG.
        :param metric: The metric type to use for the inner product search. Defaults to inner product search, which
         is equivalent to cosine similarity as inputs are normalized.
          C.f. https://github.com/facebookresearch/faiss/blob/main/faiss/MetricType.h.
        """
        # Set up class-internal RNGs for better reproducibility.
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.mask_rng = torch.Generator(device="cpu").manual_seed(seed)

        # Set up random projection matrix.
        self.projection = torch.empty(
            emb_dim, codebook_dim, requires_grad=False, device=device
        )  # Shape: (emb_dim, codebook_dim)
        # nn.init does not support custom RNGs, use default.
        torch.nn.init.xavier_normal_(self.projection)

        # Set up codebooks.
        self.codebooks = torch.normal(
            mean=0,
            std=1,
            size=(num_targets, codebook_dim),
            requires_grad=False,
            device=device,
            generator=self.rng,
        )  # Shape: (num_targets, codebook_dim)
        self.codebooks /= torch.linalg.vector_norm(
            self.codebooks, ord=2, dim=-1, keepdim=True
        )  # Shape: (num_targets, codebook_dim)

        # Initialize (GPU) resources for FAISS.
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()
        self.metric = metric

        # Masking hyper-parameters.
        self.masking_prob = masking_prob
        # Discrete masking length in number of frames to mask.
        self.masking_length = masking_length // WHISPER_FRAME_LENGTH

    def get_targets_and_features(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Transforms the given features to obtain targets and the masked features using Best-RQ.

        Modifies given data ('in_feats') in-place.

        :param data: A dictionary holding the input data. Must contain a tensor with key "in_feats" with the
         unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :return: A dictionary holding:
            * targets: A tensor holding the computed targets. Shape: (batch_size, seq_length)
            * in_feats: Features with masked parts replaced by randomly sampled features.
             Shape: (batch_size, seq_length, emb_dim)
            * mask: The mask used to replace the features. Shape: (batch_size, seq_length)
        """
        mask = self.get_mask(data["in_feats"].shape)
        data["targets"] = self.get_targets(data["in_feats"][mask])
        data.update(self.apply_mask(data["in_feats"], mask))
        return data

    def get_targets(self, in_feats: torch.Tensor) -> torch.Tensor:
        """Computes the Best-RQ targets for a given tensor of unmasked speech input features.

        :param in_feats: A tensor holding the masked speech input features. Shape: (num_masked, emb_dim)
        :return: A tensor holding the computed targets. Shape: (num_masked,)
        """
        proj_feats = in_feats @ self.projection  # Shape: (num_masked, codebook_dim)
        proj_feats /= torch.linalg.vector_norm(
            proj_feats, ord=2, dim=-1, keepdim=True
        )  # Shape: (num_masked, codebook_dim)
        _, targets = faiss.knn_gpu(
            self.res,
            xq=proj_feats,
            xb=self.codebooks,
            k=1,
            metric=self.metric,
        )  # Shape: (num_masked,)
        return targets  # Shape: (num_masked,).

    def get_mask(self, in_feats_shape: torch.Size) -> torch.Tensor:
        """Computes the mask and masked features given some input features.

        :param in_feats_shape: A torch.Size holding the shape of the input features.
         Dimensions: (batch_size, seq_length, emb_dim)
        :return: The mask used to replace the features. Shape: (batch_size, seq_length)
        """
        mask = (
            torch.rand(in_feats_shape[:-1], generator=self.mask_rng) < self.masking_prob
        )  # Shape: (batch_size, seq_length)
        # Turn individual samples into continuous masks. Get (batch_index, seq_index) pairs of individual samples.
        batch_span_indices, seq_span_indices = torch.where(mask)
        # Repeat batch indices self.masking_length times to match new sequence span indices.
        batch_span_indices = batch_span_indices.repeat_interleave(self.masking_length)
        # Repeat sequence indices self.masking_length times and turn into increasing indices (+0..self.masking_length-1)
        seq_span_indices = seq_span_indices.repeat_interleave(
            self.masking_length
        ) + torch.arange(self.masking_length).repeat(int(mask.sum().item()))
        # Remove any possibly invalid indices, i.e., clip at the end of the sequence.
        batch_span_indices = batch_span_indices[seq_span_indices < mask.shape[1]]
        seq_span_indices = seq_span_indices[seq_span_indices < mask.shape[1]]
        # Update mask with the sampled spans.
        mask[batch_span_indices, seq_span_indices] = True
        return mask

    def apply_mask(
        self, in_feats: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Applies the mask to the input features and replaces masked regions by randomly sampled values.

        :param in_feats: A tensor holding the unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :param mask: The mask used to replace the features. Shape: (batch_size, seq_length)
        :return: A dictionary holding:
            * in_feats: The masked input features. (batch_size, seq_length, emb_dim)
            * mask: The mask used to replace the features. Shape: (batch_size, seq_length)
        """
        in_feats[mask] = torch.normal(
            mean=0,
            std=0.1,
            size=(int(mask.sum().item()), in_feats.shape[-1]),
            device=in_feats.device,
            generator=self.rng,
        )
        return {"in_feats": in_feats, "mask": mask}
