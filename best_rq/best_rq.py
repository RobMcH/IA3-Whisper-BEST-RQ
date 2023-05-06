from __future__ import annotations

import faiss

# Enables PyTorch CUDA inputs/outputs for FAISS.
import faiss.contrib.torch_utils
import torch


class BestRQMasking:
    def __init__(
        self,
        num_targets: int,
        emb_dim: int,
        codebook_dim: int,
        masking_prob: float,
        device: str = "cpu",
    ) -> None:
        """Implements the Best-RQ masking strategy. Allows for both original Best-RQ and Google USM-style.

        :param num_targets: The number of quantization targets per codebook.
        :param emb_dim: The dimension of the speech input features.
        :param codebook_dim: The dimension of the codebook vectors.
        :param masking_prob: The probability of masking an input.
        :param device: The device to initialize parameters on. Defaults to "CPU".
        """
        self.projection = torch.empty(
            emb_dim, codebook_dim, requires_grad=False, device=device
        )  # Shape: (emb_dim, codebook_dim)
        torch.nn.init.xavier_normal_(self.projection)
        self.codebooks = torch.normal(
            mean=0,
            std=1,
            size=(num_targets, codebook_dim),
            requires_grad=False,
            device=device,
        )  # Shape: (num_targets, codebook_dim)
        self.codebooks /= torch.linalg.vector_norm(
            self.codebooks, ord=2, dim=-1, keepdim=True
        )  # Shape: (num_targets, codebook_dim)
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()
        self.masking_prob = masking_prob

    def get_targets(self, in_feats: torch.Tensor) -> torch.Tensor:
        """Computes the Best-RQ targets for a given tensor of unmasked speech input features.

        :param in_feats: A tensor holding the unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :return: A tensor holding the computed targets. Shape: (batch_size, seq_length)
        """
        batch_size, seq_length = in_feats.shape[0], in_feats.shape[1]
        proj_feats = (
            in_feats @ self.projection
        )  # Shape: (batch_size, seq_length, codebook_dim)
        proj_feats /= torch.linalg.vector_norm(
            proj_feats, ord=2, dim=-1, keepdim=True
        )  # Shape: (batch_size, seq_length, codebook_dim)
        targets = faiss.knn_gpu(
            self.res,
            xq=proj_feats.reshape(batch_size * seq_length, -1),
            xb=self.codebooks,
            k=1,
        )[
            -1
        ]  # Shape: (batch_size * seq_length)
        return targets.reshape(
            batch_size, seq_length
        )  # Shape: (batch_size, seq_length).

    def get_masked_features(self, in_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes the mask and masked features given some input features.

        :param in_feats: A tensor holding the unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :return: A dictionary holding:
            * in_feats: Features with masked parts replaced by randomly sampled features.
             Shape: (batch_size, seq_length, emb_dim)
            * mask: The mask used to replace the features. Shape: (batch_size, seq_length)
        """
        mask = (
            torch.rand(in_feats.shape[:-1]) < self.masking_prob
        )  # Shape: (batch_size, seq_length)
        in_feats[mask] = torch.normal(
            mean=0,
            std=0.1,
            size=(int(mask.sum().item()), in_feats.shape[-1]),
            device=in_feats.device,
        )
        return {"in_feats": in_feats, "mask": mask}

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
        data["targets"] = self.get_targets(data["in_feats"])
        data.update(self.get_masked_features(data["in_feats"]))
        return data
