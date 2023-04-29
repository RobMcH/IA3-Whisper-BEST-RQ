import torch


class BestRQMasking:
    def __init__(
        self,
        num_codebooks: int,
        num_targets: int,
        emb_dim: int,
        codebook_dim: int,
        masking_prob: float,
        device: str = "cpu",
    ) -> None:
        """Implements the Best-RQ masking strategy. Allows for both original Best-RQ and Google USM-style.

        :param num_codebooks: The number of codebooks to use for a multi-softmax pre-training (USM-style). Set to 1 for
         original Best-RQ behaviour.
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
            size=(num_codebooks, num_targets, codebook_dim),
            requires_grad=False,
            device=device,
        )  # Shape: (num_codebooks, num_targets, codebook_dim)
        self.codebooks /= torch.linalg.vector_norm(
            self.codebooks, ord=2, dim=-1, keepdim=True
        )  # Shape: (num_codebooks, num_targets, codebook_dim)
        self.masking_prob = masking_prob

    def get_targets(self, in_feats: torch.Tensor) -> torch.Tensor:
        """Computes the Best-RQ targets for a given tensor of unmasked speech input features.

        :param in_feats: A tensor holding the unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :return: A tensor holding the computed targets. Shape: (batch_size, num_codebooks, seq_length)
        """
        batch_size, num_codebooks = in_feats.shape[0], self.codebooks.shape[0]
        proj_feats = (
            in_feats @ self.projection
        )  # Shape: (batch_size, seq_length, codebook_dim)
        proj_feats /= torch.linalg.vector_norm(
            proj_feats, ord=2, dim=-1, keepdim=True
        )  # Shape: (batch_size, seq_length, codebook_dim)
        proj_feats = proj_feats[:, None, :].expand(
            -1, num_codebooks, *proj_feats.shape[1:]
        )  # Shape: (batch_size, num_codebooks, seq_length, codebook_dim)
        codebooks = self.codebooks[None, :].expand(
            batch_size, *self.codebooks.shape
        )  # Shape: (batch_size, num_codebooks, num_targets, codebook_dim)
        targets = torch.argmin(
            torch.linalg.norm(
                codebooks[:, :, :, None] - proj_feats[:, :, None, :], dim=-1
            ),
            dim=-2,
        )  # Shape: (batch_size, num_codebooks, seq_length)
        return targets

    def get_masked_features(self, in_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes the mask and masked features given some input features.

        :param in_feats: A tensor holding the unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :return: A dictionary holding:
            * masked_features: Features with masked parts replaced by randomly sampled features.
             Shape: (batch_size, seq_length, emb_dim)
            * mask: The mask used to replace the features. Shape: (batch_size, seq_length)
        """
        mask = (
            torch.rand(in_feats.shape[:-1]) < self.masking_prob
        )  # Shape: (batch_size, seq_length)
        in_feats[mask] = torch.normal(
            mean=0, std=0.1, size=(int(mask.sum().item()), in_feats.shape[-1])
        )
        return {"masked_features": in_feats, "mask": mask}

    def get_targets_and_features(
        self, in_feats: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Transforms the given features to obtain targets and the masked features using Best-RQ.

        :param in_feats: A tensor holding the unmasked speech input features. Shape: (batch_size, seq_length, emb_dim)
        :return: A dictionary holding:
            * targets: A tensor holding the computed targets. Shape: (batch_size, num_codebooks, seq_length)
            * masked_features: Features with masked parts replaced by randomly sampled features.
             Shape: (batch_size, seq_length, emb_dim)
            * mask: The mask used to replace the features. Shape: (batch_size, seq_length)
        """
        targets = self.get_targets(in_feats)
        mask_and_features = self.get_masked_features(in_feats)
        return {"targets": targets, **mask_and_features}
