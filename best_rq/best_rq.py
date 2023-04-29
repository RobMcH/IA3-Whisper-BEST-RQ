import torch


class BestRQMasking:
    def __init__(
        self,
        num_codebooks: int,
        num_targets: int,
        emb_dim: int,
        codebook_dim: int,
        device: str = "cpu",
    ) -> None:
        """Implements the Best-RQ masking strategy. Allows for both original Best-RQ and Google USM-style.

        :param num_codebooks: The number of codebooks to use for a multi-softmax pre-training (USM-style). Set to 1 for
         original Best-RQ behaviour.
        :param num_targets: The number of quantization targets per codebook.
        :param emb_dim: The dimension of the speech input features.
        :param codebook_dim: The dimension of the codebook vectors.
        :param device: The device to initialize parameters on. Defaults to "CPU".
        """
        self.projection = torch.empty(
            emb_dim, codebook_dim, requires_grad=False, device=device
        )
        torch.nn.init.xavier_normal_(self.projection)
        self.codebooks = torch.normal(
            mean=0,
            std=1,
            size=(num_codebooks, num_targets, codebook_dim),
            requires_grad=False,
            device=device,
        )
        self.codebooks /= torch.linalg.vector_norm(
            self.codebooks, ord=2, dim=-1, keepdim=True
        )
