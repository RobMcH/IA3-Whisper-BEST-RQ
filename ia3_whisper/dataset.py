"""Contains dataset and dataloader related classes and functions."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
import whisper

from ia3_whisper.log import get_logger

logger = get_logger(__name__)


class LibriSpeech(torch.utils.data.Dataset):
    """A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.

    It will drop the last few seconds of a very small portion of the utterances.
    Adapted from https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb
    """

    def __init__(self, split: str = "test-clean", device: str = "cpu") -> None:
        """Initialize the LibriSpeech dataset.

        :param split: The data split to load. Defaults to 'test-clean'.
        :param device: The device to load the tensors onto.
        """
        logger.info("(Down)Loading Librispeech %s split", split)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=Path("~/.cache").expanduser(),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        """Get and process the item with the corresponding index in the data.

        :param item: The index of the item within the dataset.
        :return: A dictionary holding the computed log mel spectograms ('in_feats'), the transcribed
         text ('text_targets'), and a mask denoting which tokens are entirely based on padding ('padding_mask').
        """
        audio, sample_rate, text, *_ = self.dataset[item]
        assert sample_rate == 16000
        audio = audio.flatten()
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, device=self.device)
        # Create mask to remove padding tokens.
        padding_mask = torch.ones(mel.shape[-1], dtype=torch.bool)
        # Mask out all positions with 0 std (== padding/no audio).
        padding_mask[torch.where(mel.std(dim=0) == 0)] = False
        return {"in_feats": mel, "text_targets": text, "padding_mask": padding_mask}


def get_dataloader(
    split: str, batch_size: int, shuffle: bool, device: str = "cpu"
) -> torch.utils.data.DataLoader[LibriSpeech]:
    """Construct a dataloader for the librispeech dataset.

    Dynamically truncates the padding to the maximum length of any input in the batch (ignoring padding).

    :param split: The data split to use.
    :param batch_size: The size of the batches the sampler fetches.
    :param shuffle: Whether to shuffle the data.
    :param device: [Optional] The device to load the returned tensors onto.
    :return: A dataloader yielding dictionaries of batches.
    """
    dataset = LibriSpeech(split, device)

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate a list of individual inputs into a batch by adding a batch dimension to all tensors.

        Dynamically truncates the padding to the maximum length of any input in the batch (ignoring padding).

        :param batch: An uncollated (list of dictionaries of tensors) batch obtained by Pytorch.
        :return: A collated (dictionary of tensors with batch dimensions) batch.
        """
        collated = torch.utils.data.default_collate(batch)
        # Calculate maximum signal length in the inputs.
        max_length = (
            collated["padding_mask"].to(torch.uint8).argmin(dim=1).max().cpu().item()
        )
        # Round to nearest multiple of 2 to avoid issues with temporal reduction.
        max_length = int(2 * round(max_length / 2))
        collated["padding_mask"] = collated["padding_mask"][..., :max_length]
        collated["in_feats"] = collated["in_feats"][..., :max_length]
        return collated

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
