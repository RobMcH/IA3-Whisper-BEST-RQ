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
        # Calculate the number of full padding tokens at the end.
        padding_tokens = (
            max(whisper.audio.N_SAMPLES - audio.shape[-1], 0)
            // whisper.audio.HOP_LENGTH
        )
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, device=self.device)
        # Create mask to remove padding tokens.
        padding_mask = torch.ones(mel.shape[-1], dtype=torch.bool)
        padding_mask[-padding_tokens:] = False
        return {"in_feats": mel, "text_targets": text, "padding_mask": padding_mask}


def get_dataloader(
    split: str, batch_size: int, shuffle: bool, device: str = "cpu"
) -> torch.utils.data.DataLoader:
    """Construct a dataloader for the librispeech dataset.

    :param split: The data split to use.
    :param batch_size: The size of the batches the sampler fetches.
    :param shuffle: Whether to shuffle the data.
    :param device: [Optional] The device to load the returned tensors onto.
    :return: A dataloader yielding dictionaries of batches.
    """
    dataset = LibriSpeech(split, device)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
