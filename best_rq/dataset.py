from pathlib import Path
from typing import Tuple

import torch
import torchaudio
import whisper

# Adapted from https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb


class LibriSpeech(torch.utils.data.Dataset):
    """A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, split: str = "test-clean", device: str = "cpu") -> None:
        """Initializes the LibriSpeech dataset.

        :param split: The data split to load. Defaults to 'test-clean'.
        :param device: The device to load the tensors onto.
        """
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=Path("~/.cache").expanduser(),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        audio, sample_rate, text, *_ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, device=self.device)
        return mel, text
