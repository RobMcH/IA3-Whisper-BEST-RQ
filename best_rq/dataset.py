from pathlib import Path

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

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        audio, sample_rate, text, *_ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, device=self.device)
        return {"in_feats": mel, "text_targets": text}


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
