import librosa
import librosa.display
import torch
from torch import Tensor
import numpy as np
import json
import os
import sys

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List, Optional
import hyperparameter as hp

class TacotronDataset(Dataset):
    def __init__(self, text, lin_targets, mel_targets) -> None:
        '''
        Args:
            - text:
                - list of tokenized text data
            - lin_targets:
                - numpy array of linear spectrograms
            - mel_targets:
                - numpy array of mel spectrograms
        '''
        super().__init__()
        self.lin_targets = lin_targets
        self.mel_targets = mel_targets
        self.text = text

    def __len__(self):
        assert len(self.audio) == len(self.text)
        return len(self.audio)

    def __getitem__(self, index) -> Tuple[Tensor]:
        text = self.text[index]
        lin = self.lin_targets[index]
        mel = self.mel_targets[index]
        return (torch.tensor(text), torch.tensor(mel), torch.tensor(lin))


def collate_fn(inputs: List[Tuple[Tensor]])->List[Tensor]:
    text, mel, lin = List(zip(*inputs))
    text = pad_sequence(text, batch_first= True, padding_value=0)
    mel = pad_sequence(mel, batch_first= True, padding_value=0)
    lin = pad_sequence(lin, batch_first= True, padding_value=0)

    return [torch.stack(text), torch.stack(mel), torch.stack(lin)]



