import librosa
import librosa.display
import torch
from torch import Tensor
import numpy as np
import json
import os
import sys

from tokenization_kobert import KoBertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List, Optional

sample_rate = 24000
frame_shift = 12.5
frame_length = 50
n_iter = 30
n_fft = 2048
hop_length = int(frame_shift * 0.001*sample_rate)
win_length = int(frame_length * 0.001*sample_rate)

class TacotronDataset(Dataset):
    def __init__(self, text_files, audio_files) -> None:
        '''
        text_files = json file for tokenized text data
        audio_files = list of raw audio data files
        '''
        super().__init__()
        self.audio = audio_files
        with open(text_files, "rt", encoding='UTF8') as f:
            self.text = f.readlines()

    def __len__(self):
        assert len(self.audio) == len(self.text)
        return len(self.audio)

    def get_mel(self, index):
        file = self.audio[index]
        data, _ = librosa.load(file, sr = sample_rate)
        return np.abs(librosa.stft(data,n_fft,hop_length,win_length))

    def get_text(self, index):
        return self.text[index]

    def __getitem__(self, index) -> Tuple(Tensor):
        mel = self.get_mel(index)
        text = self.get_text(index)
        return (torch.tensor(mel), torch.tensor(text))


def collate_fn(inputs: List(Tuple(Tensor)))->List[Tensor]:
    mel, text = List(zip(*inputs))
    mel = pad_sequence(mel, batch_first= True, padding_value=0)
    text = pad_sequence(text, batch_first= True, padding_value=0)

    return [torch.stack(mel), torch.stack(text)]


def kobert_tokenizer(text, outfile, tokenizer : KoBertTokenizer):
    '''
    text: list of the following sentences. 
    e.g.
        1/1_0002.wav|용돈을 아껴 써라.|용돈을 아껴 써라.|용돈을 아껴 써라.|1.8|Save your pocket money.
    '''
    with open(outfile, "wt", encoding='UTF8') as f:
        for line in text:
            sentence = line.split('|')[1]
            token = tokenizer.tokenize(sentence)
            token_ids = tokenizer.convert_tokens_to_ids(token)

            f.write(json.dumps(token_ids))



