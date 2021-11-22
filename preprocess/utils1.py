import librosa
import librosa.display
import json
import numpy as np
from tqdm.notebook import tqdm

from tokenization_kobert import KoBertTokenizer
import hyperparameter1 as hp

from typing import List, Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import GriffinLim

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

            f.write(json.dumps(token_ids)+"\n")


def get_tokenized(infile, outfile):
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    with open(infile, "rt", encoding='UTF8') as f:
        text = f.readlines()
    kobert_tokenizer(text, outfile, tokenizer)


def get_lin(device, outfile, dataloader):
    '''
    Args:
        - files:
            - list of paths for raw voice data
        - outfile:
            - path to save numpy array of linear spectrograms
    '''
    lin_targets = {}
    with tqdm(total = len(dataloader)) as pbar:
        window = torch.hann_window(hp.win_length,device= device)
        for id, batch in enumerate(dataloader):
            #get stft
            S = torch.abs(torch.stft(batch.to(device), hp.n_fft, hp.hop_length, hp.win_length, window = window, return_complex=True))
            #(batch_size, freq, n_frames) > (batch_size, n_frames, freq)
            S = S.transpose(1,2).detach()

            lin_targets[id] = S
            pbar.update(1)
    torch.save(lin_targets, outfile)

def get_mel(device, path, outfile, transform):
        '''
        Args:
            - path:
                - path for linear spectrogram targets
            - outfile:
                - path to save numpy array of mel scale spectrograms
            - sr:
                - Sample rate (default = 24000)
        '''
        mel_targets = {}
        lin_targets = torch.load(path)
        with tqdm(total = len(lin_targets)) as pbar:
            for id in range(len(lin_targets)):
                lin_spec = lin_targets[id].to(device)
                #(batch_size, n_frames, freq) > (batch_size, freq, n_frames)
                mel_spec = transform(lin_spec.transpose(1,2))
                
                #(batch_size, n_mels, n_frames) > (batch_size, n_frames, n_mels)
                mel_targets[id] = mel_spec.transpose(1,2).detach()
                pbar.update(1)

        torch.save(mel_targets, outfile)


#Data Loader

class AudioDataset(Dataset):
    def __init__(self, files) -> None:
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Tuple[Tensor]:
        data, _ = torchaudio.load(self.files[index])
        return data[0]


def collate_fn(inputs: List[Tuple[Tensor]])->List[Tensor]:
    audio = pad_sequence(inputs, batch_first= True, padding_value=0)

    return audio