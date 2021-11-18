import librosa
import librosa.display
import numpy as np
import json
from ..tokenizer import get_tokenized
from ..hyperparameter import hyperparameters as hp

def get_lin(files, outfile):
    '''
    Args:
        - files:
            - list of paths for raw voice data
        - outfile:
            - path to save numpy array of linear spectrograms
    '''
    lin_targets = []
    for file in files:
        data, _ = librosa.load(file, sr = hp.sample_rate)
        #get stft
        S = np.abs(librosa.stft(data, hp.n_fft, hp.hop_length, hp.win_length))
        #(freq, n_frames) > (n_frames, freq)
        S = S.transpose(0,1)

        lin_targets.append(S)
    np.save(outfile, lin_targets, allow_pickle = True)

def get_mel(path, outfile, sr):
    '''
    Args:
        - path:
            - path for linear spectrogram targets
        - outfile:
            - path to save numpy array of mel scale spectrograms
        - sr:
            - Sample rate (default = 24000)
    '''
    mel_targets = []
    lin_targets = np.load(path)
    for lin_spec in lin_targets:
        mel_spec = librosa.feature.melspectrogram(lin_spec, sr=sr)
        mel_targets.append(mel_spec)
    np.save(outfile, mel_targets, allow_pickle = True)



#get tokenized text data
infile = 'C:\\Users\\leejo\\Desktop\\파이썬\\tacotron\\archive\\transcript.v.1.4.txt'
outfile = 'C:\\Users\\leejo\\Desktop\\파이썬\\tacotron\\archive\\transcript_ko.txt'
get_tokenized(infile, outfile)

#get lin_targets
files = librosa.util.find_files('archive\kss')
lin_outfile = 'C:\\Users\\leejo\\Desktop\\파이썬\\tacotron\\archive\\lin_target.npy'
lin_targets = get_lin(files, lin_outfile)

#get mel_targets
mel_outfile = 'C:\\Users\\leejo\\Desktop\\파이썬\\tacotron\\archive\\mel_target.npy'
get_mel(lin_outfile, mel_outfile, hp.sample_rate)

