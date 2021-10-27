import torch
import torch.nn as nn
'''
k = 4
inputs = torch.randn(20,16,50)
left = (k-1)//2
right = (k-1)//2 + 1
conv = nn.Sequential(
    nn.ConstantPad1d((left, right), 0),
    nn.Conv1d(16,33,k, stride= 1)
)

outputs = conv(inputs)

print(outputs.size())

x = torch.tensor([1,2,3,4])
x = x.view(2,2)
print(x)
'''
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
sample_rate = 24000
frame_shift = 12.5
frame_length = 50
n_iter = 30
n_fft = 2048
hop_length = int(frame_shift * 0.001*sample_rate)
win_length = int(frame_length * 0.001*sample_rate)
print(hop_length)
print(win_length)

y, sr = librosa.load("archive\kss\\1\\1_0000.wav", sr = 24000)
print(y.shape)
print(84637//hop_length)
S = np.abs(librosa.stft(y,n_fft,hop_length,win_length))
print(S.shape)
'''
fig, ax = plt.subplots()
img = librosa.display.specshow(
    librosa.amplitude_to_db(S, ref=np.max),
    y_axis = "log",
    x_axis = "time",
    ax = ax)

ax.set_title('Power sepctrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
'''
import torchaudio
from torchaudio.transforms import Spectrogram

wav, sr = torchaudio.load("archive\kss\\1\\1_0000.wav")
print(wav.shape, sr)
spec = Spectrogram(n_fft, win_length,hop_length,window_fn=torch.hann_window)

wav_spec = spec(wav)
print(wav_spec.size())