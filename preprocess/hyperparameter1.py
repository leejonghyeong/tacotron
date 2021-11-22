import torch

batch_size = 32

#torchaudio.load only provides 44100 sample rate
sample_rate = 44100
frame_shift = 12.5
#frame_length 50 > 45 since win_length <= n_fft must hold.
frame_length = 45
n_iter = 30
n_fft = 2048
hop_length = int(frame_shift * 0.001*sample_rate)
win_length = int(frame_length * 0.001*sample_rate)
