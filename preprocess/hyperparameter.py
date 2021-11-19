import torch

sample_rate = 24000
frame_shift = 12.5
frame_length = 50
n_iter = 30
n_fft = 2048
hop_length = int(frame_shift * 0.001*sample_rate)
win_length = int(frame_length * 0.001*sample_rate)
