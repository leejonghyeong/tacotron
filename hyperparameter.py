import torch

batch_size = 32
vocab_size = 8002
embedding_size = 256
hidden_dim = 128
max_length = 512
learning_rate = 1e-3
n_epoch = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prenet_size = 128
enc_bank_size = 16
enc_proj_size = 128
ppn_bank_size = 8
ppn_proj_size = 80
dropout = 0.5
reduction = 2

#griffin lim
sample_rate = 24000
frame_shift = 12.5
frame_length = 50
n_iter = 30
n_fft = 2048
hop_length = int(frame_shift * 0.001*sample_rate)
win_length = int(frame_length * 0.001*sample_rate)
