import torch

class hyperparameters():
    def __init__(self, **kwargs) -> None:
        self.batch_size = kwargs.pop('batch_size', 32)
        self.vocab_size = 0
        self.embedding_size = kwargs.pop('embedding_size', 256)
        self.hidden_dim = kwargs.pop('d_hidden', 128)
        self.max_length = kwargs.pop('max_length', 512)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.n_epoch = kwargs.pop('n_epoch', 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sample_rate = 24000
        self.frame_shift = 12.5
        self.frame_length = 50
        self.n_iter = 30
        self.n_fft = 2048
        self.hop_length = int(self.frame_shift * 0.001*self.sample_rate)
        self.win_length = int(self.frame_length * 0.001*self.sample_rate)

        self.prenet_size = 128
        self.enc_bank_size = 16
        self.enc_proj_size = 128
        self.ppn_bank_size = 8
        self.ppn_proj_size = 80
        self.dropout = 0.5
        self.reduction = 2