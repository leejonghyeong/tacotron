from typing import List, Dict, Union, Tuple

import torch
from torch import nn
#from transformers import get_constant_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
from modules import PreNet, CBHG, TanhAttention, DecoderRNN
import hyperparameter as hp

from torch.nn.utils.rnn import pad_sequence
class Encoder(nn.Module):
    def __init__(self, hidden_size, bank_size, proj_size):
        super().__init__()
        self.char_emb = nn.Embedding(hp.vocab_size, hp.embedding_size, padding_idx=1)
        self.pre_net = PreNet(hp.embedding_size, hidden_size, hp.dropout)
        self.cbhg = CBHG(hidden_size, bank_size, proj_size)
    
    def forward(self, inputs):
        word_emb = self.char_emb(inputs)
        prenet_outputs = self.pre_net(word_emb)
        enc_outputs = self.cbhg(prenet_outputs)

        return enc_outputs

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        '''
        Args:
            hidden_size
                - hidden_size of pre-net
        '''
        self.hidden_dim = hp.hidden_dim
        self.device = hp.device
        self.reduction = hp.reduction
        self.dropout = hp.dropout

        self.pre_net = PreNet(self.hidden_dim, hidden_size, self.dropout)
        self.attention = TanhAttention(2*self.hidden_dim)
        self.attn_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers = 1)
        self.dec_rnn = DecoderRNN(4*self.hidden_dim, 2*self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
    
    def forward(self, enc_outputs, dec_inputs):
        '''
        Args:
            dec_inputs:
                - a batch of Ground truth frames
                - Tensor of shape (batch_size, n_frames, n_fft // 2 + 1)
            reduction factor:
                - default = 2
        '''
        batch_size = dec_inputs.size(0)
        total_steps = dec_inputs.size(1) // self.reduction

        attn_hidden = torch.zeros(batch_size, self.hidden_dim, device = self.device)
        dec_hidden1 = torch.zeros(batch_size, self.hidden_dim, device = self.device)
        dec_hidden2 = torch.zeros(batch_size, self.hidden_dim, device = self.device)

        outputs = []
        attn_wt = []
        for i in range(total_steps):
            if i == 0:
                prenet_inputs = torch.zeros(batch_size, self.hidden_dim, device = self.device)
            elif self.training:
                prenet_inputs = dec_inputs[:,self.reduction * i,:]
            else:
                prenet_inputs = mel_outputs
            
            #prenet
            prenet_outputs = self.pre_net(prenet_inputs)
            #Attention RNN
            attn_outputs, attn_hidden = self.attn_rnn(prenet_outputs, attn_hidden)
            #Attention
            context, weight = self.attention(enc_outputs, attn_outputs)
            attn_wt.append(weight)
            #Decoder RNN
            dec_inputs = torch.cat([attn_outputs, context], dim = -1)
            dec_outputs, dec_hidden1, dec_hidden2 = self.dec_rnn(dec_inputs, dec_hidden1, dec_hidden2)
            #FC Layer
            dec_outputs = self.fc(dec_outputs).view(batch_size, self.hidden_dim, -1)
            #Get the next input
            mel_outputs = dec_outputs[:,:,-1]
            #Gather decoder outputs for each frames
            outputs.append(dec_outputs.transpose(1,2))
        
        return torch.stack(outputs, dim= 1), torch.stack(attn_wt, dim = 2)


class PostProcess(nn.Module):
    def __init__(self, bank_size, proj_size):
        super().__init__()
        self.cbhg = CBHG(hp.hidden_dim, bank_size, proj_size)
    def forward(self, mel_inputs):
        return self.cbhg(mel_inputs)
from torchaudio.transforms import GriffinLim

class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.griffinlim = GriffinLim(hp.n_fft,
                                    hp.n_iter, 
                                    hp.win_length, 
                                    hp.hop_length, 
                                    window_fn=torch.hann_window)
    
    def forward(self, lin_inputs):
        return self.griffinlim(lin_inputs)
              

class Tacotron(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(hp.prenet_size, hp.enc_bank_size, hp.enc_proj_size)
        self.decoder = Decoder(hp.prenet_size)
        self.postprocess = PostProcess(hp.ppn_bank_size, hp.ppn_proj_size)
        self.vocoder = Vocoder()
    
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        mel_outputs, _ = self.decoder(enc_outputs, dec_inputs, hp.reduction)
        lin_outputs = self.postprocess(mel_outputs) 
        voc_outputs = self.vocoder(lin_outputs)
                
        return mel_outputs, lin_outputs, voc_outputs
    
    def get_attn(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        _, attn = self.decoder(enc_outputs, dec_inputs, hp.reduction)
        
        return attn