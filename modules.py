import torch
from torch import nn
from torch.nn import functional as fnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim import Adadelta
#from transformers import get_constant_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.notebook import tqdm

from torch.nn.utils.rnn import pad_sequence


class PreNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size,  2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, hidden_size//2)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, inputs):
        outputs = F.relu(self.fc1(inputs))
        outputs = self.dropout1(outputs)
        outputs = F.relu(self.fc2(outputs))
        outputs = self.dropout2(outputs)

        return outputs

class CBHG(nn.Module):
    def __init__(self, hidden_size, K, proj_out):
        '''
        Args:
            K(obj: int)
                bank size
            proj_out
                dimension of output of conv1d projection
        '''
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.conv1d_list = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad1d(((k-1)//2, (k-1)//2 +1),0),
                nn.Conv1d(128, 128, k)
            ) 
            for k in range(K)]) 
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv1d_proj = nn.Sequential(
            nn.Conv1d(128, 256, 3),
            nn.Conv1d(256, proj_out, 3)
        )
        self.highway = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                F.relu()
            )
            for _ in range(4)])

        self.bigru = nn.GRU(128, 128, num_layers = 1, batch_first = True, bidirectional = True)
    
    def forward(self, inputs):
        #inputs shape (batch_size, n_frames, hidden_dim)
        
        #Conv1D bank and stacking
        conv_output = []
        for conv in self.conv1d_list:
            conv_output.append(conv(inputs))
        bank_output = torch.stack(conv_output, dim = -1)
        bank_output = self.batch_norm(bank_output)

        #Maxpooling
        maxpool_output = self.maxpool(bank_output)

        #Conv1D projection and Residual connection
        proj_output = self.conv1d_proj(maxpool_output) + maxpool_output
        proj_output = self.batch_norm(proj_output)
        
        #Highway network
        highway_output = proj_output
        for layer in self.highway:
            highway_output = layer(highway_output)

        #bidirectional GRU
        CBHG_output = self.bigru(highway_output)

        return CBHG_output

class TanhAttention():
    def __init__(self, hidden_dim) -> None:
        '''
        Args:
            hidden_dim(`int`):
                hidden_size of encoder outputs
        '''
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, key, query):
        query_repeated = query.unsqueeze(1).repeat(1,key.size(1),1,1)

        attention = self.v(self.tanh(self.W1(key) + self.W2(query_repeated)))
        weight = nn.Softmax(dim=1)(attention)
        context = torch.matmul(weight.transpose(1,2), key).squeeze(1)

        return context


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dec_rnn1 = nn.GRU(input_size, hidden_size, num_layers = 1)
        self.dec_rnn2 = nn.GRU(hidden_size, hidden_size, num_layers = 1)

    def forward(self, dec_inputs, dec_hidden1, dec_hidden2):
        dec_outputs, dec_hidden1 = self.dec_rnn1(dec_inputs, dec_hidden1)
        dec_outputs, dec_hidden2 = self.dec_rnn2(dec_outputs, dec_hidden2)

        return dec_outputs, dec_hidden1, dec_hidden2