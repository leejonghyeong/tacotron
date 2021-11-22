import torch
from torch import nn
from torch.nn import functional as fnn
import torch.nn.functional as F


class PreNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, inputs):
        outputs = self.dropout1(F.relu(self.fc1(inputs)))
        outputs = self.dropout2(F.relu(self.fc2(outputs)))

        return outputs

class CBHG(nn.Module):
    def __init__(self, input_size, K, proj_hidden, proj_out):
        '''
        Args:
            K(obj: int)
                bank size
            proj_out
                dimension of output of conv1d projection
        '''
        super().__init__()
        self.conv_hidden = 128
        self.high_hidden = 128
        self.gru_hidden = 128

        self.batch_norm_list = nn.ModuleList([
            nn.BatchNorm1d(self.conv_hidden)
            for _ in range(K)])
        self.batch_norm_proj = nn.BatchNorm1d(proj_out)

        self.conv1d_list = nn.ModuleList([nn.Sequential(
                nn.ConstantPad1d(((k-1)//2, k//2),0),
                nn.Conv1d(input_size, self.conv_hidden, k),
                nn.ReLU()
                ) for k in range(1, K+1)])

        self.maxpool = nn.Sequential(
            nn.ConstantPad1d((0, 1),0),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv1d_proj = nn.Sequential(
            nn.ConstantPad1d((1, 1),0),
            nn.Conv1d(self.conv_hidden * K, proj_hidden, 3),
            nn.ReLU(),
            nn.ConstantPad1d((1, 1),0),
            nn.Conv1d(proj_hidden, proj_out, 3)
        )
        self.highway = nn.ModuleList([
            nn.Sequential(
              nn.Linear(proj_out,self.high_hidden),
              nn.ReLU()
            )
        ])
        
        for _ in range(3):
            self.highway.append(nn.Sequential(
                    nn.Linear(self.high_hidden, self.high_hidden),
                    nn.ReLU()
                ))

        self.bigru = nn.GRU(self.high_hidden, self.gru_hidden, num_layers = 1, batch_first = True, bidirectional = True)
    
    def forward(self, inputs):
        #(batch_size, n_frames, hidden_dim) > (batch_size, hidden_dim, n_frames)
        inputs = inputs.transpose(1,2).contiguous()
        #Conv1D bank and stacking
        conv_output = []
        for conv, batch_norm in list(zip(self.conv1d_list, self.batch_norm_list)):
            conv_output.append(batch_norm(conv(inputs)))
        bank_output = torch.cat(conv_output, dim = 1)

        #Maxpooling
        maxpool_output = self.maxpool(bank_output)

        #Conv1D projection and Residual connection
        proj_output = self.conv1d_proj(maxpool_output)
        proj_output = self.batch_norm_proj(proj_output) + inputs

        #Highway network
        highway_output = proj_output.transpose(1,2).contiguous()
        for layer in self.highway:
            highway_output = layer(highway_output)
        
        #bidirectional GRU
        cbhg_output, _ = self.bigru(highway_output)
        return cbhg_output

class TanhAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
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
        query_repeated = query.unsqueeze(1).repeat(1,key.size(1),1)
        attention = self.v(self.tanh(self.W1(key) + self.W2(query_repeated)))
        weight = nn.Softmax(dim=1)(attention.squeeze(-1))
        context = torch.matmul(weight.unsqueeze(1), key).squeeze(1)

        return context, weight


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dec_rnn1 = nn.GRUCell(input_size, hidden_size)
        self.dec_rnn2 = nn.GRUCell(hidden_size, hidden_size)
        self.attn_proj = nn.Linear(input_size, hidden_size)

    def forward(self, dec_inputs, dec_hidden1, dec_hidden2):
        dec_hidden1 = self.dec_rnn1(dec_inputs, dec_hidden1)
        dec_outputs = self.attn_proj(dec_inputs) + dec_hidden1
        dec_hidden2 = self.dec_rnn2(dec_outputs, dec_hidden2)
        dec_outputs = dec_outputs + dec_hidden2

        return dec_outputs, dec_hidden1, dec_hidden2