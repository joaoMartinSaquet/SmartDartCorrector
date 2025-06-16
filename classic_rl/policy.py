
from torch import nn
import torch

class REINFORCEnet(nn.Module):
    def __init__(self, n_input = 2, n_output = 2, layers = [16, 16]):
        super().__init__()
        self.fa = nn.Tanh()

        layers_list = []
        for i in range(len(layers)):

            if i == 0:
                layers_list.append(nn.Linear(n_input, layers[i]))
                layers_list.append(self.fa)
            else:
                layers_list.append(nn.Linear(layers[i-1], layers[i]))
                layers_list.append(self.fa)
        layers_list.append(nn.Linear(layers[-1], n_output))
        self.layers_relu_stack = nn.Sequential(*layers_list)


    def forward(self, x):
        x = self.layers_relu_stack(x)
        return x
    

class REINFORCELSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, sequence_length):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sequence_length = sequence_length

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn