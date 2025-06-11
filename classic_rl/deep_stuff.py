
from torch import nn

class networks(nn.Module):
    def __init__(self, n_input = 2, n_output = 2, layers = [16, 16]):
        super().__init__()
        self.fa = nn.Tanh()
        self.layers_relu_stack = nn.Sequential(
            nn.Linear(n_input, layers[0]),
            self.fa,
            nn.Linear(layers[0], layers[1]),
            self.fa,  
            nn.Linear(layers[1], n_output),)
                

    def forward(self, x):
        x = self.layers_relu_stack(x)
        return x
            