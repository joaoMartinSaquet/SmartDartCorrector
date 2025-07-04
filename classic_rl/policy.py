
from torch import nn
import torch

from loguru import logger

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
        # layers_list.append(self.fa)
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
    


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1, hidden_dim2):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        self.fa =  nn.Tanh()

    def forward(self, x):
        
        x = self.fa(self.fc1(x))
        x = self.fa(self.fc2(x))
        x = self.fc3(x)
        return x

def hard_update(target, source): #(m):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1, hidden_dim2):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1 + action_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.fa =  nn.ReLU()

    def forward(self, state, action):
        x = self.fa(self.fc1(state))
        x = self.fa(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x
    

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, action_std_init=0.1, device='cpu'):
        super(PPOActorCritic, self).__init__()
        self.device = device    
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # self.action_var = action_std_init
        # Shared layers
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        ).to(device)

    def forward(self, x):
        raise NotImplemented
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        # logger.debug(f"state {state} action mean : {self.action_var}  cov mat : {cov_mat}")
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)
        return action.detach(), action_logprob.detach(), state_value.detach()

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
    
    def evaluate(self, state, action):

        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy

