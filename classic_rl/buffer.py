from collections import deque
import torch
import numpy as np
from loguru import logger

class DDPGReplayBuffer():

    def __init__(self, max_size, input_shape, action_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = deque(maxlen=self.mem_size)
        self.new_state_memory = deque(maxlen=self.mem_size)
        self.action_memory = deque(maxlen=self.mem_size)
        self.reward_memory = deque(maxlen=self.mem_size)
        self.terminal_memory = deque(maxlen=self.mem_size)

        self.input_shape = input_shape
        self.action_shape = action_shape

    def store_transition(self, state, action, reward, state_, terminal):
        self.state_memory.append(state)
        self.new_state_memory.append(state_)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(terminal)

        self.mem_cntr += 1

    def sample(self, batch_size, device):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = torch.tensor(self.state_memory)[batch].to(torch.float).to(device).reshape(batch_size, -1)
        actions = torch.cat(list(self.action_memory))[batch].to(torch.float).to(device).reshape(batch_size, -1)
        rewards = torch.tensor(self.reward_memory)[batch].to(torch.float).unsqueeze(1).to(device).reshape(batch_size, -1)
        next_states = torch.tensor(self.new_state_memory)[batch].to(torch.float).to(device).reshape(batch_size, -1)
        terminals = torch.tensor(list(self.terminal_memory))[batch].to(torch.float).to(device).reshape(batch_size, -1)
        
        return states, actions, rewards, next_states, terminals
    

    def reset_buffer(self):

        self.state_memory.clear()
        self.new_state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.terminal_memory.clear()
        self.mem_cntr = 0