from collections import deque
import numpy as np
from torch import nn 
from torch import optim
import torch    
from godot_rl.core.godot_env import GodotEnv
import sys
import os
import time
import json
import tqdm
from copy import deepcopy

from classic_rl.policy import REINFORCEnet, REINFORCELSTM,DDPGActor, DDPGCritic
from classic_rl.policy import hard_update, soft_update
from classic_rl.buffer import DDPGReplayBuffer
from common.user_simulator import *
from common.perturbation import *
from common.rolloutenv import *
from common.corrector import *

# steps where we say, that's enough reset yourselves
MAXSTEPS =int(1e6)

class inputBuffer(deque):
    def __init__(self, input_dim, maxlen):
        super().__init__(maxlen=maxlen)
        self.input_dim = input_dim

    def reset(self):
        self.clear()
        
        for _ in range(self.maxlen):
            self.append(torch.zeros(self.input_dim))

    def add(self, input):
        self.append(input)


    def get(self):
        return torch.stack(list(self))


def normalize(x):
    return (x - (-MAX_DISP))/(MAX_DISP - (-MAX_DISP))

def unnormalize(x):
    return (x * (MAX_DISP - (-MAX_DISP))) + (-MAX_DISP)


class ReinforceCorrector(Corrector):
    """
    A reinforcement learning-based corrector that uses the REINFORCE algorithm to learn
    optimal correction actions for user inputs in a smart darts environment.
    
    This class implements a policy gradient method that learns to correct user movements
    by training neural networks to predict both the mean and standard deviation of
    correction actions. The algorithm uses the REINFORCE policy gradient method to
    optimize the policy based on received rewards.
    
    Inspired from: https://www.geeksforgeeks.org/reinforce-algorithm/
    
    Attributes:
        gamma (float): Discount factor for future rewards (0.99)
        learning_rate (float): Learning rate for the optimizer (0.01)
        num_episodes (int): Number of training episodes (50)
        batch_size (int): Batch size for training (64)
        mean_network (nn.Module): Neural network that predicts mean correction actions
        std_network (nn.Module): Neural network that predicts standard deviation of actions
        optimizer (torch.optim.Adam): Optimizer for training both networks
        env (GodotEnv): The Godot environment for simulation
        u_sim (UserSimulator): User simulator for generating user actions
        perturbator (Perturbator): Optional perturbation module for adding noise
    """
    def __init__(self, env : GodotEnv, u_sim : UserSimulator, perturbator : Perturbator = None, hidden_size = 64, learning_rate = 0.01, learn = False, log = False, policy_type = "StackedMLP"):
        """

        Initialize the ReinforceCorrector with environment, user simulator, and training parameters.
        
        Args:
            env (GodotEnv): The Godot environment for running simulations
            u_sim (UserSimulator): User simulator for generating user movement actions
            perturbator (Perturbator, optional): Module to add perturbations to user actions. Defaults to None.
            learn (bool, optional): Whether the corrector is in learning mode. Defaults to False.
            log (bool, optional): Whether to enable logging of training data. Defaults to False.
            policy_type = ["StackedMLP", "MLP", "LSTM"]
        """
        super().__init__(learn)
        # Logging configuration
        self.log = log
        self.log_path = "logs_corrector/Reinforce/" + time.strftime("%Y%m%d-%H%M%S") 

        # REINFORCE algorithm hyperparameters
        self.gamma = 0.99  # Discount factor for computing returns
        self.learning_rate = learning_rate # Learning rate for neural network optimization
        self.num_episodes = 50   # Number of training episodes
        self.batch_size = 64  # Batch size (currently not used in implementation)

        # Training configuration
        self.seed = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = [hidden_size, hidden_size, hidden_size]
        # Neural networks for policy representation
        # Mean network: outputs the mean of the action distribution
        input_dim = 2
        if policy_type == "MLP":
            # Mean network: outputs the mean of the action distribution
            self.mean_network = REINFORCEnet(n_input = input_dim, n_output = 2, layers = [32, 32]).to(self.device)
            # Standard deviation network: oPolicy not implementedutputs log std dev of the action distribution
            self.std_network = REINFORCEnet(n_input = input_dim, n_output = 2, layers = [32, 32]).to(self.device)
        elif policy_type == "LSTM":
            # Mean network: outputs the mean of the action distribution
            self.mean_network = REINFORCELSTM(input_dim = input_dim, hidden_dim = 128, layer_dim = 2, output_dim = 2, sequence_length = 20).to(self.device)
            # Standard deviation network: outputs log std dev of the action distribution
            self.std_network = REINFORCELSTM(input_dim = input_dim, hidden_dim = 128, layer_dim = 2, output_dim = 2, sequence_length = 20).to(self.device)

            self.buffer = inputBuffer(input_dim=input_dim, maxlen=self.mean_network.sequence_length)

        elif policy_type == "StackedMLP":
            self.sequence_length = 10
            # Mean network: outputs the mean of the action distribution
            self.mean_network = REINFORCEnet(n_input = input_dim * self.sequence_length,n_output = 2, layers = self.layers).to(self.device)
            # Standard deviation network: outputs log std dev of the action distribution
            self.std_network = REINFORCEnet(n_input = input_dim * self.sequence_length, n_output = 2, layers = self.layers).to(self.device)
            self.buffer = inputBuffer(input_dim=input_dim, maxlen=self.sequence_length)
        else :
            print("unknow policy net type")
            raise NotImplementedError
        
        self.policy_type = policy_type

        # Optimizer for both networks (combined parameter list)
        self.optimizer = optim.Adam(list(self.mean_network.parameters()) + list(self.std_network.parameters()), lr=self.learning_rate)

        # Environment and simulation components
        self.env = env  # Godot environment for smart darts simulation
        self.sb = isinstance(self.env, StableBaselinesGodotEnv)
        self.u_sim = u_sim  # User simulator for generating human-like movements
        self.perturbator = perturbator  # Optional noise/perturbation module



    def compute_return(self, rewards): 
        G = np.zeros(len(rewards))
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = self.gamma * running_return + rewards[t][0]
            G[t] = running_return
        return G


    def train_step(self, obss, actions, rewards):
        
        
        if self.sb:
            obss = obss.squeeze(1)
        G = self.compute_return(rewards)

        if self.policy_type == "LSTM":
            obss = normalize(obss)
            means_actions, _, _ = self.mean_network(obss)
            std_actions, _, _ = self.std_network(obss)
        else :
            means_actions = self.mean_network(obss)
            std_actions = self.std_network(obss)


        stds = torch.exp(std_actions)
        dist = torch.distributions.Normal(means_actions, stds)
            
        # action = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)


        # Compute policy loss
        # print(G)
        policy_loss = -(log_probs * torch.tensor(G).to(self.device)).mean()

        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


    def training_loop(self, log = True):

        ep_rewards = []
        reward_log = 0
        episode = 0
        for episode in tqdm.tqdm(range(self.num_episodes), total=self.num_episodes, desc = "Training Reinforce "):
            if episode == self.num_episodes - 1:
                game_obs = []
                u_sim_out = []
                model_out = []
                
            # reset the environment   
            if self.policy_type == "LSTM" or self.policy_type == "StackedMLP":
                self.buffer.reset()
            
            if self.sb:
                observation  = self.env.reset()
                xinit = np.array(observation["obs"][0][2:]) 
            else:
                observation, _ = self.env.reset(seed=self.seed)
                xinit = np.array(observation[0]["obs"][2:])
            
            self.u_sim.reset(xinit)

            done = False
            states, actions, rewards = [], [], []

            for t in range(MAXSTEPS):
                
                if self.sb:
                    obs = np.array(observation["obs"][0])
                else:
                    obs = np.array(observation[0]["obs"])

                # get simulator movements 
                move_action, click_action = self.u_sim.compute_displacement(obs[:2], obs[2:])

                # pertubate the movement if there is any perturbation
                if self.perturbator is not None:
                    move_action = self.perturbator(np.array(move_action))
                
                if self.policy_type == "LSTM" or self.policy_type == "StackedMLP":
                    self.buffer.add(torch.tensor(move_action))
                    state = self.buffer.get().unsqueeze(0).to(torch.float32).to(self.device)
                    if self.policy_type == "StackedMLP":
                        state = state.reshape(self.sequence_length * 2, )
                        means = self.mean_network(state)
                        log_stds = self.std_network(state)
                    else :
                        state = normalize(state)
                        means, _, _ = self.mean_network(state)
                        log_stds, _, _ = self.std_network(state)

                else:
                    state = torch.tensor(move_action, dtype=torch.float32).to(self.device)
                    means, log_stds = self.mean_network(state), self.std_network(state)
                     

                stds = torch.exp(log_stds)

                dist = torch.distributions.Normal(means, stds)
                corrector_action = dist.sample()

                # contruct msg to be send to the env
                # print("corrector actions : ",corrector_action)
                smartDart_action = np.insert(np.clip(corrector_action.to("cpu").numpy(), -80, 80), 0 , click_action)
                smartDart_action = np.array([ smartDart_action for _ in range(self.env.num_envs) ])
                

                if self.sb:
                    next_observation, reward, done, _ = self.env.step(smartDart_action)
                else:
                    next_observation, reward, done, _, _ = self.env.step(smartDart_action)
                # print("done : ", done)
                done = any(done)

                states.append(state)
                actions.append(corrector_action)
                rewards.append(reward)

                observation = next_observation

                # print("step : ",t, " reward : ", reward)
                if done:
                    break
                if t == MAXSTEPS - 1:
                    print("max steps reached : ", t)

                if episode == self.num_episodes - 1:
                    game_obs.append(obs)
                    u_sim_out.append(move_action)
                    model_out.append(corrector_action.cpu().numpy())

                    
            # print("done ! episode : ",episode)
            reward_log = np.sum(rewards)
            print("rewards summ at ep ", episode, " : ", reward_log)
            ep_rewards.append(reward_log)

            self.train_step(torch.stack(states).to(self.device), torch.stack(actions).to(self.device), rewards)

        if log:
            logger.info("loging training to : {self.log_path}")
            if not os.path.exists(self.log_path) and self.log:
                logger.info("creating log folder at : {self.log_path}")
                os.makedirs(self.log_path)
            logs = {"obs" : np.array(game_obs).tolist(), "u_sim" : np.array(u_sim_out).tolist(), "model" : np.array(model_out).tolist()}
            json.dump(logs, open(os.path.join(self.log_path, "logs.json"), "w"))    
            np.save(os.path.join(self.log_path, "ep_reward.npy"), np.array(ep_rewards))
            torch.save(self.std_network.state_dict(), os.path.join(self.log_path, "std_network.pt"))
            torch.save(self.mean_network.state_dict(), os.path.join(self.log_path, "mean_network.pt"))
            torch.onnx.export(self.mean_network, state, os.path.join(self.log_path, "mean_network.onnx"))
            torch.onnx.export(self.std_network, state, os.path.join(self.log_path, "std_network.onnx"))

        return ep_rewards, reward_log

                    
    def learn(self, log = False):
        print("Learning Start ...")
        print("Hyperparameters : ")
        print("learning rate : ", self.learning_rate)
        print("batch size : ", self.batch_size)
        print("num episodes : ", self.num_episodes)
        print("sequence length : ", self.sequence_length)
        print("policy type : ", self.policy_type)
        print("device : ", self.device)

        reward_list, reward = self.training_loop(log=log)


        # print("end of learning ", reward_list)
        return reward_list, reward
    

    def __call__(self, input):
        state = torch.tensor(input, dtype=torch.float32).to(self.device)
        means = self.mean_network(state)
        log_stds = self.std_network(state)
        stds = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, stds)
        return dist.sample()


class DDPGCorrector(Corrector):
    def __init__(self, env : GodotEnv, u_sim : UserSimulator, perturbator : Perturbator = None, learn = False, log = False,
                 policy_type = "StackedMLP", hidden_size = 512, actor_lr=1e-3, critic_lr=1e-3, gamma=0.85, tau=1e-4, decay_epsilon = 1, buffer_size=1e5, batch_size=64,
                 update_interval = 1, warmup_steps = 0):
        """

        Initialize the ReinforceCorrector with environment, user simulator, and training parameters.
        inspiration fhttps://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
        Args:
            env (GodotEnv): The Godot environment for running simulations
            u_sim (UserSimulator): User simulator for generating user movement actions
            perturbator (Perturbator, optional): Module to add perturbations to user actions. Defaults to None.
            learn (bool, optional): Whether the corrector is in learning mode. Defaults to False.
            log (bool, optional): Whether to enable logging of training data. Defaults to False.
            policy_type = ["StackedMLP", "MLP", "LSTM"]
        """
        super().__init__(learn)
        # Logging configuration
        self.log = log
        self.log_path = "logs_corrector/DDPG/" + time.strftime("%Y%m%d-%H%M%S") 

        # DDPG algorithm hyperparameters
        self.gamma = gamma  #s Discount factor for computing returns
        self.tau = tau  # Target network update rate
        self.act_lr = actor_lr  # Actor learning rate
        self.crit_lr = critic_lr  # Critic learning rate


        self.batch_size = batch_size  # Batch size (currently not used in implementation)
        self.buffer_size = int(buffer_size)  # Size of the replay buffer
        # Training configuration
        self.seed = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks for policy representation
        # Mean network: outputs the mean of the action distribution
        input_dim = 2
        self.action_dim = 2
        input_dim = 2
        if policy_type == "MLP":
            input_dim = 2
        elif policy_type == "StackedMLP":

            self.sequence_length = 10
            self.input_buffer = inputBuffer(input_dim=input_dim, maxlen=self.sequence_length)
            input_dim = 2*self.sequence_length
        else :
            logger.debug("unknow policy net type")
            raise NotImplementedError
        
        self.policy_type = policy_type
        if policy_type == "LSTM":
            print("unknow policy net type")
            raise NotImplementedError
        else:
            self.actor = DDPGActor(input_dim, self.action_dim, hidden_size, hidden_size).to(self.device)
            self.actor_target = DDPGActor(input_dim, self.action_dim, hidden_size, hidden_size).to(self.device)

            self.critic = DDPGCritic(input_dim, self.action_dim, hidden_size, hidden_size).to(self.device)
            self.critic_target = DDPGCritic(input_dim, self.action_dim, hidden_size, hidden_size).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.act_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.crit_lr)
        
        self.env = env  # Godot environment for smart darts simulation
        self.sb = isinstance(self.env, StableBaselinesGodotEnv)
        self.u_sim = u_sim  # User simulator for generating human-like movements
        self.perturbator = perturbator  # Optional noise/perturbation module
        
        self.replay_buffer = DDPGReplayBuffer(max_size=self.buffer_size, input_shape=input_dim, action_shape=self.action_dim)
        
        # exploration noise decay over episodes
        self.depislon = decay_epsilon  # decay espilon
        self.epsilon = 0
        self.warmup_steps = warmup_steps
        
        # training frequency
        self.update_freq = update_interval
        self.steps_per_episode = int(1e5)

        self.sb_env = isinstance(self.env, StableBaselinesGodotEnv)
        # print("is sb env ? ", self.sb_env)


    # def update_policy(self):
    #     # Get tensors from the batch
    #     state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size, device=self.device)

    #     # Get the actions and the state values to compute the targets
    #     next_action_batch = self.actor_target(next_state_batch)
    #     next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

    #     # Compute the target
    #     reward_batch = reward_batch.unsqueeze(1)
    #     done_batch = done_batch.unsqueeze(1)
    #     expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

    #     # TODO: Clipping the expected values here?
    #     # expected_value = torch.clamp(expected_value, min_value, max_value)

    #     # Update the critic network
    #     self.actor_optim.zero_grad()
    #     state_action_batch = self.critic(state_batch, action_batch)
    #     value_loss = nn.MSELoss()(state_action_batch, expected_values.detach())
    #     value_loss.backward()
    #     self.critic_optim.step()

    #     # Update the actor network
    #     self.actor_optim.zero_grad()
    #     policy_loss = -self.critic(state_batch, self.actor(state_batch))
    #     policy_loss = policy_loss.mean()
    #     policy_loss.backward()
    #     self.actor_optim.step()

    #     # Update the target networks
    #     soft_update(self.actor_target, self.actor, self.tau)
    #     soft_update(self.critic_target, self.critic, self.tau)

    #     return value_loss.item(), policy_loss.item()

    def update_policy(self):

        if len(self.replay_buffer) < self.batch_size:
            return
        # get batches from memory 
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = self.replay_buffer.sample(self.batch_size, device=self.device)

        # compute target Q values
        with torch.no_grad():
            # need to handkle the batch next states ... 
            next_action = self.actor_target(batch_next_states)
            next_q_values = self.critic_target(batch_next_states, next_action)
            target_q_values = batch_rewards + (1 - batch_terminals.float()) * self.gamma * next_q_values # Error here ! 

        # update critic
        
        q_batch = self.critic(batch_states, batch_actions)
        q_loss = nn.MSELoss()(q_batch, target_q_values)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # update actor
        self.actor.zero_grad()
        action = self.actor(batch_states)
        policy_loss = -self.critic(batch_states, action).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # update target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def act(self, state, noise=0.1):
        if torch.is_tensor(state):
            state = state.float().to(self.device)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.float().to(self.device)
        action = self.actor(state)
        action = action + noise * torch.tensor(np.random.randn(self.action_dim), dtype=torch.float).to(self.device)
        return action.detach()
    

    def learn(self, episodes):

        rewards = []
        
        episode_step = 0
        global_step = 0
        for episode in range(episodes):
        
            reward_of_episode = 0
            state = self.env.reset()

            state = read_obs(state, self.sb_env)
            u_sim_inital = np.array(state[2:])
            self.u_sim.reset(u_sim_inital)

            if self.perturbator is not None:
                u_sim_obs_displacement = self.perturbator(u_sim_obs_displacement)
                

            done = False
            if self.policy_type != "MLP":
                self.input_buffer.reset()
            # self.replay_buffer.reset_buffer()
            for step in range(self.steps_per_episode):
                
                if step == 0:
                    u_sim_obs_displacement, u_sim_click_action  = self.u_sim.compute_displacement(u_sim_inital[:2], state[2:]) 
                    if self.perturbator is not None:
                        u_sim_obs_displacement = self.perturbator(u_sim_obs_displacement)

                    if self.policy_type == "stackedMLP":     
                        self.input_buffer.add(torch.tensor(u_sim_obs_displacement))
                        current_usim_disps = self.input_buffer.get().reshape(1 ,self.sequence_length * 2)
                if global_step < self.warmup_steps:
                    # take random actions
                    action = torch.randn(self.action_dim).to(self.device)
                else:
                    if self.policy_type == "MLP":
                        action = self.act(torch.tensor(u_sim_obs_displacement), noise=self.epsilon)
                    else : 
                        action = self.act(current_usim_disps, noise=self.epsilon)
                    # and decay noise over time
                    if self.epsilon > 0:
                        self.epsilon -= self.depislon
                        self.epsilon = max(0, self.epsilon)
                
                
                smartDart_action = np.insert(np.clip(action.to("cpu").detach().numpy(), -80, 80), 0 , u_sim_click_action)
                smartDart_action = np.array([ smartDart_action for _ in range(self.env.num_envs) ])
                
                # logger.debug(f"smartDart_action {smartDart_action} "    )
                next_state, reward, done, _ = self.env.step(smartDart_action)
                reward_of_episode += reward
                # compute the next movement of user (next state)
                
                next_state = np.array(next_state["obs"][0])
                next_u_sim_obs_displacement, u_sim_click_action  = self.u_sim.compute_displacement(next_state[:2], next_state[2:])
                if self.perturbator is not None:
                    next_u_sim_obs_displacement = self.perturbator(u_sim_obs_displacement)

                 
                # observe the transisions
                if self.policy_type == "stackedMLP":
                    self.input_buffer.add(torch.tensor(next_u_sim_obs_displacement))
                    next_u_sim_obs_displacement = deepcopy(self.input_buffer.get()).reshape(1 ,self.sequence_length * 2)
                    u_sim_obs_displacement = deepcopy(current_usim_disps).reshape(1 ,self.sequence_length * 2)
                else :
                    u_sim_obs_displacement = next_u_sim_obs_displacement
                self.replay_buffer.store_transition(u_sim_obs_displacement, action, torch.tensor(reward).to(self.device), next_u_sim_obs_displacement, False)
                # update policy
                if global_step > self.warmup_steps and global_step % self.update_freq == 0:
                    self.update_policy()

                if self.sb_env:
                    done = done
                else :
                    done = done[0]
                if done:
                    # logger.debug(f"replay buffer size = {self.replay_buffer.terminal_memory}")
                    # logger.debug(f"done  = {done}")

                    self.replay_buffer.store_transition(u_sim_obs_displacement, action, torch.tensor(reward).to(self.device), next_u_sim_obs_displacement, True)
                    episode_step = 0
                    break   

                if self.policy_type != "stackedMLP":
                    current_usim_disps = deepcopy(next_u_sim_obs_displacement)
                else:
                    u_sim_obs_displacement = deepcopy(next_u_sim_obs_displacement)
                
                global_step += 1
                episode_step += 1

            logger.debug(f"episode {episode} reward {reward_of_episode} done {done}")
            rewards.append(reward_of_episode)
            # self.replay_buffer.store_transition(u_sim_obs_displacement, action, reward, next_u_sim_obs_displacement, done)
            # self.replay_buffer.store_transition(u_sim_obs_displacement, action, reward, next_u_sim_obs_displacement, done)

        return rewards, reward_of_episode


    def training_loop(self, num_episode):
        
        rewards = []

        global_step = 0
        for i in range(num_episode):
            
            current_episode_steps = 0
            reward_episode = 0
            done = False

            # reset the environment
            if self.sb:
                current_state = self.env.reset()
            else:
                current_state, _ = self.env.reset()

            current_state = read_obs(current_state, self.sb)
            # reset u_sim for current episode
            
            self.u_sim.reset(current_state[2:])

            # compute first u_sim movement 
            u_sim_displacement, u_sim_click_action = self.u_sim.step(current_state[2:], current_state[:2], self.perturbator)

            # episode loop
            for t in range(self.steps_per_episode):
                    

                # observe state and select action
                action_corrector = self.act(u_sim_displacement, self.epsilon)
                self.decay_epsilon()

                # step environment
                msg_step = action_to_msg(action_corrector, u_sim_click_action, self.env.num_envs)
                if self.sb :
                    next_state, reward, done, _ = self.env.step(msg_step)
                else :
                    next_state, reward, done, _, _ = self.env.step(msg_step)
                    reward = reward[0]

                next_state = read_obs(next_state, self.sb)
                # compute next u_sim movement
                next_u_sim_displacement, u_sim_click_action = self.u_sim.step(next_state[2:], next_state[:2], self.perturbator)
                
                # print("reward : ", reward)
                # observe the transition
                reward_episode += reward
                self.replay_buffer.store_transition(torch.tensor(u_sim_displacement), action_corrector, torch.tensor(reward).to(self.device), torch.tensor(next_u_sim_displacement), done)

                # update policy
                self.update_policy()

                # if  global_step % self.update_freq == 0 & global_step > 0:
                #     self.update_policy()

                u_sim_displacement = next_u_sim_displacement
                current_episode_steps += 1
                global_step += 1

                if any(done):

                    # logger.debug(f"replay buffer size = {self.replay_buffer.terminal_memory}")
                    logger.debug(f"done  = {done}")
                    break

            logger.debug(f"episode {i} reward {reward_episode} done {done}, episodes steps {current_episode_steps}")
            rewards.append(reward_episode)

        return rewards, reward_episode


    def decay_epsilon(self):
        if self.epsilon > 0:
            self.epsilon -= self.depislon
            self.epsilon = max(0, self.epsilon)

if __name__ == "__main__":
    

        # create a perturbation
    perturbator = NormalJittering(0, 20)
    # perturbator = None


    # create a corrector
    corrector = None
    # corrector = LowPassCorrector(5)

    # Initialize the environment
    env = GodotEnv(convert_action_space=True)
    
    
    u_sim = VITE_USim([0, 0])

    corr = ReinforceCorrector(env, u_sim, perturbator, learn = True, log = True)
    # corr.training_loop(Corrector)
    corr.training_loop()

    env.close()