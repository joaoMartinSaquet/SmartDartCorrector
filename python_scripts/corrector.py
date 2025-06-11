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


from deep_stuff import networks
from user_simulator import *
from perturbation import *
from rolloutenv import *

# steps where we say, that's enough reset yourselves
MAXSTEPS =int(1e6)

class Corrector():
    def __init__(self, learn = False):
        self.learning = learn
        self.feedback_list = []
        self.step = 0

    def reset(self):

        self.step = 0
        self.feedback_list = []


    def __call__(self, input):
        return input
    

    def get_feedback(self, fb):
        """function register the feedback that occurs inside the env (reward, done, etc...)

        Args:
            fb (_type_): _description_
        """

        self.feedback_list = []



class LowPassCorrector(Corrector):
    ''' averages overs 5 steps and then applies the correction'''
    def __init__(self, nsteps = 5,learn = False):
        super().__init__(learn)
        self.feedback_list = []
        self.step = 0
        self.inputs = deque(maxlen = nsteps)
        self.nsteps = nsteps

    def __call__(self, input):
        self.inputs.append(input)

        if len(self.inputs) == self.nsteps:
            return np.mean(self.inputs, axis=0)
        else:
            return input    
        

        
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
    def __init__(self, env : GodotEnv, u_sim : UserSimulator, perturbator : Perturbator = None, learn = False, log = False):
        """
        Initialize the ReinforceCorrector with environment, user simulator, and training parameters.
        
        Args:
            env (GodotEnv): The Godot environment for running simulations
            u_sim (UserSimulator): User simulator for generating user movement actions
            perturbator (Perturbator, optional): Module to add perturbations to user actions. Defaults to None.
            learn (bool, optional): Whether the corrector is in learning mode. Defaults to False.
            log (bool, optional): Whether to enable logging of training data. Defaults to False.
        """
        super().__init__(learn)
        # Logging configuration
        self.log = log
        self.log_path = "logs_corrector/Reinforce/" + time.strftime("%Y%m%d-%H%M%S") 
        if not os.path.exists(self.log_path) and self.log:
            print("creating log folder at : ", self.log_path)
            os.makedirs(self.log_path)

        # REINFORCE algorithm hyperparameters
        self.gamma = 0.99  # Discount factor for computing returns
        self.learning_rate = 0.01  # Learning rate for neural network optimization
        self.num_episodes = 50  # Number of training episodes
        self.batch_size = 64  # Batch size (currently not used in implementation)

        # Training configuration
        self.seed = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks for policy representation
        # Mean network: outputs the mean of the action distribution
        self.mean_network = networks(n_input = 2, n_output = 2, layers = [32, 32]).to(self.device)
        # Standard deviation network: outputs log std dev of the action distribution
        self.std_network = networks(n_input = 2, n_output = 2, layers = [32, 32]).to(self.device)

        # Optimizer for both networks (combined parameter list)
        self.optimizer = optim.Adam(list(self.mean_network.parameters()) + list(self.std_network.parameters()), lr=self.learning_rate)

        # Environment and simulation components
        self.env = env  # Godot environment for smart darts simulation
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
        
        G = self.compute_return(rewards)
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


    def training_loop(self):

        ep_reward = []
        reward_log = 0
        episode = 0
        for episode in tqdm.tqdm(range(self.num_episodes), desc = "Training Reinforce ep {}/{} : reward {}".format(episode, self.num_episodes, reward_log)):

            if episode == self.num_episodes - 1:
                game_obs = []
                u_sim_out = []
                model_out = []
            print("reset env , episode : ",episode)
            observation, info = self.env.reset(seed=self.seed)
            xinit = np.array(observation[0]["obs"][2:]) 

            self.u_sim.reset(xinit)

            done = False
            states, actions, rewards = [], [], []

            for t in range(MAXSTEPS):
                    
                obs = np.array(observation[0]["obs"])

                # get simulator movements 
                move_action, click_action = self.u_sim.step(obs[:2], obs[2:])

                # pertubate the movement if there is any perturbation
                if self.perturbator is not None:
                    move_action = self.perturbator(np.array(move_action))
                state = torch.tensor(move_action, dtype=torch.float32).to(self.device)
                means = self.mean_network(state)
                log_stds = self.std_network(state)

                stds = torch.exp(log_stds)

                dist = torch.distributions.Normal(means, stds)
                corrector_action = dist.sample()

                # contruct msg to be send to the env
                # print("corrector actions : ",corrector_action)
                smartDart_action = np.insert(np.clip(corrector_action.to("cpu").numpy(), -80, 80), 0 , click_action)
                smartDart_action = np.array([ smartDart_action for _ in range(self.env.num_envs) ])
                


                next_observation, reward, done, info, _ = self.env.step(smartDart_action)
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

            print("rewards summ at ep ", episode, " : ", )
            ep_reward.append(np.sum(rewards))

            self.train_step(torch.stack(states).to(self.device), torch.stack(actions).to(self.device), rewards)
            reward_log = np.sum(rewards)
        if self.log:
            print("loging training to : ", self.log_path)
            logs = {"obs" : np.array(game_obs).tolist(), "u_sim" : np.array(u_sim_out).tolist(), "model" : np.array(model_out).tolist()}
            json.dump(logs, open(os.path.join(self.log_path, "logs.json"), "w"))    
            np.save(os.path.join(self.log_path, "ep_reward.npy"), np.array(ep_reward))
            torch.save(self.mean_network.state_dict(), os.path.join(self.log_path, "mean_network.pt"))
            torch.save(self.std_network.state_dict(), os.path.join(self.log_path, "std_network.pt"))
                    

class CartesianGeneticCorrector(Corrector):

    def __init__(self, env : GodotEnv, u_sim : UserSimulator, perturbator : Perturbator = None, learn = False, log = False):
        super().__init__(env, u_sim, perturbator, learn, log)
        self.log = log
        self.log_path = "logs_corrector/CGP/" + time.strftime("%Y%m%d-%H%M%S") 
        if not os.path.exists(self.log_path) and self.log:
            print("creating log folder at : ", self.log_path)
            os.makedirs(self.log_path)
        
        self.env = env 
        self.u_sim = u_sim
        self.perturbator = perturbator

    def fitness_function(self, individual):
        
        # rollout the env with the individual
        # return the reward
        # return the reward
        cum_reward, _ = rolloutSmartDartEnv(self.env, MAXSTEPS, self.perturbator, corrector = individual)
        
        return cum_reward[-1]
        
        







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