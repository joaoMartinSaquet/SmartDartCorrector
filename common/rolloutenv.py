from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv 
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import tqdm
from loguru import logger
from collections import deque
import torch

from common.user_simulator import *
from common.perturbation import *


MAX_DISP = 40
MAX_TS = 1e3
GAME_PATH = "games/SmartDartEnvNormalized/smartDartEnv.x86_64"
# GAME_PATH = "games/SmartDartPlusDist/smartDartEnv.x86_64"

class Buffer:
    """Inpired from stable_baselines3
    """
    
    def __init__(self):
        self.observations = []
        self.rewards = []
        self.dones = []

    def store(self, observation, reward, done, info):
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
    

    def reset(self):
        self.observations = []
        self.rewards = []
        self.dones = []

    def get(self):
        return np.array(self.observations), np.array(self.rewards), np.array(self.dones)
    

def obs_handling(obs, sb_env):
    if sb_env:
        return obs["obs"]
    else:
        # obs = np.array(obs"obs"])
        return np.array(obs[0]["obs"])



def stepSmartDartEnv(env, obs, u_simulator : UserSimulator, perturbator : Perturbator, corrector = None):
    """
        Does not work we need to place our corrector inside... 
    """
    obs = np.array(observation[0]["obs"])
    move_action, click_action = u_simulator.compute_displacement(obs[:2], obs[2:])
    
    # clamp action to don't have to big displacement
    move_action = np.clip(move_action, -MAX_DISP, MAX_DISP)

    # add perturbation and corrector
    if perturbator is not None:
        observation = perturbator(observation)
    if corrector is not None:
        observation = corrector(observation)
    

    # contruct msg to be send to the env
    action = np.insert(move_action, 0 , click_action)
    action = np.array([ action for _ in range(env.num_envs) ])

    observation, reward, done, info, _ = env.step(action)

    return observation, reward, done, info

def rolloutSmartDartEnv(env, Nstep, pertubator : Perturbator, corrector = None, seed = 0, log = 0):

    num_envs = env.num_envs
    sb = isinstance(env, StableBaselinesGodotEnv)
    player_positions = []
    if sb:
        observation = env.reset()
    else:
        observation, _ = env.reset(seed=seed)
    
    obs = obs_handling(observation, sb)[0]
    xinit = np.array(obs[2:])
    player_positions.append(obs[2:])
    u_simulator = VITE_USim(xinit)
    
    perturbator = pertubator
    reward_list = []
    # rolling out env
    for i in range(Nstep):
        # get controller actions and process it (clamp, norm, pert, etc...)
        move_action, click_action = u_simulator.compute_displacement(obs[:2], obs[2:])

        # convert moveaction to numpy
        move_action = np.array(move_action)
        click_action = np.array(click_action)

        # add perturbation if there is any
        if perturbator is not None:
            if log > 2: logger.debug("RolloutSmartDartEnv perturbator input = ", move_action)
            move_action = perturbator(move_action)
            if log > 2: logger.debug("RolloutSmartDartEnv  perturbator output = ", move_action)
        if corrector is not None:
            if log > 3: 
                logger.debug("RolloutSmartDartEnv corrector input = ", move_action)
                logger.debug("RolloutSmartDartEnv corrector input shape = ", move_action.shape)
            move_action = corrector(move_action)
            if log > 3: logger.debug("RolloutSmartDartEnv corrector output = ", move_action)


        # clamp action to don't have to big displacement
        move_action = np.clip(move_action, -MAX_DISP, MAX_DISP) 
        
        # contruct msg to be send to the env
        action = np.insert(move_action, 0 , click_action)
        action = np.array([ action for _ in range(num_envs) ])

        # step the env
        if log > 3:
            logger.debug("RolloutSmartDartEnv  action sended at step {i}, action = {action}".format(i = i, action = action))
        
        if sb:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done, info, _ = env.step(action)

        obs = obs_handling(observation, sb)[0]
        player_positions.append(obs[2:])
        # update reward list
        reward_list.append(reward)

        if log > 3:
            logger.debug("done , reward = ", done, reward)
        # see how to do this with several env 
        if any(done):
            if log > 0:
                logger.debug("done")
            break
    if log > 0:
        logger.debug("RolloutSmartDartEnv reward list = ", np.sum(reward_list),)
    return np.sum(reward_list), reward_list, player_positions

        

def rolloutMultiSmartDartEnv(env, Nstep, pertubator : Perturbator, corrector = None, seed = 0):

    num_envs = env.num_envs

    sb_env = isinstance(env, StableBaselinesGodotEnv)
    if  sb_env:
        observation = env.reset()
    else : 
        observation, _ = env.reset()
    
    logger.debug("observation ", observation)
    # initialize controller
    # xinit = np.array(observation[0]["obs"][2:] + [0, 0]) 
    # get all xinit
    if sb_env:
        xinit = [np.array(observation["obs"][k][2:] + [0, 0]) for k in range(num_envs)]
    else :
        xinit = [np.array(observation[k]["obs"][2:]) for k in range(num_envs)]
    u_simulators = [VITE_USim(xinit) for _ in range(num_envs)]
    
    perturbator = pertubator
    reward_list = []
    # rolling out env
    for i in tqdm.tqdm(range(Nstep)):


        # get controller actions and process it (clamp, norm, pert, etc...)
        move_actions = []
        click_actions = []
        for k, u_sim in zip(range(num_envs), u_simulators):
            if sb_env:
                obs = np.array(observation["obs"][k])
            else :
                obs = np.array(observation[k]["obs"])
            move_action, click_action = u_sim.compute_displacement(obs[:2], obs[2:])
            move_actions.append(move_action)
            click_actions.append(click_action)

        

        # add perturbation if there is any
        if perturbator is not None:
            for k in range(num_envs):
                move_actions[k] = perturbator(np.array(move_actions[k]))


        if corrector is not None:
            for k in range(num_envs):
                move_actions[k] = corrector(move_actions[k])
                
        # clamp action to don't have to big displacement
        move_actions = np.clip(move_actions, -MAX_DISP, MAX_DISP) 

        # contruct msg to be send to the env
        action = np.hstack((np.array([click_actions]).T, move_actions))
        # step the env
        # print("action sended at step {i}, action = {action}".format(i = i, action = action))
        if sb_env:
            observation, reward, done, _ = env.step(action)
        else :
            observation, reward, done, info, _ = env.step(action)
            reward = reward[0]
        # print("observations = ", observation)
        # update reward list
        reward_list.append(reward)


        # print("done , reward = ", done, reward)
        # see how to do this with several env 
        if any(done):
            # print("done")
            break

    return np.cumsum(reward_list), reward_list

def action_to_msg(displacement, click, num_envs = 1):
    if torch.is_tensor(displacement):
        displacement = displacement.numpy()

    displacement = np.clip(displacement, -MAX_DISP, MAX_DISP),
    action = np.insert(displacement, 0 , click)
    action = np.array([ action for _ in range(num_envs) ])
    return action


def normalize(x):
    return (x + MAX_DISP)/(2*MAX_DISP)

if __name__ == "__main__":
    
    N = 1
    # create a perturbation
    # perturbator = NormalJittering(10, 20)
    perturbator = None

    # create a corrector
    corrector = None
    # corrector = LowPassCorrector(5)
    
    # Initialize the environment
    env = GodotEnv(convert_action_space=True)

    logger.debug("env created")
    logger.debug("env number is : ", env.num_envs)

    for j in range(N):
        logger.debug("ep : ", j)
        # Run the environment
        if env.num_envs > 1:
            r_summ, r_list = rolloutMultiSmartDartEnv(env, 10000, perturbator, corrector)
            logger.debug("reward summ = ", r_summ[-1])
        else:
            r_summ, r_list = rolloutSmartDartEnv(env, 10000, perturbator, corrector)    
        logger.debug("reward summ = ", r_summ[-1])
    
    # closing environment
    env.close()
    


def read_obs(obs, sb_env : bool):
    if sb_env:
        obs = np.array(obs["obs"][0])
    else:
        obs = np.array(obs[0]["obs"])

    return obs


class smartDartEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self, usim=None, perturbator=None, render = False, n_stack : int = 1, n_parallel=1, normalize: bool =False, port = None, reward_shape = 1, game_path = GAME_PATH, speedup = None):
        super(smartDartEnv, self).__init__()
        base_obs_shape  = tuple(map(lambda x: x * n_stack, (2, )))
        self.action_space = spaces.Box(low=-MAX_DISP, high=MAX_DISP, shape=(2, ) , dtype=np.float32)
        self.observation_space = spaces.Box(low=-MAX_DISP, high=MAX_DISP, shape=base_obs_shape, dtype=np.float32)
        if port == None:
            self.godot_env = StableBaselinesGodotEnv(game_path, num_envs=n_parallel, show_window=render, speedup=speedup)
        else :
            self.godot_env = StableBaselinesGodotEnv(game_path, num_envs=n_parallel, show_window=render, port=port, speedup=speedup)
        
        self.sb = isinstance(self.godot_env, StableBaselinesGodotEnv)

        self.player_positions = []
        self.observations = deque(maxlen=n_stack)
        self.usim = usim
        self.perturbator = perturbator
        self.normalize = normalize
        self.reward_shape = reward_shape
        self.info = {"episode" : 0}
        # self.click = 0

    def reset(self, seed=None):
        
        self.player_positions.clear()

        game_obs = self.godot_env.reset()
        game_obs = obs_handling(game_obs, self.sb)[0]
        user_state_initial = np.array(game_obs[2:]) 
        self.player_positions.append(game_obs[2:])
        self.usim.reset(user_state_initial)
        move_action, self.click =self.usim.step(game_obs[0:2], game_obs[2:], self.perturbator)

        if self.normalize:


            dx = move_action[0]
            dy = move_action[1]

            mag = np.linalg.norm([dx, dy], 2)/MAX_DISP
            theta = np.arctan2(dy, dx)

            move_action = np.array([mag * np.cos(theta), mag * np.sin(theta)])
            
            # move_action = move_action / MAX_DISP
        self.observations.clear()
        for _ in range(self.observations.maxlen):
            self.observations.append(np.zeros(move_action.shape))
        self.observations.append(move_action)
        obs = self.get_obs()
        self.ts = 0
        self.reward = 0
        # self.info["episode"] = 0
        return obs, {}
    
    def close(self):
        self.godot_env.close()
        return super().close()
    
    def step(self, move_action):
        move_action = move_action
        game_obs, reward, done, info = self.godot_env.step(action_to_msg(move_action, self.click))

        game_obs = obs_handling(game_obs, self.sb)[0]
        self.player_positions.append(game_obs[2:])
        new_reward = 0
        if self.reward_shape ==1:
            # get distance between target and player
            dist = np.linalg.norm(game_obs[:2] - game_obs[2:])
            # normalize the distance to not get too big negative rewards
            dist /= 1e5
            # the target has been hitted
            if reward > 0:
                new_reward = [10]
            # out of board and reset
            elif reward < 0:
                new_reward = [-10]
            # no progress we should remove the steps
            elif reward == 0:
                new_reward = [-dist]

        if self.reward_shape == 2:
            pass
            # compute the variance of the positions of the player
            # first we need to get 

        if self.reward_shape: 
            reward = new_reward
        self.reward += reward[0]
        move_action, self.click = self.usim.step(game_obs[:2], game_obs[2:], self.perturbator)

        if self.normalize:
            move_action = move_action / MAX_DISP
        self.observations.append(move_action)
        obs = self.get_obs()
        self.ts += 1

        done = self.ts >= MAX_TS or done

        info = {}
        if done:
            # implement it here
            # loog at the jittering 
            info["episode"] = {"r": self.reward, "l": self.ts}
        
        return obs, reward[0], done, False, info

    def render(self):
        pass
    

    def set_usim(self, usim):
        self.usim = usim

    def set_perturbator(self, perturbator):
        self.perturbator = perturbator


    def get_obs(self):
        return np.concatenate(self.observations)