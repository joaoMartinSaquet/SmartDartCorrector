from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv 
from gymnasium import spaces
import numpy as np
import tqdm
from loguru import logger


from common.user_simulator import *
from common.perturbation import *


MAX_DISP = 40

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
        return obs



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

    observation, info = env.reset()
    xinit = np.array(observation[0]["obs"][2:])
    
    u_simulator = VITE_USim(xinit)
    
    perturbator = pertubator
    reward_list = []
    # rolling out env
    for i in range(Nstep):
        # get controller actions and process it (clamp, norm, pert, etc...)
        obs = np.array(observation[0]["obs"])
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
        
        observation, reward, done, info, _ = env.step(action)

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
    return np.sum(reward_list), reward_list

        

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
    displacement = np.clip(displacement.to("cpu").detach().numpy(), -MAX_DISP, MAX_DISP),
    action = np.insert(displacement, 0 , click)
    action = np.array([ action for _ in range(num_envs) ])
    return action


if __name__ == "__main__":
    
    N = 1
    # create a perturbation
    # perturbator = NormalJittering(0, 20)
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
        return np.array(obs["obs"][0])
    else:
        return np.array(obs[0]["obs"])