from godot_rl.core.godot_env import GodotEnv
from gymnasium import spaces
import numpy as np
import tqdm

from user_simulator import *
from perturbation import *
from corrector import *


MAX_DISP = 40


def stepSmartDartEnv(env, obs, u_simulator : UserSimulator, perturbator : Perturbator, corrector = None):
    """
        Does'nt work we need to place our corrector inside... 
    """
    obs = np.array(observation[0]["obs"])
    move_action, click_action = u_simulator.step(obs[:2], obs[2:])
    
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

def rolloutSmartDartEnv(env, Nstep, pertubator : Perturbator, corrector = None, seed = 0):

    observation, info = env.reset(seed=seed)
    
    # initialize controller
    # xinit = np.array(observation[0]["obs"][2:] + [0, 0]) 
    xinit = np.array(observation[0]["obs"][2:]) 

    u_simulator = VITE_USim(xinit)
    
    perturbator = pertubator
    reward_list = []
    # rolling out env
    for i in tqdm.tqdm(range(Nstep)):
        # get controller actions and process it (clamp, norm, pert, etc...)
        obs = np.array(observation[0]["obs"])
        move_action, click_action = u_simulator.step(obs[:2], obs[2:])

        # add perturbation if there is any
        if perturbator is not None:
            move_action = perturbator(move_action)

        if corrector is not None:
            move_action = corrector(move_action)


        # clamp action to don't have to big displacement
        move_action = np.clip(move_action, -MAX_DISP, MAX_DISP) 

        # contruct msg to be send to the env
        action = np.insert(move_action, 0 , click_action)
        action = np.array([ action for _ in range(env.num_envs) ])

        # step the env
        # print("action sended at step {i}, action = {action}".format(i = i, action = action))
        observation, reward, done, info, _ = env.step(action)
        # print("observations = ", observation)
        # update reward list
        reward_list.append(reward)


        # print("done , reward = ", done, reward)
        # see how to do this with several env 
        if any(done):
            # print("done")
            break

    return np.cumsum(reward_list), reward_list

        

def rolloutMultiSmartDartEnv(env, Nstep, pertubator : Perturbator, corrector = None, seed = 0):

    num_envs = env.num_envs

    observation, info = env.reset(seed=seed)
    
    # initialize controller
    # xinit = np.array(observation[0]["obs"][2:] + [0, 0]) 
    # get all xinit

    u_simulators = [VITE_USim(observation[0]["obs"][2:] + [0, 0]) for _ in range(num_envs)]
    
    perturbator = pertubator
    reward_list = []
    # rolling out env
    for i in tqdm.tqdm(range(Nstep)):
        # get controller actions and process it (clamp, norm, pert, etc...)
        
        obs = np.array(observation[0]["obs"])
        move_actions = []
        click_actions = []
        for k, u_sim in zip(range(num_envs), u_simulators):
            obs = np.array(observation[k]["obs"])
            move_action, click_action = u_sim.step(obs[:2], obs[2:])
            move_actions.append(move_action)
            click_actions.append(click_action)

        

        # add perturbation if there is any
        if perturbator is not None:
            for k in range(num_envs):
                move_actions[k] = perturbator(move_actions[k])


        if corrector is not None:
            for k in range(num_envs):
                move_actions[k] = corrector(move_actions[k])
                
        # clamp action to don't have to big displacement
        move_actions = np.clip(move_actions, -MAX_DISP, MAX_DISP) 

        # contruct msg to be send to the env
        action = np.hstack((np.array([click_actions]).T, move_actions))
        # step the env
        # print("action sended at step {i}, action = {action}".format(i = i, action = action))
        observation, reward, done, info, _ = env.step(action)
        # print("observations = ", observation)
        # update reward list
        reward_list.append(reward)


        # print("done , reward = ", done, reward)
        # see how to do this with several env 
        if any(done):
            # print("done")
            break

    return np.cumsum(reward_list), reward_list


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

    print("env created")
    print("env number is : ", env.num_envs)

    for j in range(N):
        print("ep : ", j)
        # Run the environment
        if env.num_envs > 1:
            r_summ, r_list = rolloutMultiSmartDartEnv(env, 10000, perturbator, corrector)
            print("reward summ = ", r_summ[-1])
        else:
            r_summ, r_list = rolloutSmartDartEnv(env, 10000, perturbator, corrector)    
        print("reward summ = ", r_summ[-1])
    
    # closing environment
    env.close()
    


        
