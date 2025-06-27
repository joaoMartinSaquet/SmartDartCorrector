from classic_rl import rl_corrector
import gymnasium as gym

env = gym.make("MountainCarContinuous-v0")  # default goal_velocity=0

corrector = rl_corrector.PPOSB3Corrector(env, None) 
rlist  = corrector.train(10)

