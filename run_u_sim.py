from common.rolloutenv import *
from common.perturbation import *
from classic_rl.rl_corrector import *


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