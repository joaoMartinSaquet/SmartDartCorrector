
from common.rolloutenv import *
from common.perturbation import *
from classic_rl.corrector import *

if __name__ == "__main__":
    

    # create a perturbation
    # perturbator = NormalJittering(0, 20)
    perturbator = None


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