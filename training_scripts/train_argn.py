from pathlib import Path
import sys
import os

import pandas as pd
from loguru import logger
import wandb
import pprint
from functools import partial
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Queue, Manager

# ---------------------------------------------------------------------------
# Repo‑local imports – SmartDart env
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from common.rolloutenv import *
from common.perturbation import *
# from GA.pyGRN.pygrn import problems, evolution, grns
from agrn import  EATMuPlusLambda, gymProblem, GRN
from agrn import   utils

from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
class grnSmartDart():

    def __init__(self, start_nreg, perturbator, reward_shaping, normalize, nenv = 1):
        # super().__init__(env_info, env_name, max_ts)

        self.perturbator = perturbator
        self.reward_shaping = reward_shaping
        self.normalize = normalize
        self.nin = 2
        self.nout = 2*2
        self.nreg = start_nreg
    
        self.n_env = nenv
        self.envs = [smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=self.perturbator, render=False, reward_shape=self.reward_shaping, normalize=self.normalize) for _ in range(nenv)]
        self.current_index = 0
        # self.env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=self.perturbator, render=False, reward_shape=self.reward_shaping, normalize=self.normaliz


    def eval(self, genome, render=False, testing = False ):

        while len(self.envs) < 1:
            pass # waiting for an environent to be freed 
        env = self.envs.pop()

        rgathered = self.run_env(env, genome, testing = testing)
        self.envs.append(env)
        return rgathered

    def run_env(self, env, genome, testing = False):
        # print("environement created ", env)
        g = GRN(genome, self.nin, self.nout)
        g.setup()
        g.warmup(25)
        if testing: 
            actions = []
            obss = []

        # print("running the environment")
        obs, info = env.reset()
        fit = 0
        done = False
        while not done:
            # print("step")
            obs_grn = (obs/MAX_DISP +    1)/2 
            g.set_input(obs_grn)
            g.step(10)
            output_concentrations = g.get_output().tolist()
            # action = utils.compute_output_concentrations_diff(output_concentrations)
            # action = utils.compute_output_concentrations_diff(output_concentrations)
            dx = (output_concentrations[0] - output_concentrations[1])
            dy = (output_concentrations[2] - output_concentrations[3])
            action = np.array([dx, dy])*80

            # r = action[0]*40
            # theta = action[1] * 2*np.pi - np.pi

            # action = [r*np.cos(theta), r*np.sin(theta)]
            # action = action * 

            if testing:
                actions.append(action)
                obss.append(obs)

            obs, reward, done, truncated, terminated, = env.step(action)
            
            fit += reward
            # fit += -np.linalg.norm(obs[:2]-action) 
            if truncated or terminated: 
                done = True
            # print("step, done = ", done)
        
        # print("fit = ", fit)
        if testing:
            player_positions = np.array(env.player_positions)
       

        if testing :

            return fit, player_positions, actions, obss
        else :
            return fit, 
    def close_envs(self):
        for env in self.envs:
            env.close()

def main(args):
    reward_shaping = args.reward_shaping
    normalize = args.normalize
    ngen = args.ngen

    perturbator = None
    if args.perturbator == 'None':
        perturbator = None
    elif args.perturbator == 'RAM':
        perturbator = None
        logger.warning(f"Not implemented yet: {args.perturbator}")
    elif args.perturbator == 'Noise':
        perturbator = NormalJittering(10, 20)
        
        
    p = grnSmartDart(0, perturbator, reward_shaping, normalize, nenv = args.n_envs)

    e = EATMuPlusLambda(nin=p.nin, nout=p.nout, nreg=0, eval_fun = p.eval, nproc=args.n_envs-1, log_interval=5, log_name='grn.json')
    hof, hist_run  = e.run(100, 90, 500)   
    e.visualize_evolutions()


    print("best genome is ", hof[0])    
    best = GRN(hof[0], p.nin, p.nout)
    fits = []
    pps = []
    test_env = smartDartEnv(VITE_USim(x_init=[0,0]), perturbator=perturbator, render=True, reward_shape=False, normalize=normalize)


    for i in range(10):
        fit, pp, _, _ = p.run_env(test_env, hof[0], testing=True)
        pps.append(pp)
        fits.append(fit)


    print("fits are ", fits)
    print("mean fit is ", np.mean(fits))
    print("std fit is ", np.std(fits))
    print("min fit is ", np.min(fits))
    print("max fit is ", np.max(fits))
    # logs the best fit
    # dict_log = {"best_genome": str(hof[0]), 'best_train_fit': hof[0].fitness.values[0], 'best_test_fit': np.max(fits)}
    
    # pd.DataFrame(dict_log, index=[0]).to_csv(f"logs/GRN_training.csv", index=False)
    


    max_fit_ind = np.argmax(fits)
    # plt.plot(pp[:, 0], pp[:,1], '.')
    plt.plot(pps[max_fit_ind][:, 0], pps[max_fit_ind][:,1], 'r.')
    plt.show()




if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        description="Train GRN on SmartDartCorrector or run Optuna tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment params
    parser.add_argument("--n-envs", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--reward_shaping", action=argparse.BooleanOptionalAction, default=True)
    # CGP args
    parser.add_argument("--ngen", type=int, default=50)
    
    # smartDarts
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation during tuning")
    parser.add_argument("--perturbator", choices=['None', 'Noise'], default='None')
    parser.add_argument("--nstack", type=int, default=1)

    args = parser.parse_args()

    main(args)
