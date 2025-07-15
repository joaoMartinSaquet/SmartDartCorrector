from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from common.rolloutenv import *
from common.perturbation import *

from classic_rl.rl_corrector import *

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv 
import pandas as pd

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run USIM and the environment')
    parser.add_argument('--perturbator', choices=['None', 'RAM', 'Noise'], required=True)
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--perturbation_std', type=float, default=20.0,
                       help='Standard deviation for normal jittering perturbation (default: 20.0)')
    parser.add_argument('--perturbation_bias', type=float, default=0,
                       help='bias ')
    

    env_path = GAME_PATH
    args = parser.parse_args()
    N = args.N
    # create a perturbation
    # perturbator = NormalJittering(10, 20)
    if args.perturbator == 'None':
        perturbator = None
        print("Training without perturbation")
    elif args.perturbator == 'RAM':
        print(f"Not implemented yet: {args.method}")
    elif args.perturbator == 'Noise':
        perturbator = NormalJittering(args.perturbation_bias, args.perturbation_std)
        print(f"Training with normal jittering (std={args.perturbation_std})")

    # create a corrector
    corrector = None
    # corrector = LowPassCorrector(5)
    
    # Initialize the environment
    # env = GodotEnv(convert_action_space=True)
    env = StableBaselinesGodotEnv(env_path=env_path, n_parallel=1)

    print("env created")
    print("env number is : ", env.num_envs)
    print("RolloutMultiSmartDartEnv Env = ", env.envs[0])

    rewards = []
    for j in range(N):
        print("ep : ", j)
        # Run the environment
        if env.num_envs > 1:
            
            r_sum, r_list = rolloutMultiSmartDartEnv(env, 10000, perturbator, corrector)
            # print("reward summ = ", r_summ[-1])
        else:
            r_sum, r_list, player_pos = rolloutSmartDartEnv(env, 10000, perturbator, corrector) 

        

        print("reward summ = ", r_sum)
        rewards.append(r_sum)

    df = pd.DataFrame(rewards)
    print(f"logging to u_sim{args.perturbator}_{args.perturbation_std}_{args.perturbation_bias}.csv")
    df.to_csv(f"{N}_u_sim{args.perturbator}_{args.perturbation_std}_{args.perturbation_bias}.csv")
    # closing environment
    env.close()