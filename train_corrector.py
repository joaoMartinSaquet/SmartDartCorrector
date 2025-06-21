import argparse
from common.rolloutenv import *
from common.perturbation import *
from classic_rl.rl_corrector import *
from GA.cgp_corrector import *

import wandb
import pprint
from functools import partial

import pandas as pd
from loguru import logger
# wandb.login()

def train_rl_corrector_wandb(config = None, args=None):
    print("Starting Reinforcement Learning training...")
    if args.no_perturbation:
        perturbator = None
        print("Training without perturbation")
    else:
        perturbator = NormalJittering(0, args.perturbation_std)
        print(f"Training with normal jittering (std={args.perturbation_std})")

    u_sim = VITE_USim([0, 0])
    env = StableBaselinesGodotEnv(env_path="games/SmartDartSingleEnv/smartDartEnv.x86_64", show_window=False, n_parallel=1)

    with wandb.init(config=config):
        config = wandb.config
        corr = ReinforceCorrector(env, u_sim, perturbator, hidden_size=config.fc_layer_size, learning_rate=config.learning_rate, learn=True, log=True, policy_type="StackedMLP")
        reward_list, reward = corr.learn(False)
        wandb.log({"reward": reward})

                
        data = [[x, y] for (x, y) in zip(np.arange(len(reward_list)), reward_list)]
        table = wandb.Table(data=data, columns=["episodes", "rewards"])
        wandb.log(
            {
                "reward over episodes": wandb.plot.line(
                    table, "x", "y", title="rewards"
                )
            })
    print("Training completed using Reinforcement Learning method!")
    env.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train corrector using RL or CGP')
    parser.add_argument('--method', choices=['rl', 'cgp'], required=True,
                       help='Choose training method: rl (Reinforcement Learning) or cgp (Cartesian Genetic Programming)')
    parser.add_argument('--perturbation_std', type=float, default=20.0,
                       help='Standard deviation for normal jittering perturbation (default: 20.0)')
    parser.add_argument('--no_perturbation', action='store_true',
                       help='Disable perturbation during training')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    global args
    args = parser.parse_args()

    if args.wandb:
        # wandb.init(project="RL_corrector")
        sweep_config = {
            'method' : 'random',
        }

        metric = {
            'name': 'reward',
            'goal': 'maximize'
        }

        sweep_config['metric'] = metric
        if args.method == 'rl':
            parameters_dict = {
                'learning_rate': {
                    'distribution': 'uniform',
                    'min': 1e-5,
                    'max': 2e-3
                },
                'fc_layer_size': { 
                        'values': [128, 256, 512]
                                },
            }

            sweep_config['parameters'] = parameters_dict
            pprint.pprint(sweep_config)



            train = partial(train_rl_corrector_wandb, args = args)
            sweep_id = wandb.sweep(sweep_config, project="smartDart_RL_corrector")
            
            wandb.agent(sweep_id, train, count=20)
            
    else:
        

        # Create perturbation
        if args.no_perturbation:
            perturbator = None
            print("Training without perturbation")
        else:
            perturbator = NormalJittering(0, args.perturbation_std)
            print(f"Training with normal jittering (std={args.perturbation_std})")

        # Initialize the environment
        # env = GodotEnv(convert_action_space=True)
        if args.method == 'rl':
            n_parallel = 1
        elif args.method == 'cgp':
            n_parallel = 6
        else :
            raise ValueError(f"Unknown method: {args.method}")
        
        # env = StableBaselinesGodotEnv(env_path="games/SmartDartSingleEnv/smartDartEnv.x86_64", show_window=False, n_parallel=n_parallel)
        env = StableBaselinesGodotEnv(env_path="games/SmartDartSingleEnv/smartDartEnv.x86_64", show_window=False, n_parallel=n_parallel)

        # Initialize user simulator
        u_sim = VITE_USim([0, 0])
        Nruns = 10
        reward_runs = []
        reward_lists = []
        for i in range(Nruns):
            # Train based on selected method
            if args.method == 'rl':
                n_episodes=100
                logger.info(f"Reinforcement Learning training : {i+1} / {Nruns}")
                # corr = ReinforceCorrector(env, u_sim, perturbator, hidden_size=256, learning_rate=1e-4, learn=True, log=True, policy_type="StackedMLP")
                corr = DDPGCorrector(env, u_sim, perturbator, hidden_size=256, log=True, policy_type="StackedMLP")
                reward_list, final_reward = corr.learn(n_episodes)
                reward_runs.append(final_reward)
                reward_lists.append(reward_list)

                logger.info(f"final reward of iteration {i+1} : {final_reward}", )
                file_name = "10_rl_corrector_None_stacked_mlp_DDPG.csv"

                log_dict = {
                    "reward": reward_runs,
                    "reward_list": reward_lists}
                reward_df = pd.DataFrame(log_dict)
                reward_df.to_csv(file_name)

            elif args.method == 'cgp':
                ngen = 50
                print("Starting Cartesian Genetic Programming training...")
                corr = CGPCorrector(env, ngen, MAXSTEPS, 20, 1, perturbator)
                fit_history = corr.learn(8, 96, random_genomes=True)
            else:
                print(f"Unknown method: {args.method}")

            print(f"Training completed using {args.method.upper()} method!")

            # Close environment
        env.close()
