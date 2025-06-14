import argparse
from common.rolloutenv import *
from common.perturbation import *
from classic_rl.rl_corrector import *
from GA.cgp_corrector import *

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train corrector using RL or CGP')
    parser.add_argument('--method', choices=['rl', 'cgp'], required=True,
                       help='Choose training method: rl (Reinforcement Learning) or cgp (Cartesian Genetic Programming)')
    parser.add_argument('--perturbation_std', type=float, default=20.0,
                       help='Standard deviation for normal jittering perturbation (default: 20.0)')
    parser.add_argument('--no_perturbation', action='store_true',
                       help='Disable perturbation during training')

    args = parser.parse_args()

    # Create perturbation
    if args.no_perturbation:
        perturbator = None
        print("Training without perturbation")
    else:
        perturbator = NormalJittering(0, args.perturbation_std)
        print(f"Training with normal jittering (std={args.perturbation_std})")

    # Initialize the environment
    env = GodotEnv(convert_action_space=True)

    # Initialize user simulator
    u_sim = VITE_USim([0, 0])

    # Train based on selected method
    if args.method == 'rl':
        print("Starting Reinforcement Learning training...")
        corr = ReinforceCorrector(env, u_sim, perturbator, learn=True, log=True)
        corr.learn()
    elif args.method == 'cgp':
        ngen = 10   
        print("Starting Cartesian Genetic Programming training...")
        corr = CGPCorrector(env, ngen, MAXSTEPS, 20, 1, perturbator)
        corr.learn()
    else:
        print(f"Unknown method: {args.method}")

    print(f"Training completed using {args.method.upper()} method!")

    # Close environment
    env.close()
