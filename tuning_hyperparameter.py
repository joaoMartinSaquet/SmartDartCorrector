import argparse
from common.rolloutenv import *
from common.perturbation import *
from classic_rl.rl_corrector import *
from GA.cgp_corrector import *

import optuna
from optuna.trial import TrialState
from plotly.io import show
import matplotlib.pyplot as plt


n_jobs = 1  
env = StableBaselinesGodotEnv(env_path="games/SmartDartPlusDist/smartDartEnv.x86_64", show_window=False, n_parallel=n_jobs)
envs = deque(env.envs, maxlen=env.num_envs) 
def objective(trial):
    # Define hyperparameters
    learning_rate_actor = trial.suggest_float("learning_rate_actor", 1e-5, 1e-3, log=True)
    learning_rate_critic = trial.suggest_float("learning_rate_critic", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    # update_interval = trial.suggest_int("update_interval", 1, 1000)
    
    num_episodes = 10
    perturbator = None

    env = envs.pop()
    env.reset()
    # env = StableBaselinesGodotEnv(env_path="games/SmartDartEnvNormalized/smartDartEnv.x86_64", show_window=False, n_parallel=n_jobs)
    u_sim = VITE_USim([0, 0])
    corr = DDPGCorrector(env, u_sim, perturbator, hidden_size=hidden_size, 
                         actor_lr=learning_rate_actor, critic_lr=learning_rate_critic, batch_size=batch_size, policy_type="MLP")
    
    
    reward_list, final_reward = corr.training_loop(num_episodes)
    reward = final_reward
    
    envs.append(env)
    return reward


def objectivePPO(trial):
    # Define hyperparameters
    lr_actor = trial.suggest_float("lr_actor", 1e-6, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-6, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    noptimsteps = trial.suggest_int("noptimsteps", 10, 100)
    action_std = trial.suggest_float("action_std", 0.1, 0.5)
    decay = trial.suggest_float("decay", 0.01, 0.1)
    num_episodes = 10
    perturbator = None

    env = StableBaselinesGodotEnv(env_path="games/SmartDartPlusDist/smartDartEnv.x86_64", show_window=False, n_parallel=n_jobs)
    u_sim = VITE_USim([0, 0])

    # Initialize the corrector with the suggested hyperparameters
    corr = PPOCorrector(
        env=env,
        u_sim=u_sim,
        perturbator=perturbator,
        hidden_size=hidden_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        decay_action_std=decay,
        action_std_init=action_std,
        gamma=gamma,
        noptimsteps=noptimsteps,
        clip_epsilon=clip_epsilon,
        gae_lambda=gae_lambda,
        num_episodes=num_episodes,
        policy_type="MLP"
    )

    # Run the training loop and get the final reward
    reward_list, reward = corr.train(num_episodes)
    # Return the final reward as the objective value
    return reward

logger.info("Start study")
study = optuna.create_study(direction="maximize")
study.optimize(objectivePPO, n_trials=5, n_jobs=n_jobs, show_progress_bar=True)
    
    
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

logger.info("Study statistics: ")
logger.info("  Number of finished trials: ", len(study.trials))
logger.info("  Number of pruned trials: ", len(pruned_trials))
logger.info("  Number of complete trials: ", len(complete_trials))

logger.info("Best trial:")
trial = study.best_trial

logger.info("  Value:  {trial.value}")

logger.info("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

    
fig = optuna.visualization.plot_parallel_coordinate(study)
# plt.savefig("plots/parallel_coordinate.png", dpi=300)
fig.write_html("parralel_plot.html")

fig = optuna.visualization.plot_optimization_history(study)
fig.write_html("otpim_history.html")