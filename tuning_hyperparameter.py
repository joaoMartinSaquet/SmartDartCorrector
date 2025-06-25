import argparse
from common.rolloutenv import *
from common.perturbation import *
from classic_rl.rl_corrector import *
from GA.cgp_corrector import *

import optuna
from optuna.trial import TrialState
from plotly.io import show
import matplotlib.pyplot as plt


n_jobs = 10
env = StableBaselinesGodotEnv(env_path="games/SmartDartEnvNormalized/smartDartEnv.x86_64", show_window=False, n_parallel=n_jobs)
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

logger.info("Start study")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=n_jobs, show_progress_bar=True)
    
    
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