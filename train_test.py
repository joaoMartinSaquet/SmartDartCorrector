
from common.rolloutenv import *
from common.perturbation import *
from common.user_simulator import *
from classic_rl.rl_corrector import *
import pandas as pd


TUNE_HYPERPARAMETERS = False

if TUNE_HYPERPARAMETERS:
    import optuna
    from optuna.trial import TrialState
    def obejectivePPO(trial):
        lr_actor = trial.suggest_float("lr_actor", 1e-6, 1e-3, log=True)
        lr_critic = trial.suggest_float("lr_critic", 1e-6, 1e-3, log=True)
        # hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        gamma = trial.suggest_float("gamma", 0.95, 0.9999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        K_epochs = trial.suggest_int("K_epochs", 10, 100)
        action_std = trial.suggest_float("action_std", 0.1, 0.5)
        decay = trial.suggest_float("decay", 0.01, 0.1)
        min_action_std = trial.suggest_float("min_action_std", 0.0  , .1)
        update_factor = trial.suggest_int("update_factor", 1, 10)
        perturbator = None
        u_sim = VITE_USim([0, 0])
        env = smartDartEnv(u_sim, perturbator, render = False, n_parallel=1)

        corr = PPOCorrector(env, u_sim, perturbator,
                             lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, clip_epsilon=clip_epsilon, k_epochs=K_epochs, action_std_init=action_std,
                             decay_action_std=decay, min_action_std=min_action_std, update_factor=update_factor)
        log = corr.train_classic_env()
        df = pd.DataFrame(log)
        mean_ep_reward = df["reward"][-5:].mean()
        return mean_ep_reward

if __name__ == "__main__":


    if TUNE_HYPERPARAMETERS:
        study = optuna.create_study(direction="maximize")
        study.optimize(obejectivePPO, n_trials=5)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info(f"  Number of finished trials: {len(study.trials)}")
        logger.info(f"  Number of pruned trials: {len(pruned_trials)}", )
        logger.info(f"  Number of complete trials: {len(complete_trials)}" )

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info(f"  Value:  {trial.value}")

        logger.info(f"  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        u_sim = VITE_USim([0, 0])
        # perturbator = NormalJittering(0, 20)
        perturbator = None

        env = smartDartEnv(u_sim, perturbator, render = False, n_parallel=1)
        # working one (need to be tuned)
        # corrector = PPOCorrector(env, u_sim, perturbator, hidden_size=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, k_epochs=80, clip_epsilon=0.2, gae_lambda=1, action_std_init = 0.6, decay_action_std = 0.05, min_action_std = 0.01, max_training_timesteps = 3e6, max_ep_len = 1000, update_factor = 4)
        corrector = PPOCorrector(env, u_sim, perturbator, hidden_size=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, k_epochs=80, clip_epsilon=0.2, gae_lambda=1, action_std_init = 0.6, decay_action_std = 0.05, min_action_std = 0.01, max_training_timesteps = 2.5e6, max_ep_len = 1000, update_factor = 4)
        log = corrector.train()


        pd.DataFrame(log).to_csv(os.path.join(corrector.log_dir, "log.csv"))
    



    