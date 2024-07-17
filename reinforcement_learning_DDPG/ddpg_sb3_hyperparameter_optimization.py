import gymnasium as gym 
from stable_baselines3 import PPO, DDPG, HerReplayBuffer, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import torch
import torch.nn as nn
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict
import torch
import torch.nn as nn
from argparse import ArgumentParser


N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 2  # Number of evaluations during the training
N_TIMESTEPS = int(2e4)  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 15)  # 15 minutes
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
ENV_ID = "FetchReach-v2"
ENV_NAME = "FetchReach-v2"

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "env": ENV_ID,
}

def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    tau = trial.suggest_float("tau", 0, 1, log=False)
    batch_size = trial.suggest_int("batch_size", 64,256,log=False )
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000, log=True)

    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["tiny","small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])


    # Display true values
    trial.set_user_attr("gamma_", gamma)
    # trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64,64], "vf": [64,64]}
        if net_arch == "tiny"
        else {"pi": [400, 300], "vf": [400, 300]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    # goal_selection_strategy = {"future": "future", "final": "final", "episode": "episode"}[goal_selection_strategy]
    return {
        # "n_steps": n_steps,
        "gamma": gamma,
        "tau":tau,
        "batch_size":batch_size,
        "buffer_size":buffer_size,
        "learning_rate": learning_rate,
        # "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            # "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
    }

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    
    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = False,
        verbose: int = 0,
        best_model_save_path: str = "/teamspace/studios/this_studio/A2C_Mode", #Path to save the both model

    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path = best_model_save_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """
    params = sample_ddpg_params(trial)
    
    env = gym.make(ENV_NAME)
    env = DummyVecEnv([lambda: env])

    eval_envs = make_vec_env(ENV_ID, n_envs=N_EVAL_ENVS)

    # kwargs = DEFAULT_HYPERPARAMS.copy()
    ### YOUR CODE HERE
    #: 
    # 1. Sample hyperparameters and update the default keyword arguments: `kwargs.update(other_params)`
    # 2. Create the evaluation envs
    # 3. Create the `TrialEvalCallback`

    # 1. Sample hyperparameters and update the keyword arguments
    # print(sample_ddpg_params(trial))
    #kwargs.update(sample_a2c_params(trial))
    # kwargs.update(sample_ddpg_params(trial))

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Create the RL model
    #model = A2C(**kwargs)
    model = DDPG(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=0,
        **params,
        learning_starts=100,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        ),
    )

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    # eval_envs = make_vec_env(ENV_ID, n_envs=N_EVAL_ENVS)
    # eval_envs = HERGoalEnvWrapper(eval_envs)

    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    eval_callback = TrialEvalCallback(eval_envs, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ)

    ### END OF YOUR CODE

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float(0) #replace with "nan" if error

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--path', type=str, default=os.getcwd()+"hp_list.csv")
    args = parser.parse_args()

    torch.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    print(args.n_trials)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=N_JOBS) # timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv(args.path)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()

