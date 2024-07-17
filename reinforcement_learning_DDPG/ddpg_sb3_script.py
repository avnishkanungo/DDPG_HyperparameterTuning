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
from argparse import ArgumentParser


def ddpg(timesteps):
    environment_name = "FetchReach-v2"
    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    goal_selection_strategy = "future"
    model = DDPG('MultiInputPolicy', 
                env, 
                action_noise=action_noise,
                replay_buffer_class = HerReplayBuffer, 
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=goal_selection_strategy,
                ), 
                verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=20, save_path='/teamspace/studios/this_studio/optimized_logs/', name_prefix='ddpg_model')
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback) #learn 
    ddpg_path = '/teamspace/studios/this_studio/FetchReach_model_ddpg_test'
    model.save(ddpg_path)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()

    ddpg(args.timesteps)