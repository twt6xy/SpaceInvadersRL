import os

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from spaceinvadersrl.checkpoint import SaveOnBestTrainingRewardCallback
from spaceinvadersrl.cnn_lstm_policy import CnnLstmPolicy
from spaceinvadersrl.video import ModelVideoRecorder


class ModelTrainer:
    def __init__(
        self,
        model_type,
        policy_name,
        env_id,
        log_dir,
        video_dir,
        video_length,
        timesteps,
        check_freq,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.policy_name = policy_name

        if policy_name == "CnnLstm":
            self.policy = CnnLstmPolicy
        else:
            self.policy = policy_name

        self.env_id = env_id
        self.log_dir = log_dir
        self.video_dir = video_dir
        self.video_length = video_length
        self.timesteps = timesteps
        self.check_freq = check_freq
        self.model_kwargs = model_kwargs

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        self.env = self.setup_env()
        self.model = self.setup_model()

    def setup_env(self):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        env = Monitor(env, self.log_dir)
        env.metadata["render_fps"] = 30
        return env

    def setup_model(self):
        if self.model_type == "PPO":
            return PPO(self.policy, self.env, verbose=0, **self.model_kwargs)
        elif self.model_type == "DQN":
            return DQN(self.policy, self.env, verbose=0, **self.model_kwargs)
        else:
            raise ValueError("Unsupported model type")

    def train(self):
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=self.check_freq, log_dir=self.log_dir, verbose=0
        )
        self.model.learn(
            total_timesteps=int(self.timesteps), callback=callback, progress_bar=True
        )
        self.model.save(os.path.join(self.log_dir, "final_model"))

    def evaluate(self):
        model = self.model.load(
            os.path.join(self.log_dir, "final_model.zip"), env=self.env
        )
        mean_reward, std_reward = evaluate_policy(
            model, model.get_env(), n_eval_episodes=20
        )
        print(f"Mean Reward: {mean_reward}\nReward Standard Deviation: {std_reward}")

    def record(self):
        recorder = ModelVideoRecorder(
            self.model,
            video_folder=self.video_dir,
            video_length=self.video_length,
            env_id=self.env_id,
            model_type=self.model_type,
            policy_type=self.policy_name,
        )
        recorder.record_gif()

    def plot_results(self):
        plot_results(
            [self.log_dir],
            self.timesteps,
            results_plotter.X_TIMESTEPS,
            f"{self.model_type}-{self.policy_name}",
        )
        plt.show()
