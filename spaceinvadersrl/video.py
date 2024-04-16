import os

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


class ModelVideoRecorder:
    def __init__(self, model, video_folder, video_length, env_id, model_type):
        self.model = model
        self.video_folder = video_folder
        self.video_length = video_length
        self.model_type = model_type
        self.env_id = env_id
        os.makedirs(self.video_folder, exist_ok=True)
        self.vec_env = None

    def setup_env(self):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = Monitor(env, self.video_folder)
        self.vec_env = DummyVecEnv([lambda: env])

        self.vec_env = VecVideoRecorder(
            self.vec_env,
            self.video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=self.video_length,
            name_prefix=f"{self.model_type}-{self.env_id}",
        )

    def record_video(self):
        if self.vec_env is None:
            self.setup_env()

        obs = self.vec_env.reset()
        for _ in range(self.video_length + 1):
            action = [self.vec_env.action_space.sample()]
            obs, _, _, _ = self.vec_env.step(action)

    def close(self):
        if self.vec_env is not None:
            self.vec_env.close()
            self.vec_env = None
