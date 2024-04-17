import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CnnLstmPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        *args,
        **kwargs
    ):
        super(CnnLstmPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )

        in_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros(
            1, in_channels, observation_space.shape[1], observation_space.shape[2]
        )
        dummy_output = self.cnn(dummy_input)
        n_flatten = dummy_output.view(1, -1).size(1)

        self.lstm = nn.LSTM(n_flatten, 512)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, action_space.n)

    def forward(self, obs, deterministic=False, device=None, **kwargs):
        obs = obs.to(torch.float32) / 255.0
        features = self.cnn(obs)
        features = features.view(1, -1, features.size(1))

        lstm_output, (hidden_state, cell_state) = self.lstm(features)
        lstm_output = lstm_output[-1]

        value = self.critic_linear(lstm_output)
        logits = self.actor_linear(lstm_output)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = dist.probs.argmax(dim=1)
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action)

        return action, action_log_probs, value
