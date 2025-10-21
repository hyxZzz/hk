from typing import Optional, Sequence

import torch
import torch.nn as nn


class Double_DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Optional[Sequence[int]] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (256, 256)

        layers = []
        in_features = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        value_hidden = max(128, in_features // 2)
        self.value_stream = nn.Sequential(
            nn.Linear(in_features, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(in_features, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values
