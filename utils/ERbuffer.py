"""Prioritized replay buffer with optional n-step returns."""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import numpy as np


class PrioritizedReplayBuffer:
    """Prioritized experience replay with support for n-step returns."""

    def __init__(
        self,
        capacity: int,
        gamma: float,
        n_step: int = 3,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.n_step = max(1, int(n_step))
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.beta_increment = (1.0 - beta_start) / max(1, beta_frames)
        self.epsilon = float(epsilon)

        self.buffer: Deque[Dict[str, object]] = deque(maxlen=self.capacity)
        self.priorities: Deque[float] = deque(maxlen=self.capacity)
        self.n_step_buffer: Deque[Dict[str, object]] = deque(maxlen=self.n_step)

    def clear(self) -> None:
        self.buffer.clear()
        self.priorities.clear()
        self.n_step_buffer.clear()

    def size(self) -> int:
        return len(self.buffer)

    def can_sample(self, batch_size: int) -> bool:
        return self.size() >= batch_size

    def add(
        self,
        state,
        actions,
        reward,
        next_state,
        done,
        state_masks,
        next_masks,
    ) -> None:
        transition = {
            "state": copy.deepcopy(state),
            "actions": copy.deepcopy(actions),
            "reward": dict(reward),
            "next_state": copy.deepcopy(next_state),
            "done": done,
            "state_masks": copy.deepcopy(state_masks),
            "next_masks": copy.deepcopy(next_masks),
        }
        self.n_step_buffer.append(transition)
        self._process_n_step_buffer(force=done != -1)

        if done != -1:
            while self.n_step_buffer:
                self._process_n_step_buffer(force=True)

    def on_episode_end(self) -> None:
        while self.n_step_buffer:
            self._process_n_step_buffer(force=True)

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray]:
        batch_size = min(batch_size, self.size())
        if batch_size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        priorities = np.array(self.priorities, dtype=np.float32)
        if priorities.size == 0:
            raise ValueError("Replay buffer priorities are empty")

        scaled_priorities = priorities ** self.alpha
        if scaled_priorities.sum() == 0:
            scaled_priorities.fill(1.0)
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        self.beta = min(1.0, self.beta + self.beta_increment)

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max() if weights.max() > 0 else 1.0

        experiences = [self.buffer[idx] for idx in indices]
        return experiences, indices, weights.astype(np.float32)

    def update_priorities(self, indices: Iterable[int], priorities: Sequence[float]) -> None:
        for idx, priority in zip(indices, priorities):
            updated_priority = float(abs(priority) + self.epsilon)
            if idx < 0 or idx >= len(self.priorities):
                continue
            self.priorities[idx] = updated_priority

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _process_n_step_buffer(self, force: bool = False) -> None:
        if not self.n_step_buffer:
            return
        if not force and len(self.n_step_buffer) < self.n_step:
            last_done = self.n_step_buffer[-1]["done"]
            if last_done == -1:
                return

        reward_sum, next_state, done_flag, steps, next_masks = self._get_n_step_info()
        first_transition = self.n_step_buffer[0]
        experience = {
            "state": copy.deepcopy(first_transition["state"]),
            "actions": copy.deepcopy(first_transition["actions"]),
            "reward": reward_sum,
            "next_state": copy.deepcopy(next_state),
            "done": done_flag,
            "state_masks": copy.deepcopy(first_transition["state_masks"]),
            "next_masks": copy.deepcopy(next_masks),
            "n_steps": steps,
        }

        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(experience)
        self.priorities.append(max_priority)
        self.n_step_buffer.popleft()

    def _get_n_step_info(self) -> Tuple[Dict[int, float], object, int, int, object]:
        reward_sum: Dict[int, float] = defaultdict(float)
        discount = 1.0
        next_state = None
        next_masks = None
        done_flag = -1
        steps = 0

        for transition in self.n_step_buffer:
            steps += 1
            reward_dict = transition["reward"]
            for agent_id, value in reward_dict.items():
                reward_sum[int(agent_id)] += discount * float(value)

            next_state = transition["next_state"]
            next_masks = transition["next_masks"]
            done_flag = transition["done"]

            if done_flag != -1 or steps >= self.n_step:
                break
            discount *= self.gamma

        return dict(reward_sum), next_state, done_flag, steps, next_masks
