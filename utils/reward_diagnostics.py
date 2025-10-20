"""Utilities for summarising reward and action statistics from logged rollouts."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class RewardActionStats:
    total_steps: int
    reward_min: float
    reward_max: float
    reward_mean: float
    quantiles: Dict[int, float]
    action_counts: Counter
    action_positive_ratio: Dict[int, float]
    action_mean_reward: Dict[int, float]

    def top_actions_by_positive_ratio(self, limit: int = 10) -> List[Tuple[int, float, float, int]]:
        entries: List[Tuple[int, float, float, int]] = []
        for action, ratio in self.action_positive_ratio.items():
            entries.append((action, ratio, self.action_mean_reward[action], self.action_counts[action]))
        entries.sort(key=lambda item: item[1], reverse=True)
        return entries[:limit]

    def bottom_actions_by_mean_reward(self, limit: int = 10) -> List[Tuple[int, float, float, int]]:
        entries: List[Tuple[int, float, float, int]] = []
        for action, mean_reward in self.action_mean_reward.items():
            entries.append((action, mean_reward, self.action_positive_ratio[action], self.action_counts[action]))
        entries.sort(key=lambda item: item[1])
        return entries[:limit]


def _parse_values(path: Path, prefix: str) -> List[float]:
    values: List[float] = []
    needle = f"{prefix}"
    for line in path.read_text().splitlines():
        if line.startswith(needle):
            try:
                value = float(line.split(":")[-1])
            except ValueError:
                continue
            values.append(value)
    return values


def _parse_actions(path: Path) -> List[int]:
    actions: List[int] = []
    needle = "action for t="
    for line in path.read_text().splitlines():
        if line.startswith(needle):
            try:
                value = int(line.split(":")[-1])
            except ValueError:
                continue
            actions.append(value)
    return actions


def _quantiles(values: Sequence[float], percentiles: Iterable[int]) -> Dict[int, float]:
    if not values:
        return {p: float("nan") for p in percentiles}
    sorted_values = sorted(values)
    size = len(sorted_values)
    results: Dict[int, float] = {}
    for percentile in percentiles:
        if percentile <= 0:
            results[percentile] = sorted_values[0]
            continue
        if percentile >= 100:
            results[percentile] = sorted_values[-1]
            continue
        index = int((percentile / 100) * (size - 1))
        results[percentile] = sorted_values[index]
    return results


def load_stats(action_log: Path, reward_log: Path) -> RewardActionStats:
    actions = _parse_actions(action_log)
    rewards = _parse_values(reward_log, "reward for t=")
    if len(actions) != len(rewards):
        raise ValueError(
            f"Mismatched log lengths: {len(actions)} actions versus {len(rewards)} rewards"
        )

    total_steps = len(actions)
    reward_min = min(rewards)
    reward_max = max(rewards)
    reward_mean = sum(rewards) / total_steps
    quantiles = _quantiles(rewards, percentiles=[5, 25, 50, 75, 95])

    action_counts: Counter = Counter(actions)
    action_positive_ratio: Dict[int, float] = {}
    action_mean_reward: Dict[int, float] = {}

    reward_sums: Dict[int, float] = defaultdict(float)
    reward_positive_counts: Dict[int, int] = defaultdict(int)

    for action, reward in zip(actions, rewards):
        reward_sums[action] += reward
        if reward > 0:
            reward_positive_counts[action] += 1

    for action, count in action_counts.items():
        action_positive_ratio[action] = reward_positive_counts[action] / count
        action_mean_reward[action] = reward_sums[action] / count

    return RewardActionStats(
        total_steps=total_steps,
        reward_min=reward_min,
        reward_max=reward_max,
        reward_mean=reward_mean,
        quantiles=quantiles,
        action_counts=action_counts,
        action_positive_ratio=action_positive_ratio,
        action_mean_reward=action_mean_reward,
    )


def main() -> None:
    base_dir = Path.cwd()
    action_log = base_dir / "action.txt"
    reward_log = base_dir / "reward.txt"
    stats = load_stats(action_log, reward_log)

    print(f"Total transitions: {stats.total_steps}")
    print(
        f"Reward range: [{stats.reward_min:.3f}, {stats.reward_max:.3f}], "
        f"mean={stats.reward_mean:.3f}"
    )
    print("Selected quantiles:")
    for percentile, value in stats.quantiles.items():
        print(f"  {percentile}% -> {value:.3f}")

    most_common = stats.action_counts.most_common(10)
    print("Most frequent actions:")
    for action, count in most_common:
        ratio = stats.action_positive_ratio[action]
        mean_reward = stats.action_mean_reward[action]
        print(
            f"  action {action:>2}: count={count:>6}, positive_ratio={ratio:>5.3f}, "
            f"mean_reward={mean_reward:>7.3f}"
        )

    print("\nActions with the worst mean reward:")
    for action, mean_reward, ratio, count in stats.bottom_actions_by_mean_reward():
        print(
            f"  action {action:>2}: count={count:>6}, positive_ratio={ratio:>5.3f}, "
            f"mean_reward={mean_reward:>7.3f}"
        )


if __name__ == "__main__":
    main()
