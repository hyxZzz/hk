"""Utilities for summarising reward and action statistics from logged rollouts."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


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


@dataclass
class EvaluationHistory:
    episodes: List[int]
    success_rates: List[float]
    mean_interceptor_launches: List[float]
    illegal_launch_rates: List[float]
    mean_overkills: List[float]
    mean_cooperative_intercepts: List[float]
    mean_unprotected_rates: List[float]
    mean_survival_times: List[float]


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


def load_evaluation_history(csv_path: Path) -> EvaluationHistory:
    episodes: List[int] = []
    success_rates: List[float] = []
    mean_launches: List[float] = []
    illegal_rates: List[float] = []
    mean_overkills: List[float] = []
    mean_cooperative: List[float] = []
    mean_unprotected: List[float] = []
    mean_survival: List[float] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            episodes.append(int(row["episode"]))
            success_rates.append(float(row["success_rate"]))
            mean_launches.append(float(row.get("mean_interceptor_launches", 0.0)))
            illegal_rates.append(float(row.get("illegal_launch_rate", 0.0)))
            mean_overkills.append(float(row.get("mean_overkill", 0.0)))
            mean_cooperative.append(float(row.get("mean_cooperative_intercepts", 0.0)))
            mean_unprotected.append(float(row.get("mean_unprotected_rate", 0.0)))
            mean_survival.append(float(row.get("mean_survival_time", 0.0)))

    return EvaluationHistory(
        episodes=episodes,
        success_rates=success_rates,
        mean_interceptor_launches=mean_launches,
        illegal_launch_rates=illegal_rates,
        mean_overkills=mean_overkills,
        mean_cooperative_intercepts=mean_cooperative,
        mean_unprotected_rates=mean_unprotected,
        mean_survival_times=mean_survival,
    )


def plot_evaluation_history(history: EvaluationHistory, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot(series: List[float], ylabel: str, filename: str) -> None:
        plt.figure()
        plt.plot(history.episodes, series, marker="o")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    _plot(history.success_rates, "Success Rate", "success_rate.png")
    _plot(
        history.mean_interceptor_launches,
        "Mean Interceptor Launches",
        "mean_interceptor_launches.png",
    )
    _plot(history.illegal_launch_rates, "Illegal Launch Rate", "illegal_launch_rate.png")
    _plot(history.mean_overkills, "Mean Overkill", "mean_overkill.png")
    _plot(
        history.mean_cooperative_intercepts,
        "Mean Cooperative Intercepts",
        "mean_cooperative_intercepts.png",
    )
    _plot(history.mean_unprotected_rates, "Mean Unprotected Rate", "mean_unprotected_rate.png")
    _plot(history.mean_survival_times, "Mean Survival Time", "mean_survival_time.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse reward logs and optional evaluation metrics"
    )
    parser.add_argument(
        "--action-log",
        type=Path,
        default=Path("action.txt"),
        help="Path to the action log file (default: action.txt)",
    )
    parser.add_argument(
        "--reward-log",
        type=Path,
        default=Path("reward.txt"),
        help="Path to the reward log file (default: reward.txt)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV file containing evaluation metrics (from utils.validate)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics"),
        help="Directory where evaluation plots will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = load_stats(args.action_log, args.reward_log)

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

    if args.metrics_csv is not None and args.metrics_csv.exists():
        history = load_evaluation_history(args.metrics_csv)
        plot_evaluation_history(history, args.output_dir)
        print(f"Evaluation plots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
