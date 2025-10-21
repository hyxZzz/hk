"""Validation script for evaluating multi-agent DDQN checkpoints.

This module scans the ``models`` directory for checkpoint files named
``DDQN_agent{agent_id}_episode{episode}.pth`` (starting from episode 100),
groups them by episode, evaluates each complete set for a fixed number of
episodes, and logs the intercept success rate to ``runs/val`` using
TensorBoard summaries as well as a CSV file for convenience.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent, device as agent_device
from Environment.env import SOFT_CONSTRAINT_DEFAULT
from Environment.init_env import init_env


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration values used during validation."""

    episodes: int = 100
    num_planes: int = 2
    interceptor_num: int = 6
    num_missiles: int = 3
    step_num: int = 3500
    gamma: float = 0.993
    learning_rate: float = 5e-4
    soft_penalty_scale: float = SOFT_CONSTRAINT_DEFAULT


@dataclass(frozen=True)
class EvaluationSummary:
    mean_total_reward: Dict[int, float]
    success_rate: float
    mean_interceptor_launches: float
    illegal_launch_rate: float
    mean_overkill: float
    mean_cooperative_intercepts: float
    mean_unprotected_rate: float
    mean_survival_time: float
    survival_time_distribution: List[float]


CHECKPOINT_PATTERN = re.compile(r"DDQN_agent(\d+)_episode(\d+)\.pth$")


def collect_checkpoints(
    model_root: str,
    start_episode: int,
) -> List[Tuple[int, Dict[int, str]]]:
    """Return a sorted list of multi-agent checkpoint paths grouped by episode.

    Args:
        model_root: Root directory that potentially contains checkpoint files.
        start_episode: Minimum episode index (inclusive) that a checkpoint must
            have to be considered.

    Returns:
        A list of tuples ``(episode, {agent_id: path, ...})`` sorted by
        ``episode`` in ascending order. Only episodes that contain at least one
        checkpoint are returned. Missing agent checkpoints are filtered later
        during evaluation.
    """

    grouped_checkpoints: MutableMapping[int, Dict[int, str]] = {}
    if not os.path.isdir(model_root):
        return []

    for root, _, files in os.walk(model_root):
        for filename in files:
            match = CHECKPOINT_PATTERN.search(filename)
            if not match:
                continue
            agent_id = int(match.group(1))
            episode = int(match.group(2))
            if episode < start_episode:
                continue
            grouped_checkpoints.setdefault(episode, {})[agent_id] = os.path.join(
                root, filename
            )

    return sorted(grouped_checkpoints.items(), key=lambda item: item[0])


def build_agent(state_size: int, action_size: int, config: EvaluationConfig) -> MyDQNAgent:
    """Construct a ``MyDQNAgent`` ready for evaluation."""

    model = Double_DQN(state_size=state_size, action_size=action_size)
    agent = MyDQNAgent(
        model,
        action_size,
        gamma=config.gamma,
        lr=config.learning_rate,
        e_greed=0.0,
        e_greed_decrement=0.0,
    )
    agent.model.eval()
    agent.target_model.eval()
    return agent


def load_checkpoint(agent: MyDQNAgent, checkpoint_path: str) -> None:
    """Load model parameters from ``checkpoint_path`` into ``agent``."""

    load_kwargs = {"map_location": agent_device}
    try:
        # ``weights_only`` was added in PyTorch 2.1; fall back gracefully on
        # older versions that do not recognise the argument.
        load_kwargs["weights_only"] = True
        state = torch.load(checkpoint_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        state = torch.load(checkpoint_path, **load_kwargs)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    agent.model.load_state_dict(state)
    agent.target_model.load_state_dict(agent.model.state_dict())


def select_action(agent: MyDQNAgent, state, valid_actions=None) -> int:
    """Return the greedy action for ``state`` using ``agent``'s policy."""

    return agent.predict(state, valid_actions)


def _validate_checkpoint_mapping(
    checkpoint_paths: Mapping[int, str],
    expected_agents: Iterable[int],
) -> None:
    missing = [agent_id for agent_id in expected_agents if agent_id not in checkpoint_paths]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoints for agent IDs: {}".format(
                ", ".join(str(agent_id) for agent_id in missing)
            )
        )


def evaluate_checkpoint(
    checkpoint_paths: Mapping[int, str],
    config: EvaluationConfig,
) -> EvaluationSummary:
    """Evaluate ``checkpoint_paths`` and return aggregate metrics."""

    expected_agent_ids = range(config.num_planes)
    _validate_checkpoint_mapping(checkpoint_paths, expected_agent_ids)

    env, _, _ = init_env(
        num_missiles=config.num_missiles,
        StepNum=config.step_num,
        interceptor_num=config.interceptor_num,
        num_planes=config.num_planes,
        soft_penalty_scale=config.soft_penalty_scale,
    )
    action_size = env._get_actSpace()
    state_size = env._getNewStateSpace()[0]

    agents: Dict[int, MyDQNAgent] = {}
    for agent_id in expected_agent_ids:
        agent = build_agent(state_size, action_size, config)
        load_checkpoint(agent, checkpoint_paths[agent_id])
        agents[agent_id] = agent

    total_rewards = {agent_id: [] for agent_id in expected_agent_ids}
    success_count = 0
    launch_counts: List[int] = []
    illegal_launches = 0
    total_launch_attempts = 0
    intercept_lock_counts: List[int] = []
    cooperative_counts: List[int] = []
    unprotected_rates: List[float] = []
    survival_times: List[float] = []

    for _ in range(config.episodes):
        state, done_flag, _ = env.reset()
        episode_reward = {agent_id: 0.0 for agent_id in expected_agent_ids}
        episode_info = None

        while True:
            state_masks = {agent_id: env.valid_action_mask(agent_id) for agent_id in expected_agent_ids}
            actions = {
                agent_id: select_action(agent, state[agent_id], state_masks[agent_id])
                for agent_id, agent in agents.items()
            }
            state, reward, done_flag, info = env.step(actions)
            episode_info = info
            for agent_id in expected_agent_ids:
                episode_reward[agent_id] += reward.get(agent_id, 0.0)
            if done_flag != -1:
                if done_flag == 2:
                    success_count += 1
                break

        metrics = None
        if episode_info is not None:
            metrics = episode_info.get("metrics")
        if metrics is None:
            metrics = env.collect_episode_metrics()

        launch_counts.append(int(metrics.get("successful_launches", 0)))
        total_launch_attempts += int(metrics.get("launch_attempts", 0))
        illegal_launches += int(metrics.get("invalid_launches", 0))
        intercept_lock_counts.extend(int(value) for value in metrics.get("intercept_lock_counts", []))
        cooperative_counts.append(int(metrics.get("cooperative_intercepts", 0)))
        episode_length = max(1, int(metrics.get("episode_length", 1)))
        unprotected_rates.append(
            float(metrics.get("unprotected_threat_steps", 0)) / float(episode_length)
        )
        survival_times.extend(float(value) for value in metrics.get("plane_survival_time", []))

        for agent_id in expected_agent_ids:
            total_rewards[agent_id].append(episode_reward[agent_id])

    success_rate = success_count / float(config.episodes) if config.episodes > 0 else 0.0
    mean_total_reward = {
        agent_id: float(np.mean(rewards)) if rewards else 0.0
        for agent_id, rewards in total_rewards.items()
    }
    mean_launches = float(np.mean(launch_counts)) if launch_counts else 0.0
    illegal_rate = (
        illegal_launches / float(total_launch_attempts)
        if total_launch_attempts > 0
        else 0.0
    )
    mean_overkill = float(np.mean(intercept_lock_counts)) if intercept_lock_counts else 0.0
    mean_cooperative = float(np.mean(cooperative_counts)) if cooperative_counts else 0.0
    mean_unprotected = float(np.mean(unprotected_rates)) if unprotected_rates else 0.0
    survival_time_distribution = [time for time in survival_times if time >= 0.0]
    mean_survival_time = (
        float(np.mean(survival_time_distribution)) if survival_time_distribution else 0.0
    )

    return EvaluationSummary(
        mean_total_reward=mean_total_reward,
        success_rate=success_rate,
        mean_interceptor_launches=mean_launches,
        illegal_launch_rate=illegal_rate,
        mean_overkill=mean_overkill,
        mean_cooperative_intercepts=mean_cooperative,
        mean_unprotected_rate=mean_unprotected,
        mean_survival_time=mean_survival_time,
        survival_time_distribution=survival_time_distribution,
    )


def create_writer() -> Tuple[SummaryWriter, str]:
    """Create a ``SummaryWriter`` under ``runs/val`` and return it and its path."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "val", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def save_csv(
    log_dir: str, results: Sequence[Tuple[int, Mapping[int, str], EvaluationSummary]]
) -> str:
    """Persist evaluation results as a CSV file and return its path."""

    csv_path = os.path.join(log_dir, "intercept_success_rates.csv")
    header = (
        "episode,checkpoints,success_rate,mean_interceptor_launches,illegal_launch_rate," \
        "mean_overkill,mean_cooperative_intercepts,mean_unprotected_rate,mean_survival_time\n"
    )
    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write(header)
        for episode, checkpoints, summary in results:
            checkpoint_repr = ";".join(
                f"agent{agent_id}:{path}" for agent_id, path in sorted(checkpoints.items())
            )
            csv_file.write(
                f"{episode},{checkpoint_repr},{summary.success_rate:.6f},"
                f"{summary.mean_interceptor_launches:.6f},{summary.illegal_launch_rate:.6f},"
                f"{summary.mean_overkill:.6f},{summary.mean_cooperative_intercepts:.6f},"
                f"{summary.mean_unprotected_rate:.6f},{summary.mean_survival_time:.6f}\n"
            )
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DDQN checkpoints")
    parser.add_argument(
        "--model-root",
        default="models",
        help="Root directory that stores DDQN checkpoints (default: models)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EvaluationConfig.episodes,
        help="Number of evaluation episodes per checkpoint (default: 100)",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=100,
        help="Minimum checkpoint episode index to evaluate (default: 100)",
    )
    parser.add_argument(
        "--num-missiles",
        type=int,
        default=EvaluationConfig.num_missiles,
        help="Number of incoming missiles used to initialise the environment",
    )
    parser.add_argument(
        "--num-planes",
        type=int,
        default=EvaluationConfig.num_planes,
        help="Number of cooperative agents expected per checkpoint",
    )
    parser.add_argument(
        "--interceptor-num",
        type=int,
        default=EvaluationConfig.interceptor_num,
        help="Number of interceptors per plane when initialising the environment",
    )
    parser.add_argument(
        "--step-num",
        type=int,
        default=EvaluationConfig.step_num,
        help="Maximum number of steps per episode for the environment",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=EvaluationConfig.gamma,
        help="Discount factor used by the agent (default: 0.99)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=EvaluationConfig.learning_rate,
        help="Learning rate placeholder required by the agent (default: 0.001)",
    )
    parser.add_argument(
        "--soft-penalty-scale",
        type=float,
        default=EvaluationConfig.soft_penalty_scale,
        help="Soft penalty applied when violating launch constraints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        episodes=args.episodes,
        num_missiles=args.num_missiles,
        num_planes=args.num_planes,
        interceptor_num=args.interceptor_num,
        step_num=args.step_num,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        soft_penalty_scale=args.soft_penalty_scale,
    )

    checkpoints = collect_checkpoints(args.model_root, args.start_episode)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching 'DDQN_agent*_episode*.pth' (>= episode {args.start_episode}) "
            f"were found under '{args.model_root}'."
        )

    writer, log_dir = create_writer()
    results: List[Tuple[int, Mapping[int, str], EvaluationSummary]] = []

    for episode, checkpoint_paths in checkpoints:
        try:
            summary = evaluate_checkpoint(checkpoint_paths, config)
        except FileNotFoundError as exc:
            print(
                f"Skipping episode {episode} due to incomplete checkpoints: {exc}"
            )
            continue
        writer.add_scalar("intercept_success_rate", summary.success_rate, global_step=episode)
        writer.add_scalar(
            "val/mean_interceptor_launches", summary.mean_interceptor_launches, global_step=episode
        )
        writer.add_scalar("val/illegal_launch_rate", summary.illegal_launch_rate, global_step=episode)
        writer.add_scalar("val/mean_overkill", summary.mean_overkill, global_step=episode)
        writer.add_scalar(
            "val/mean_cooperative_intercepts", summary.mean_cooperative_intercepts, global_step=episode
        )
        writer.add_scalar("val/mean_unprotected_rate", summary.mean_unprotected_rate, global_step=episode)
        writer.add_scalar("val/mean_survival_time", summary.mean_survival_time, global_step=episode)
        results.append((episode, dict(checkpoint_paths), summary))
        formatted_paths = ", ".join(
            f"agent{agent_id}:{path}" for agent_id, path in sorted(checkpoint_paths.items())
        )
        print(
            f"Episode {episode:>4} | success rate: {summary.success_rate:.4f} | {formatted_paths}"
        )

    writer.close()
    csv_path = save_csv(log_dir, results)
    print(f"Validation complete. Results saved to {csv_path}")


if __name__ == "__main__":
    main()

