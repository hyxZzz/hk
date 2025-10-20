"""Evaluate DDQN checkpoints on an identical batch of environments.

This script sequentially loads DDQN checkpoints stored in the training
``models`` directory, reuses a fixed batch of randomly generated evaluation
scenarios, and reports the intercept success rate for each checkpoint.  The
results are printed to stdout and persisted to a CSV file.

Example usage::

    python -m utils.validate_shared_env \
        --model-root models/DQNmodels/DDQNmodels3_23 \
        --episodes 100

"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from Environment.init_env import init_env
from utils.validate import (
    EvaluationConfig,
    build_agent,
    collect_checkpoints,
    load_checkpoint,
    select_action,
)


@dataclass(frozen=True)
class ScenarioSeeds:
    """Seeds required to reproduce evaluation scenarios."""

    env_seed: int
    episode_seeds: Sequence[int]


def generate_scenario_seeds(
    num_episodes: int,
    base_seed: int | None = None,
) -> ScenarioSeeds:
    """Return deterministic seeds for the environment and each episode."""

    rng = np.random.default_rng(base_seed)
    env_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    episode_seeds = [int(seed) for seed in rng.integers(0, np.iinfo(np.uint32).max, size=num_episodes)]
    return ScenarioSeeds(env_seed=env_seed, episode_seeds=episode_seeds)


def evaluate_on_shared_env(
    checkpoint_path: str,
    config: EvaluationConfig,
    seeds: ScenarioSeeds,
) -> float:
    """Evaluate ``checkpoint_path`` on a shared batch of scenarios."""

    original_state = np.random.get_state()
    try:
        np.random.seed(seeds.env_seed)
        env, _, _ = init_env(
            num_missiles=config.num_missiles,
            StepNum=config.step_num,
        )

        action_size = env._get_actSpace()
        state_size = env._getNewStateSpace()[0]

        agent = build_agent(state_size, action_size, config)
        load_checkpoint(agent, checkpoint_path)

        successes = 0
        for seed in seeds.episode_seeds:
            np.random.seed(seed)
            state, done_flag, _ = env.reset()
            while True:
                action = select_action(agent, state)
                state, _, done_flag, _ = env.step(action)
                if done_flag != -1:
                    if done_flag == 2:
                        successes += 1
                    break

        return successes / float(len(seeds.episode_seeds))
    finally:
        np.random.set_state(original_state)


def write_results_csv(path: str, results: Iterable[Tuple[int, str, float]]) -> None:
    """Persist evaluation ``results`` to ``path`` in CSV format."""

    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "checkpoint", "success_rate"])
        for episode, checkpoint, success_rate in results:
            writer.writerow([episode, checkpoint, f"{success_rate:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate DDQN checkpoints using an identical batch of evaluation "
            "environments"
        )
    )
    parser.add_argument(
        "--model-root",
        default="models/DQNmodels/DDQNmodels3_23",
        help="Directory that stores DDQN checkpoints (default: models/DQNmodels/DDQNmodels3_23)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EvaluationConfig.episodes,
        help="Number of evaluation episodes to run per checkpoint (default: 100)",
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
        help="Number of incoming missiles used during evaluation",
    )
    parser.add_argument(
        "--step-num",
        type=int,
        default=EvaluationConfig.step_num,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=EvaluationConfig.gamma,
        help="Discount factor used by the agent",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=EvaluationConfig.learning_rate,
        help="Learning rate placeholder required by the agent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for generating evaluation scenarios",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for the CSV results (default: runs/val_shared/<timestamp>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        episodes=args.episodes,
        num_missiles=args.num_missiles,
        step_num=args.step_num,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
    )

    checkpoints = collect_checkpoints(args.model_root, args.start_episode)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching 'DDQN_episode*.pth' (>= episode {args.start_episode}) "
            f"were found under '{args.model_root}'."
        )

    seeds = generate_scenario_seeds(config.episodes, args.seed)

    results: List[Tuple[int, str, float]] = []
    for episode, checkpoint_path in checkpoints:
        success_rate = evaluate_on_shared_env(checkpoint_path, config, seeds)
        print(f"Episode {episode:>4} | success rate: {success_rate:.4f} | {checkpoint_path}")
        results.append((episode, checkpoint_path, success_rate))

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("runs", "val_shared", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "intercept_success_rates.csv")
    write_results_csv(csv_path, results)
    print(f"Validation complete. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
