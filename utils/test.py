"""
使用指定的 DDQN 权重在一批共享环境上重复验证（seed=1..100）。

本脚本固定使用
    /home/data/heyuxin/hangkong/1014/hk1/hk3/models/DQNmodels/DDQNmodels3_23/DDQN_episode300.pth
这个 checkpoint，针对 seed=1 到 100（共 100 次），每次生成相同数量（100）的评估场景，
顺序评估并输出每次的拦截成功率，同时把所有结果保存为 CSV。

示例用法（全部参数都有默认值，可以直接运行）::

    python -m utils.test \
        --checkpoint /home/data/heyuxin/hangkong/1014/hk1/hk3/models/DQNmodels/DDQNmodels3_23/DDQN_episode300.pth \
        --episodes 100 \
        --start-seed 1 \
        --end-seed 100

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
    load_checkpoint,
    select_action,
)


@dataclass(frozen=True)
class ScenarioSeeds:
    """生成并复现实验场景所需的种子集合。"""

    env_seed: int
    episode_seeds: Sequence[int]


def generate_scenario_seeds(
    num_episodes: int,
    base_seed: int | None = None,
) -> ScenarioSeeds:
    """给定 base_seed，生成确定性的环境种子及每个 episode 的种子。"""

    rng = np.random.default_rng(base_seed)
    env_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    episode_seeds = [int(seed) for seed in rng.integers(0, np.iinfo(np.uint32).max, size=num_episodes)]
    return ScenarioSeeds(env_seed=env_seed, episode_seeds=episode_seeds)


def evaluate_on_shared_env(
    checkpoint_path: str,
    config: EvaluationConfig,
    seeds: ScenarioSeeds,
) -> float:
    """在一批共享（可复现）的场景上评估给定 checkpoint，返回成功率。"""

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
    """将 (seed, checkpoint, success_rate) 结果以 CSV 形式写入 path。"""

    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["seed", "checkpoint", "success_rate"])
        for seed, checkpoint, success_rate in results:
            writer.writerow([seed, checkpoint, f"{success_rate:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用固定 DDQN 权重，针对 seed=1..100（可配置）进行评估；"
            "每个 seed 评估相同数量的 episodes（默认 100）。"
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/data/heyuxin/hangkong/1014/hk1/hk3/models/DQNmodels/DDQNmodels3_23/DDQN_episode300.pth",
        help="要加载的 DDQN checkpoint 路径（默认为题目要求的 episode300.pth）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="每次评估所运行的 episode 数（默认 100）",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="起始 seed（包含，默认 1）",
    )
    parser.add_argument(
        "--end-seed",
        type=int,
        default=100,
        help="结束 seed（包含，默认 100）",
    )
    parser.add_argument(
        "--num-missiles",
        type=int,
        default=EvaluationConfig.num_missiles,
        help="评估时使用的来袭导弹数量",
    )
    parser.add_argument(
        "--step-num",
        type=int,
        default=EvaluationConfig.step_num,
        help="每个 episode 的最大步数",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=EvaluationConfig.gamma,
        help="Agent 的折扣因子",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=EvaluationConfig.learning_rate,
        help="Agent 需要的学习率占位参数",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="结果 CSV 的输出目录（默认：runs/val_shared/<timestamp>）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 按照用户要求：每次验证 100 轮（episodes），seed 从 1 到 100
    config = EvaluationConfig(
        episodes=args.episodes,
        num_missiles=args.num_missiles,
        step_num=args.step_num,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
    )

    # 逐个 seed 评估同一个 checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在：{checkpoint_path}")

    if args.start_seed > args.end_seed:
        raise ValueError(f"--start-seed({args.start_seed}) 不能大于 --end-seed({args.end_seed})")

    results: List[Tuple[int, str, float]] = []
    for seed in range(args.start_seed, args.end_seed + 1):
        seeds = generate_scenario_seeds(config.episodes, base_seed=seed)
        success_rate = evaluate_on_shared_env(checkpoint_path, config, seeds)
        print(f"Seed {seed:>3} | success rate: {success_rate:.4f} | {checkpoint_path}")
        results.append((seed, checkpoint_path, success_rate))

    # 写入 CSV
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("runs", "val_shared", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "intercept_success_rates.csv")
    write_results_csv(csv_path, results)
    print(f"验证完成，共 {len(results)} 次。结果已保存到 {csv_path}")


if __name__ == "__main__":
    main()
