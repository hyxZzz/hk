import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent
from Environment.env import SOFT_CONSTRAINT_DEFAULT
from Environment.init_env import init_env
from utils.ERbuffer import PrioritizedReplayBuffer
from utils.validate import (
    EvaluationConfig,
    EvaluationSummary,
    create_writer as create_validation_writer,
    evaluate_checkpoint,
    save_csv as save_validation_csv,
)


writer = SummaryWriter("./models/DQNmodels/DDQNmodels3_23/runs/train_process_multi_agent")


@dataclass
class EvaluationMetrics:
    mean_total_reward: Dict[int, float]
    success_rate: float
    mean_interceptor_launches: float
    illegal_launch_rate: float
    mean_overkill: float
    mean_cooperative_intercepts: float
    mean_unprotected_rate: float
    mean_survival_time: float
    survival_time_distribution: List[float]


def run_train_episode(
    agents: Dict[int, MyDQNAgent],
    env,
    rpmemory: PrioritizedReplayBuffer,
    MEMORY_WARMUP_SIZE: int,
    LEARN_FREQ: int,
    BATCH_SIZE: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    total_reward = {agent_id: 0.0 for agent_id in agents}
    train_loss = {agent_id: 0.0 for agent_id in agents}
    state, escapeFlag, _ = env.reset()
    step = 0

    while True:
        step += 1
        state_masks = {agent_id: env.valid_action_mask(agent_id) for agent_id in agents}
        actions = {
            agent_id: agents[agent_id].sample(state[agent_id], state_masks[agent_id])
            for agent_id in agents
        }
        next_state, reward, done, _ = env.step(actions)

        next_masks = None if done != -1 else {
            agent_id: env.valid_action_mask(agent_id) for agent_id in agents
        }

        rpmemory.add(state, actions, reward, next_state, done, state_masks, next_masks)

        if (rpmemory.size() > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            experiences, indices, weights = rpmemory.sample(BATCH_SIZE)
            if len(experiences) > 0:
                priority_updates = np.zeros(len(indices), dtype=np.float32)
                unique_agents: Dict[int, Tuple[MyDQNAgent, List[int]]] = {}
                for agent_id, agent in agents.items():
                    unique_agents.setdefault(id(agent), (agent, []))[1].append(agent_id)

                for _, (agent, plane_ids) in unique_agents.items():
                    for plane_id in plane_ids:
                        batch_state = [exp["state"][plane_id] for exp in experiences]
                        batch_action = [exp["actions"][plane_id] for exp in experiences]
                        batch_reward = [exp["reward"].get(plane_id, 0.0) for exp in experiences]
                        batch_next_state = [exp["next_state"][plane_id] for exp in experiences]
                        batch_done = [exp["done"] for exp in experiences]
                        batch_next_masks = [
                            None if exp["next_masks"] is None else exp["next_masks"].get(plane_id)
                            for exp in experiences
                        ]
                        batch_n_steps = [exp["n_steps"] for exp in experiences]
                        loss, td_errors = agent.learn(
                            batch_state,
                            batch_action,
                            batch_reward,
                            batch_next_state,
                            batch_done,
                            weights=weights,
                            next_valid_actions=batch_next_masks,
                            n_steps=batch_n_steps,
                        )
                        train_loss[plane_id] = loss
                        priority_updates = np.maximum(priority_updates, np.abs(td_errors))

                rpmemory.update_priorities(indices, priority_updates)

        for agent_id in agents:
            total_reward[agent_id] += reward.get(agent_id, 0.0)

        state = next_state
        if done != -1:
            break

    rpmemory.on_episode_end()

    return total_reward, train_loss


def evaluate_agents(
    agents: Dict[int, MyDQNAgent],
    env,
    eval_episodes: int = 10,
    render: bool = False,
) -> EvaluationMetrics:
    total_rewards = {agent_id: [] for agent_id in agents}
    successes = 0
    launch_counts: List[int] = []
    illegal_launches = 0
    total_launch_attempts = 0
    intercept_lock_counts: List[int] = []
    cooperative_counts: List[int] = []
    unprotected_rates: List[float] = []
    survival_times: List[float] = []

    previous_modes = {
        agent_id: (
            agent.model.training,
            agent.target_model.training,
            agent.e_greed,
        )
        for agent_id, agent in agents.items()
    }

    for agent in agents.values():
        agent.model.eval()
        agent.target_model.eval()
        agent.e_greed = 0.0

    try:
        for _ in range(eval_episodes):
            state, _, _ = env.reset()
            episode_reward = {agent_id: 0.0 for agent_id in agents}

            episode_info = None
            while True:
                state_masks = {agent_id: env.valid_action_mask(agent_id) for agent_id in agents}
                actions = {
                    agent_id: agent.predict(state[agent_id], state_masks[agent_id])
                    for agent_id, agent in agents.items()
                }
                state, reward, done, info = env.step(actions)
                episode_info = info
                for agent_id in agents:
                    episode_reward[agent_id] += reward.get(agent_id, 0.0)

                if render:
                    env.render()

                if done != -1:
                    if done == 2:
                        successes += 1
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

            for agent_id in agents:
                total_rewards[agent_id].append(episode_reward[agent_id])

    finally:
        for agent_id, agent in agents.items():
            model_mode, target_mode, epsilon = previous_modes[agent_id]
            agent.model.train(model_mode)
            agent.target_model.train(target_mode)
            agent.e_greed = epsilon

    mean_total_reward = {
        agent_id: float(np.mean(reward_list)) if reward_list else 0.0
        for agent_id, reward_list in total_rewards.items()
    }
    success_rate = successes / float(eval_episodes) if eval_episodes > 0 else 0.0
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

    return EvaluationMetrics(
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


def main():
    parser = argparse.ArgumentParser(description="Multi-agent DDQN training")
    parser.add_argument("--memory_size", type=int, default=60000, help="Size of replay memory")
    parser.add_argument("--memory_warmup_size", type=int, default=4000, help="Warmup size of replay memory")
    parser.add_argument("--learn_freq", type=int, default=20, help="Frequency of learning")
    parser.add_argument("--batch_size", type=int, default=384, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--gamma", type=float, default=0.993, help="Discount factor")
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=15,
        help="Number of learning steps between target network updates",
    )
    parser.add_argument("--max_episode", type=int, default=1000, help="Maximum number of episodes")
    parser.add_argument(
        "--validation_episodes",
        type=int,
        default=EvaluationConfig.episodes,
        help="Number of validation episodes to run after each checkpoint save",
    )
    parser.add_argument("--n_step", type=int, default=3, help="Number of steps for n-step TD updates")
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=0.6,
        help="Prioritized replay alpha parameter",
    )
    parser.add_argument(
        "--per_beta_start",
        type=float,
        default=0.4,
        help="Initial beta value for prioritized replay",
    )
    parser.add_argument(
        "--per_beta_frames",
        type=int,
        default=100000,
        help="Number of sampling steps to anneal beta to 1.0",
    )
    parser.add_argument(
        "--soft_penalty_scale",
        type=float,
        default=SOFT_CONSTRAINT_DEFAULT,
        help="Soft penalty applied when violating launch cooldown or lock limits",
    )

    args = parser.parse_args()

    MEMORY_SIZE = args.memory_size
    MEMORY_WARMUP_SIZE = args.memory_warmup_size
    LEARN_FREQ = args.learn_freq
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    TARGET_UPDATE_FREQ = args.target_update_freq
    N_STEP = max(1, args.n_step)
    PER_ALPHA = args.per_alpha
    PER_BETA_START = args.per_beta_start
    PER_BETA_FRAMES = args.per_beta_frames
    soft_penalty_scale = args.soft_penalty_scale

    num_missiles = 4
    interceptors_per_plane = 6
    num_planes = 2
    step_num = 3500

    env, aircraft, missiles = init_env(
        num_missiles=num_missiles,
        StepNum=step_num,
        interceptor_num=interceptors_per_plane,
        num_planes=num_planes,
        soft_penalty_scale=soft_penalty_scale,
    )

    action_size = env._get_actSpace()
    state_size = env._getNewStateSpace()[0]

    rpm = PrioritizedReplayBuffer(
        MEMORY_SIZE,
        gamma=GAMMA,
        n_step=N_STEP,
        alpha=PER_ALPHA,
        beta_start=PER_BETA_START,
        beta_frames=PER_BETA_FRAMES,
    )

    shared_model = Double_DQN(state_size=state_size, action_size=action_size)
    shared_agent = MyDQNAgent(
        shared_model,
        action_size,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        e_greed=0.85,
        e_greed_decrement=5e-7,
        update_target_steps=TARGET_UPDATE_FREQ,
        target_tau=0.05,
    )
    agents: Dict[int, MyDQNAgent] = {agent_id: shared_agent for agent_id in range(num_planes)}

    validation_config = EvaluationConfig(
        episodes=args.validation_episodes,
        num_missiles=num_missiles,
        num_planes=num_planes,
        interceptor_num=interceptors_per_plane,
        step_num=step_num,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        soft_penalty_scale=soft_penalty_scale,
    )
    val_writer, val_log_dir = create_validation_writer()
    validation_results: List[Tuple[int, Dict[int, str], EvaluationSummary]] = []
    validation_csv_path = None

    max_episode = args.max_episode
    start_time = time.time()
    episode = 0

    print("start training...")
    while episode < max_episode:
        for _ in range(50):
            total_reward, train_loss = run_train_episode(
                agents,
                env,
                rpm,
                MEMORY_WARMUP_SIZE,
                LEARN_FREQ,
                BATCH_SIZE,
            )
            for agent_id in agents:
                writer.add_scalar(f"train/loss_agent_{agent_id}", train_loss[agent_id], episode)
                writer.add_scalar(
                    f"train/total_reward_agent_{agent_id}", total_reward[agent_id], episode
                )
            episode += 1

        if episode % 50 == 0:
            eval_metrics = evaluate_agents(agents, env, eval_episodes=args.validation_episodes, render=False)
            for agent_id, reward_value in eval_metrics.mean_total_reward.items():
                writer.add_scalar(f"eval/mean_total_reward_agent_{agent_id}", reward_value, episode)
            writer.add_scalar("eval/success_rate", eval_metrics.success_rate, episode)
            writer.add_scalar(
                "eval/mean_interceptor_launches", eval_metrics.mean_interceptor_launches, episode
            )
            writer.add_scalar("eval/illegal_launch_rate", eval_metrics.illegal_launch_rate, episode)
            writer.add_scalar("eval/mean_overkill", eval_metrics.mean_overkill, episode)
            writer.add_scalar(
                "eval/mean_cooperative_intercepts", eval_metrics.mean_cooperative_intercepts, episode
            )
            writer.add_scalar("eval/mean_unprotected_rate", eval_metrics.mean_unprotected_rate, episode)
            writer.add_scalar("eval/mean_survival_time", eval_metrics.mean_survival_time, episode)
            print(
                "episode:{}    e_greed:{}   Success rate:{:.2%}".format(
                    episode,
                    np.mean([agent.e_greed for agent in agents.values()]),
                    eval_metrics.success_rate,
                )
            )

        if episode % 100 == 0:
            checkpoint_paths: Dict[int, str] = {}
            for agent_id, agent in agents.items():
                checkpoint_path = (
                    f"./models/DQNmodels/DDQNmodels3_23/"
                    f"DDQN_agent{agent_id}_episode{episode}.pth"
                )
                torch.save({"model": agent.model.state_dict()}, checkpoint_path)
                checkpoint_paths[agent_id] = checkpoint_path

            validation_summary = evaluate_checkpoint(checkpoint_paths, validation_config)
            val_writer.add_scalar("intercept_success_rate", validation_summary.success_rate, episode)
            val_writer.add_scalar(
                "val/mean_interceptor_launches", validation_summary.mean_interceptor_launches, episode
            )
            val_writer.add_scalar("val/illegal_launch_rate", validation_summary.illegal_launch_rate, episode)
            val_writer.add_scalar("val/mean_overkill", validation_summary.mean_overkill, episode)
            val_writer.add_scalar(
                "val/mean_cooperative_intercepts", validation_summary.mean_cooperative_intercepts, episode
            )
            val_writer.add_scalar("val/mean_unprotected_rate", validation_summary.mean_unprotected_rate, episode)
            val_writer.add_scalar("val/mean_survival_time", validation_summary.mean_survival_time, episode)
            validation_results.append((episode, dict(checkpoint_paths), validation_summary))
            validation_csv_path = save_validation_csv(val_log_dir, validation_results)
            print(
                "Validation after episode {}: intercept success rate {:.4f}".format(
                    episode, validation_summary.success_rate
                )
            )
            if validation_csv_path:
                print("Validation results saved to {}".format(validation_csv_path))

    elapsed = time.time() - start_time
    print("all used time {:.2f}s = {:.2f}h".format(elapsed, elapsed / 3600))
    val_writer.close()


if __name__ == "__main__":
    main()
