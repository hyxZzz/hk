import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent
from Environment.init_env import init_env
from utils.ERbuffer import MyMemoryBuffer
from utils.validate import (
    EvaluationConfig,
    create_writer as create_validation_writer,
    evaluate_checkpoint,
    save_csv as save_validation_csv,
)


writer = SummaryWriter("./models/DQNmodels/DDQNmodels3_23/runs/train_process_multi_agent")


@dataclass
class EvaluationMetrics:
    mean_total_reward: Dict[int, float]
    success_rate: float


def run_train_episode(
    agents: Dict[int, MyDQNAgent],
    env,
    rpmemory: MyMemoryBuffer,
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
        actions = {agent_id: agents[agent_id].sample(state[agent_id]) for agent_id in agents}
        next_state, reward, done, _ = env.step(actions)

        rpmemory.add((state, actions, reward, next_state, done))

        if (rpmemory.size() > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            experiences = rpmemory.sample(BATCH_SIZE)
            for agent_id, agent in agents.items():
                batch_state = [exp[0][agent_id] for exp in experiences]
                batch_action = [exp[1][agent_id] for exp in experiences]
                batch_reward = [exp[2][agent_id] for exp in experiences]
                batch_next_state = [exp[3][agent_id] for exp in experiences]
                batch_done = [exp[4] for exp in experiences]
                train_loss[agent_id] = agent.learn(
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done,
                )

        for agent_id in agents:
            total_reward[agent_id] += reward.get(agent_id, 0.0)

        state = next_state
        if done != -1:
            break

    return total_reward, train_loss


def evaluate_agents(
    agents: Dict[int, MyDQNAgent],
    env,
    eval_episodes: int = 10,
    render: bool = False,
) -> EvaluationMetrics:
    total_rewards = {agent_id: [] for agent_id in agents}
    successes = 0

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

            while True:
                actions = {agent_id: agent.predict(state[agent_id]) for agent_id, agent in agents.items()}
                state, reward, done, _ = env.step(actions)
                for agent_id in agents:
                    episode_reward[agent_id] += reward.get(agent_id, 0.0)

                if render:
                    env.render()

                if done != -1:
                    if done == 2:
                        successes += 1
                    break

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

    return EvaluationMetrics(mean_total_reward=mean_total_reward, success_rate=success_rate)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent DDQN training")
    parser.add_argument("--memory_size", type=int, default=60000, help="Size of replay memory")
    parser.add_argument("--memory_warmup_size", type=int, default=4000, help="Warmup size of replay memory")
    parser.add_argument("--learn_freq", type=int, default=20, help="Frequency of learning")
    parser.add_argument("--batch_size", type=int, default=384, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
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

    args = parser.parse_args()

    MEMORY_SIZE = args.memory_size
    MEMORY_WARMUP_SIZE = args.memory_warmup_size
    LEARN_FREQ = args.learn_freq
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    TARGET_UPDATE_FREQ = args.target_update_freq

    num_missiles = 4
    interceptors_per_plane = 6
    num_planes = 2
    step_num = 3500

    env, aircraft, missiles = init_env(
        num_missiles=num_missiles,
        StepNum=step_num,
        interceptor_num=interceptors_per_plane,
        num_planes=num_planes,
    )

    action_size = env._get_actSpace()
    state_size = env._getNewStateSpace()[0]

    rpm = MyMemoryBuffer(MEMORY_SIZE)

    agents: Dict[int, MyDQNAgent] = {}
    for agent_id in range(num_planes):
        model = Double_DQN(state_size=state_size, action_size=action_size)
        agent = MyDQNAgent(
            model,
            action_size,
            gamma=GAMMA,
            lr=LEARNING_RATE,
            e_greed=0.85,
            e_greed_decrement=5e-7,
            update_target_steps=TARGET_UPDATE_FREQ,
        )
        agents[agent_id] = agent

    validation_config = EvaluationConfig(
        episodes=args.validation_episodes,
        num_missiles=num_missiles,
        num_planes=num_planes,
        interceptor_num=interceptors_per_plane,
        step_num=step_num,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
    )
    val_writer, val_log_dir = create_validation_writer()
    validation_results: List[Tuple[int, Dict[int, str], float]] = []
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

            success_rate = evaluate_checkpoint(checkpoint_paths, validation_config)
            val_writer.add_scalar("intercept_success_rate", success_rate, episode)
            validation_results.append((episode, dict(checkpoint_paths), success_rate))
            validation_csv_path = save_validation_csv(val_log_dir, validation_results)
            print(
                "Validation after episode {}: intercept success rate {:.4f}".format(
                    episode, success_rate
                )
            )
            if validation_csv_path:
                print("Validation results saved to {}".format(validation_csv_path))

    elapsed = time.time() - start_time
    print("all used time {:.2f}s = {:.2f}h".format(elapsed, elapsed / 3600))
    val_writer.close()


if __name__ == "__main__":
    main()
