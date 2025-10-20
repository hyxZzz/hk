import time
import argparse
from dataclasses import dataclass
import numpy as np
import torch

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent
from utils.ERbuffer import MyMemoryBuffer
from Environment.init_env import init_env
from tensorboardX import SummaryWriter

from utils.validate import (
    EvaluationConfig,
    create_writer as create_validation_writer,
    evaluate_checkpoint,
    save_csv as save_validation_csv,
)

# MEMORY_SIZE = 40000
# MEMORY_WARMUP_SIZE = 2000
# LEARN_FREQ = 50
# BATCH_SIZE = 512
# LEARNING_RATE = 0.001
# GAMMA = 0.5

writer = SummaryWriter('./models/DQNmodels/DDQNmodels3_23/runs/train_process3_21')


# 启用环境进行训练，done=1则结束该次训练，返回奖励值
def run_train_episode(agent, env, rpmemory, MEMORY_WARMUP_SIZE, LEARN_FREQ, BATCH_SIZE):
    total_reward = 0
    train_loss = 1e8
    state, escapeFlag, info = env.reset()
    step = 0
    while True:
        step += 1
        # 智能体抽样动作
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        # print(f"reward:{reward}\n")
        # print(reward)
        rpmemory.add((state, action, reward, next_state, done))

        # 当经验回放数组中的经验数量足够多时（大于给定阈值，手动设定），每50个时间步训练一次
        if (rpmemory.size() > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            experiences = rpmemory.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)
            # 智能体更新价值网络
            train_loss = agent.learn(batch_state, batch_action, batch_reward, batch_next_state, batch_done)

        total_reward += reward
        state = next_state
        if done != -1:
            break
    return total_reward, train_loss


# 评估若干回合，返回奖励与成功率统计
@dataclass
class EvaluationMetrics:
    mean_total_reward: float
    mean_reward_per_step: float
    success_rate: float


def evaluate_agent(agent, env, eval_episodes=10, render=False):
    total_rewards = []
    per_step_rewards = []
    successes = 0

    previous_model_mode = agent.model.training
    previous_target_mode = agent.target_model.training
    previous_epsilon = agent.e_greed

    agent.model.eval()
    agent.target_model.eval()
    agent.e_greed = 0.0

    try:
        for _ in range(eval_episodes):
            state, _, _ = env.reset()
            episode_reward = 0.0
            steps = 0
            success = False

            while True:
                action = agent.predict(state)
                state, reward, done, _ = env.step(action)
                steps += 1
                episode_reward += reward

                if render:
                    env.render()

                if done != -1:
                    if done == 2:
                        success = True
                    break

            total_rewards.append(episode_reward)
            per_step_rewards.append(episode_reward / max(steps, 1))
            if success:
                successes += 1

    finally:
        agent.model.train(previous_model_mode)
        agent.target_model.train(previous_target_mode)
        agent.e_greed = previous_epsilon

    mean_total_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    mean_reward_per_step = float(np.mean(per_step_rewards)) if per_step_rewards else 0.0
    success_rate = successes / float(eval_episodes) if eval_episodes > 0 else 0.0

    return EvaluationMetrics(
        mean_total_reward=mean_total_reward,
        mean_reward_per_step=mean_reward_per_step,
        success_rate=success_rate,
    )

def main():

    parser = argparse.ArgumentParser(description='612DD')

    parser.add_argument('--memory_size', type=int, default=60000, help='Size of replay memory')
    parser.add_argument('--memory_warmup_size', type=int, default=4000, help='Warmup size of replay memory')
    parser.add_argument('--learn_freq', type=int, default=20, help='Frequency of learning')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.993, help='Discount factor')
    parser.add_argument(
        '--target_update_freq',
        type=int,
        default=15,
        help='Number of learning steps between target network updates',
    )
    parser.add_argument('--max_episode', type=int, default=1000, help='Maximum number of episodes')
    parser.add_argument(
        '--validation_episodes',
        type=int,
        default=EvaluationConfig.episodes,
        help='Number of validation episodes to run after each checkpoint save',
    )

    args = parser.parse_args()

    MEMORY_SIZE = args.memory_size
    MEMORY_WARMUP_SIZE = args.memory_warmup_size
    LEARN_FREQ = args.learn_freq
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    TARGET_UPDATE_FREQ = args.target_update_freq

    start = time.time()

    num_missiles = 3
    step_num = 3500
    Env, aircraft, missiles = init_env(
        num_missiles=num_missiles,
        StepNum=step_num,
    )

    action_size = Env._get_actSpace()

    state_size = Env._getNewStateSpace()[0]
    # print(state_size)

    # 初始化经验数组
    rpm = MyMemoryBuffer(MEMORY_SIZE)

    # 生成智能体
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

    max_episode = 2000

    validation_config = EvaluationConfig(
        episodes=args.validation_episodes,
        num_missiles=num_missiles,
        step_num=step_num,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
    )
    val_writer, val_log_dir = create_validation_writer()
    validation_results = []
    validation_csv_path = None

    train_loss = 0

    # start training
    start_time = time.time()
    print('start training...')
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(50):
            total_reward, train_loss = run_train_episode(agent, Env, rpm, MEMORY_WARMUP_SIZE, LEARN_FREQ, BATCH_SIZE)
            writer.add_scalar('train/loss', train_loss, episode)
            episode += 1

        # test part
        if episode % 50 == 0:
            eval_metrics = evaluate_agent(agent, Env, eval_episodes=args.validation_episodes, render=False)
            writer.add_scalar('eval/mean_total_reward', eval_metrics.mean_total_reward, episode)
            writer.add_scalar('eval/mean_reward_per_step', eval_metrics.mean_reward_per_step, episode)
            writer.add_scalar('eval/success_rate', eval_metrics.success_rate, episode)
            print(
                'episode:{}    e_greed:{}   Eval total reward:{:.4f}   Reward/step:{:.4f}   Success rate:{:.2%}   Train Loss:{}'.format(
                    episode,
                    agent.e_greed,
                    eval_metrics.mean_total_reward,
                    eval_metrics.mean_reward_per_step,
                    eval_metrics.success_rate,
                    train_loss,
                )
            )
        if episode % 100 == 0:
            ## 保存模型
            checkpoint_path = './models/DQNmodels/DDQNmodels3_23/DDQN_episode{}.pth'.format(episode)
            torch.save({'model': model.state_dict()}, checkpoint_path)

            success_rate = evaluate_checkpoint(checkpoint_path, validation_config)
            val_writer.add_scalar('intercept_success_rate', success_rate, episode)
            validation_results.append((episode, checkpoint_path, success_rate))
            validation_csv_path = save_validation_csv(val_log_dir, validation_results)
            print(
                'Validation after episode {}: intercept success rate {:.4f}'.format(
                    episode, success_rate
                )
            )
            print('Validation results saved to {}'.format(validation_csv_path))

    print('all used time {:.2}s = {:.2}h'.format(time.time() - start_time, (time.time() - start_time) / 3600))
    # state, reward, escapeFlag, info = Env.step(15)
    # print(state)
    end = time.time()
    total_train_time = end - start
    print("FINAL")
    print(f"total train time for {max_episode} games = {total_train_time} sec")
    val_writer.close()
    # ag = Aircraft(Env.aircraftList, V, Pitch, Heading, dt=0.01, g=9.6     g.num_unique_cards, g.card_dict, cache_limit, epsilon)

    # state_size = g.num_unique_cards + 1  # playable cards + 1 card on top of play deck
    # action_size = g.num_unique_cards  # playable cards

    # init deep q network (it's just a simple feedforward bro)
    # dqn = DQN(state_size, action_size)


main()
