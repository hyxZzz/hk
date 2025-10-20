import copy
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from matplotlib import animation

from Environment.env import ManeuverEnv
from DDQN.DQNAgent import MyDQNAgent
from Environment.init_env import init_env
from DDQN.DDQN import Double_DQN
from flat_models.ThreatEvaluate import CalTreat, intervalEvaluation

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def squeezeState(NormalState):
    return NormalState.flatten()


def resize_flattenState(flattenState, rows=None):
    if rows is None:
        if flattenState.size % 6 != 0:
            raise ValueError("flattenState length is not divisible by 6; please provide 'rows' explicitly")
        rows = flattenState.size // 6
    return np.resize(flattenState, (rows, 6))


def calTreatFromState(apos, mpos, v_a, v_m):
    return CalTreat(apos, mpos, v_a, v_m)


LEARNING_RATE = 5e-4
MAXSTEP = 3500
GAMMA = 0.993
DEFAULT_NUM_MISSILES = 3
DEFAULT_INTERCEPTOR_NUM = 8


# 预测函数，输入模型的保存地址，进行一次预测，返回一次游戏的state序列

def predictResult(model_path):
    # 生成2个导弹的 随机的 环境
    Env, aircraft, missiles = init_env(
        num_missiles=DEFAULT_NUM_MISSILES,
        StepNum=3500,
        interceptor_num=DEFAULT_INTERCEPTOR_NUM,
    )
    num_missiles = Env.missileNum
    escapeFlag = -1

    state_size = Env._getNewStateSpace()[0]
    action_size = Env._get_actSpace()
    # # 生成智能体
    model = Double_DQN(state_size=state_size, action_size=action_size)

    state_dic = torch.load(model_path, map_location='cuda:0')
    new_state = {}
    for k, v in state_dic.items():  # 去除关键字”model"
        new_state = v
    # for k, v in new_state.items():
    #     print(k)

    model.load_state_dict(new_state)

    agent = MyDQNAgent(model, action_size, gamma=GAMMA, lr=LEARNING_RATE, e_greed=0.1, e_greed_decrement=1e-6)

    state = np.zeros((MAXSTEP + 1, state_size), dtype=np.float32)
    obs_rows = Env._get_obs().shape[0]
    statefromEnv = np.zeros((MAXSTEP + 1, obs_rows, 6), dtype=np.float32)
    Treat_value = np.zeros((MAXSTEP,), dtype=np.float32)
    state[0], escapeFlag, info = Env.reset()
    state_copy = [state[0], escapeFlag, info]
    Env_compare = copy.deepcopy(Env)
    # print(Env.aircraftList.X, Env.aircraftList.Y, Env.aircraftList.Z)
    statefromEnv[0] = Env._get_obs()

    # 记录步数，防止画图时过多0点
    t = 0
    Treat = 0
    # 循环
    for i in range(MAXSTEP):
        # 通过智能体模型选动作
        # print(state[i])
        act = agent.predict(state[i])
        # print("act")
        # print(act)
        # 与环境交互
        # if i % 2 == 0:
        #     act = 62
        # else:
        #     act = 61

        next_state, reward, escapeFlag, info = Env.step(act)
        # 获取环境中飞机的速度
        v_a = Env.aircraftList.V
        apos = [Env.aircraftList.X, Env.aircraftList.Y, Env.aircraftList.Z]
        for j in range(num_missiles):
            # 计算威胁度：这里的威胁度将已失效的导弹减去
            if Env.missileList[j].attacking:
                v_m = Env.missileList[j].V
                mpos = [Env.missileList[j].X, Env.missileList[j].Y, Env.missileList[j].Z]
                T = calTreatFromState(apos, mpos, v_a, v_m)
                if T > 100 or T < 0:
                    print(T)
                if Treat < calTreatFromState(apos, mpos, v_a, v_m):
                    Treat_value[i] = calTreatFromState(apos, mpos, v_a, v_m)
                if Treat_value[i] > 100 or Treat_value[i] < 0:
                    print(Treat_value[i])
                # print(Treat_value[i])
        EvaResult = intervalEvaluation(Treat_value[i])
        # print(Treat_value[i], EvaResult)
        # 记录下一个时刻状态
        # 拉长向量
        statefromEnv[i + 1] = Env._get_obs()
        state[i + 1] = next_state
        t = i + 1
        if escapeFlag != -1:
            # print(info)
            break
    remain_intceptor = Env.interceptor_remain
    if escapeFlag == 1:
        Treat_value[t - 1] = 0

    return t, statefromEnv, Treat_value, Env_compare, state_copy, remain_intceptor, escapeFlag


# 对比预测函数，随机策略展示强化学习的作用，进行一次预测，返回一次游戏的state序列

def ComparepredictResult(Env: ManeuverEnv, state_copy: List):
    # print(Env.aircraftList.X, Env.aircraftList.Y, Env.aircraftList.Z)
    state_size = Env._getNewStateSpace()[0]
    # # 生成智能体
    escapeFlag = -1
    state = np.zeros((MAXSTEP + 1, state_size), dtype=np.float32)
    obs_rows = Env._get_obs().shape[0]
    statefromEnv = np.zeros((MAXSTEP + 1, obs_rows, 6), dtype=np.float32)
    Treat_value = np.zeros((MAXSTEP,), dtype=np.float32)
    state[0], escapeFlag, info = state_copy
    statefromEnv[0] = Env._get_obs()

    # 记录步数，防止画图时过多0点
    t = 0
    Treat = 0
    # 循环
    for i in range(MAXSTEP):
        # 通过智能体模型选动作
        act = 0
        # act = 2
        # 与环境交互
        next_state, reward, escapeFlag, info = Env.compareTest(act)
        # print(act)
        # 获取环境中飞机的速度
        v_a = Env.aircraftList.V
        apos = [Env.aircraftList.X, Env.aircraftList.Y, Env.aircraftList.Z]
        for j in range(Env.missileNum):
            # 计算威胁度：这里的威胁度将已失效的导弹减去
            if Env.missileList[j].attacking:
                v_m = Env.missileList[j].V
                mpos = [Env.missileList[j].X, Env.missileList[j].Y, Env.missileList[j].Z]
                if Treat < calTreatFromState(apos, mpos, v_a, v_m):
                    Treat_value[i] = calTreatFromState(apos, mpos, v_a, v_m)
                if Treat_value[i] > 100 or Treat_value[i] < 0:
                    print(Treat_value[i])
        # 记录下一个时刻状态
        # 拉长向量
        statefromEnv[i + 1] = Env._get_obs()
        state[i + 1] = next_state
        t = i + 1
        if escapeFlag != -1:
            # print(info)
            break
    interceptor_remain = Env.interceptor_remain

    return t, statefromEnv, Treat_value, interceptor_remain, escapeFlag


def aniPlot(t, statefromEnv, Treat_value):
    # plt.style.use('ggplot')
    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d', alpha=0.8)
    ax2 = fig.add_subplot(122)

    y = np.arange(1, 4, 1)
    y_unique = np.unique(y - 1)  # 可以看作图例类型个数
    color = ['r', 'g', 'b', 'k']
    methods = ('来袭导弹', '拦截弹群', '飞行器', 'Interceptor')
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in y_unique]
    legend_labels = [methods[y] for y in y_unique]
    ax1.legend(legend_lines, legend_labels, numpoints=1, title='机动与拦截', fontsize=18)

    # 定义标签
    ax1.set_xlabel('正北方位移')
    ax1.set_ylabel('正东方位移')
    ax1.set_zlabel('水平高度')
    Multistate = []
    Multistate.append(statefromEnv)
    Multistate.append(Treat_value)
    axList = []
    axList.append(ax1)
    axList.append(ax2)
    ani = animation.FuncAnimation(fig, showPredictProgress, frames=t, fargs=(Multistate, axList), interval=1,
                                  blit=False)
    plt.show()


def showPredictProgress(t, Multistate, axList):
    state = Multistate[0]
    Treat_value = Multistate[1]
    ax1 = axList[0]
    ax2 = axList[1]
    maxstep = t
    missileNum = DEFAULT_NUM_MISSILES
    planeState = state[:, 0, 0:3]
    missileState = state[:, 1:missileNum + 1, 0:3]
    interceptorState = state[:, missileNum + 1:, 0:3]
    ax1.plot(planeState[0:maxstep, 0], planeState[0:maxstep, 2], planeState[0:maxstep, 1], linewidth=4, c='b')
    for i in range(missileNum):
        ax1.plot(missileState[0:maxstep, i, 0], missileState[0:maxstep, i, 2], missileState[0:maxstep, i, 1],
                 linewidth=2, c='r')

    for i in range(DEFAULT_INTERCEPTOR_NUM):
        ax1.plot(interceptorState[0:maxstep, i, 0], interceptorState[0:maxstep, i, 2],
                 interceptorState[0:maxstep, i, 1],
                 linewidth=2, c='g')
    # for i in range(missileNum):
    #     ax1.plot(:, )
    arrayx = np.arange(0, Treat_value[0:maxstep].shape[0])
    ax2.plot(arrayx, Treat_value[0:maxstep])


def showPredictResult(t, statefromEnv, Treat_value):
    # style.use('ggplot')
    fig = plt.figure()
    state = statefromEnv
    Treat_value = copy.deepcopy(Treat_value[0:t])
    maxstep = t
    missileNum = DEFAULT_NUM_MISSILES
    ax1 = fig.add_subplot(121, projection='3d', alpha=0.8)
    ax2 = fig.add_subplot(122)

    y = np.arange(1, 4, 1)
    y_unique = np.unique(y - 1)  # 可以看作图例类型个数
    color = ['r', 'g', 'b', 'k']
    methods = ('来袭导弹', '拦截弹群', '飞行器', 'Interceptor')
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in y_unique]
    legend_labels = [methods[y] for y in y_unique]
    ax1.legend(legend_lines, legend_labels, numpoints=1, title='机动与拦截', fontsize=18)
    planeState = state[:, 0, 0:3]
    missileState = state[:, 1:missileNum + 1, 0:3]
    interceptorState = state[:, missileNum + 1:, 0:3]
    ax1.plot(planeState[0:maxstep, 0], planeState[0:maxstep, 2], planeState[0:maxstep, 1], linewidth=4, c='b')
    for i in range(missileNum):
        ax1.plot(missileState[0:maxstep, i, 0], missileState[0:maxstep, i, 2], missileState[0:maxstep, i, 1],
                 linewidth=2, c='r')

    for i in range(DEFAULT_INTERCEPTOR_NUM):
        ax1.plot(interceptorState[0:maxstep, i, 0], interceptorState[0:maxstep, i, 2],
                 interceptorState[0:maxstep, i, 1],
                 linewidth=2, c='g')
    # for i in range(missileNum):
    #     ax1.plot(:, )
    arrayx = np.arange(0, maxstep)
    ax2.plot(arrayx, Treat_value)
    # 定义标签
    ax1.set_xlabel('x正北方位移')
    ax1.set_ylabel('z正东方位移')
    ax1.set_zlabel('y垂直高度')

    plt.show()


def compare_figure(t_p, statefromEnv_p, Treat_value_p, t_com, statefromEnv_com, Treat_value_com):
    fig = plt.figure()
    state_p = statefromEnv_p
    state_com = statefromEnv_com

    # 预测的威胁值和比较的预测值
    Treat_value_p = copy.deepcopy(Treat_value_p[0:t_p])
    Treat_value_com = copy.deepcopy(Treat_value_com[0:t_com])
    maxstep = t_p
    missileNum = DEFAULT_NUM_MISSILES
    ax1 = fig.add_subplot(221, projection='3d', alpha=0.8)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection='3d', alpha=0.8)
    ax4 = fig.add_subplot(224)

    y = np.arange(1, 4, 1)
    y_unique = np.unique(y - 1)  # 可以看作图例类型个数
    color = ['r', 'g', 'b', 'k']
    methods = ('来袭导弹', '拦截弹群', '飞行器', 'Interceptor')
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none", marker='o', c=color[y]) for y in y_unique]
    legend_labels = [methods[y] for y in y_unique]
    ax1.legend(legend_lines, legend_labels, numpoints=1, title='机动与拦截-预测', fontsize=5)
    ax3.legend(legend_lines, legend_labels, numpoints=1, title='机动与拦截-对比', fontsize=5)
    planeState = state_p[:, 0, 0:3]
    missileState = state_p[:, 1:missileNum + 1, 0:3]
    interceptorState = state_p[:, missileNum + 1:, 0:3]
    ax1.plot(planeState[0:maxstep, 0], planeState[0:maxstep, 2], planeState[0:maxstep, 1], linewidth=4, c='b')
    for i in range(missileNum):
        ax1.plot(missileState[0:maxstep, i, 0], missileState[0:maxstep, i, 2], missileState[0:maxstep, i, 1],
                 linewidth=2, c='r')

    for i in range(DEFAULT_INTERCEPTOR_NUM):
        ax1.plot(interceptorState[0:maxstep, i, 0], interceptorState[0:maxstep, i, 2],
                 interceptorState[0:maxstep, i, 1],
                 linewidth=2, c='g')

    # 对比的数据
    maxstep = t_com
    planeState = state_com[:, 0, 0:3]
    missileState = state_com[:, 1:missileNum + 1, 0:3]
    interceptorState = state_com[:, missileNum + 1:, 0:3]
    ax3.plot(planeState[0:maxstep, 0], planeState[0:maxstep, 2], planeState[0:maxstep, 1], linewidth=4, c='b')
    for i in range(missileNum):
        ax3.plot(missileState[0:maxstep, i, 0], missileState[0:maxstep, i, 2], missileState[0:maxstep, i, 1],
                 linewidth=2, c='r')

    for i in range(DEFAULT_INTERCEPTOR_NUM):
        ax3.plot(interceptorState[0:maxstep, i, 0], interceptorState[0:maxstep, i, 2],
                 interceptorState[0:maxstep, i, 1],
                 linewidth=2, c='g')
    # for i in range(missileNum):
    #     ax1.plot(:, )
    arrayx = np.arange(0, t_p)
    ax2.plot(arrayx, Treat_value_p)
    # 定义标签
    ax1.set_xlabel('x正北方位移')
    ax1.set_ylabel('z正东方位移')
    ax1.set_zlabel('y垂直高度')

    ax3.set_xlabel('x正北方位移')
    ax3.set_ylabel('z正东方位移')
    ax3.set_zlabel('y垂直高度')

    arrayx = np.arange(0, t_com)
    ax4.plot(arrayx, Treat_value_com)

    plt.show()


# for _ in range(20):
#     t, statefromEnv, Treat_value, Env, state_copy, info_pre, escapeFlag_pre = predictResult(
#         './models/DQNmodels/DDQNmodels3_23/DDQN_episode100.pth')
#
#     showPredictResult(t, statefromEnv, Treat_value)
#     # aniPlot(t, statefromEnv, Treat_value)
#     ct, cstatefromEnv, cTreat_value, info_com, escapeFlag_com = ComparepredictResult(Env, state_copy)
#     compare_figure(t, statefromEnv, Treat_value, ct, cstatefromEnv, cTreat_value)

ct_list = []
escape_probability_preList = []
escape_probability_comList = []
p_interceptor_remainList = []
c_interceptor_remainList = []
p_interceptor_remain = DEFAULT_INTERCEPTOR_NUM
c_interceptor_remain = DEFAULT_INTERCEPTOR_NUM
# 测试逃避率
for _ in range(100):
    Test_epi = 100
    pre_num = 0
    com_num = 0
    for i in range(Test_epi):
        t, statefromEnv, Treat_value, Env, state_copy, p_interceptor_remain, escapeFlag_pre = predictResult(
            './models/DQNmodels/DDQNmodels3_23/DDQN_episode100.pth')
        if escapeFlag_pre != 0:
            pre_num += 1

        ct, cstatefromEnv, cTreat_value, c_interceptor_remain, escapeFlag_com = ComparepredictResult(Env, state_copy)
        if escapeFlag_com != 0:
            com_num += 1
        ct_list.append(ct)
        # print(ct)
    escape_probability_pre = pre_num / Test_epi * 100
    escape_probability_com = com_num / Test_epi * 100
    escape_probability_preList.append(escape_probability_pre)
    escape_probability_comList.append(escape_probability_com)
    p_interceptor_remainList.append(p_interceptor_remain)
    c_interceptor_remainList.append(c_interceptor_remain)
    ct_array = np.array(ct_list)
    ct = ct_array.mean()
    print('平均被打中的步数' + str(ct))
    print('强化学习下，飞机的逃离概率为：' + str(escape_probability_pre) + '%')
    print('非强化学习下，逃离概率为：' + str(escape_probability_com) + '%')
print(escape_probability_preList)
print(escape_probability_comList)
# 保存数据open函数
with open('./TestData/predict.txt', 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
    for probability in escape_probability_preList:
        f.write(str(probability) + '\n')  # 写入数据，文件保存在上面指定的目录，加\n为了换行更方便阅读

# 保存数据open函数
with open('./TestData/compare.txt', 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
    for probability in escape_probability_comList:
        f.write(str(probability) + '\n')  # 写入数据，文件保存在上面指定的目录，加\n为了换行更方便阅读

# 保存数据open函数
with open('./TestData/predict_interceptor_remain.txt', 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
    for remain in p_interceptor_remainList:
        f.write(str(remain) + '\n')  # 写入数据，文件保存在上面指定的目录，加\n为了换行更方便阅读

# 保存数据open函数
with open('./TestData/compare_interceptor_remain.txt', 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
    for remain in c_interceptor_remainList:
        f.write(str(remain) + '\n')  # 写入数据，文件保存在上面指定的目录，加\n为了换行更方便阅读

