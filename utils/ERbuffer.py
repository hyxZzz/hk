"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import numpy as np

from collections import deque


class MyMemoryBuffer(object):

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    # 增加经验，因为经验数组是存放在deque中的，deque是双端队列，
    # 我们的deque指定了大小，当deque满了之后再add元素，则会自动把队首的元素出队
    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    # continuous=True则表示连续取batch_size个经验
    def sample(self, batch_szie, continuous=False):

        # 选取的经验数目是否超过缓冲区内经验的数目
        if batch_szie > len(self.buffer):
            batch_szie = len(self.buffer)

        # 是否连续取经验
        if continuous:
            # random.randint(a，b) 返回[a，b]之间的任意整数
            rand = np.random.randint(0, len(self.buffer) - batch_szie)
            return [self.buffer[i] for i in range(rand, rand + batch_szie)]
        else:
            # numpy.random.choice(a, size=None, replace=True, p=None)
            # a 如果是数组则在数组中采样；a如果是整数，则从[0,a-1]这个序列中随机采样
            # size 如果是整数则表示采样的数量
            # replace为True可以重复采样；为false不会重复
            # p 是一个数组，表示a中每个元素采样的概率；为None则表示等概率采样
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_szie, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

# class ReplayMemory:
#     def __init__(self, config):
#         self.buffer = deque
#         # self.cnn_format = config.cnn_format
#         self.memory_size = config.memory_size
#         self.actions = np.empty(self.memory_size, dtype=np.integer)
#         self.rewards = np.empty(self.memory_size, dtype=np.integer)
#         # tot_num 为飞机数+导弹数+拦截弹数
#         # data_dim 为数据维数 x, y, z, Pitch, Heading, roll
#         self.state = np.empty((self.memory_size, config.tot_num, config.data_dim), dtype=np.float16)
#         self.terminals = np.empty(self.memory_size, dtype=np.bool)
#         self.history_length = config.history_length
#         self.dims = (config.tot_num, config.data_dim)
#         self.batch_size = config.batch_size
#         self.count = 0
#         self.current = 0
#
#         # pre-allocate prestates and poststates for minibatch
#         self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
#         self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
#
#     def add(self, state, reward, action, terminal):
#         assert state.shape == self.dims
#         # NB! screen is post-state, after action and reward
#         self.actions[self.current] = action
#         self.rewards[self.current] = reward
#         self.state[self.current, ...] = state
#         self.terminals[self.current] = terminal
#         self.count = max(self.count, self.current + 1)
#         self.current = (self.current + 1) % self.memory_size
#
#     def getState(self, index):
#         assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
#         # normalize index to expected range, allows negative indexes
#         index = index % self.count
#         # if is not in the beginning of matrix
#         if index >= self.history_length - 1:
#             # use faster slicing
#             return self.state[(index - (self.history_length - 1)):(index + 1), ...]
#         else:
#             # otherwise normalize indexes and use slower list based access
#             indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
#             return self.state[indexes, ...]
#
#     def sample(self):
#         # memory must include poststate, prestate and history
#         assert self.count > self.history_length
#         # sample random indexes
#         indexes = []
#         while len(indexes) < self.batch_size:
#             # find random index
#             while True:
#                 # sample one index (ignore states wraping over
#                 index = random.randint(self.history_length, self.count - 1)
#                 # if wraps over current pointer, then get new one
#                 if index >= self.current and index - self.history_length < self.current:
#                     continue
#                 # if wraps over episode end, then get new one
#                 # NB! poststate (last screen) can be terminal state!
#                 if self.terminals[(index - self.history_length):index].any():
#                     continue
#                 # otherwise use this index
#                 break
#
#             # NB! having index first is fastest in C-order matrices
#             self.prestates[len(indexes), ...] = self.getState(index - 1)
#             self.poststates[len(indexes), ...] = self.getState(index)
#             indexes.append(index)
#
#         actions = self.actions[indexes]
#         rewards = self.rewards[indexes]
#         terminals = self.terminals[indexes]
#
#         return self.prestates, actions, rewards, self.poststates, terminals
