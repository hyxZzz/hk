from typing import List

import numpy as np
import math as m
from gym import spaces

from flat_models.trajectory import Missiles, Aircraft, Interceptor
from flat_models.ThreatEvaluate import CalTreat
from utils.common import CalDistance
from Environment.ActionDepository import getActionDepository, getNewActionDepository
from Environment.reset_env import reset_para

act_num = 29  # 机动动作的多少
Gostep = 1  # 机动策略改变的频率

INTERCEPT_SUCCESS_DISTANCE = 20.0  # 拦截弹与导弹的命中阈值
MISSILE_HIT_DISTANCE = 20.0  # 来袭导弹命中飞机的阈值
SPARSE_REWARD_SCALE = 0.75  # 稀疏奖励的尺度
DANGER_DISTANCE = 3000 # 危险距离 用于奖励函数的非线性分段
LanchGap = 70 # 发射间隔
ERRACTIONSCALE = 2 # 惩罚加的系数 原先设为10
DANGERSCALE = 3 # 危险情况下 距离影响的系数 原先设为5
CURIOSITYSCALE = 0.5 #  好奇心加数 鼓励探索
class ManeuverEnv:
    """
                        导弹编号	    X位置	Y位置	Z位置	速度	    俯仰角	偏转角
                            飞机  	X位置	Y位置	Z位置	速度	    俯仰角	偏转角
                        拦截弹编号    X位置	Y位置	Z位置	速度	    俯仰角	偏转角
                        """

    def __init__(self, missileList: List[Missiles], aircraftList: Aircraft, planeSpeed=170,
                 missilesNum=3, spaceSize=5000, missilesSpeed=680, InterceptorNum=8, InterceptorSpeed=540):
        self.escapeFlag = -1
        self.action_space = spaces.Discrete(act_num)
        self.spaceSize = spaceSize
        self.Treat_t = 0
        self.action_dep = getNewActionDepository(act_num)
        """
            初始化参数：
            导弹个数    missileNum，
            导弹初速度   missilesSpeed，
            导弹对象列表  missileList，
            飞机对象列表  aircraftList，
            飞机初速度   planeSpeed，
            拦截弹数目   InterceptorNum，
            拦截弹初速度  InterceptorSpeed，
            空间大小    spaceSize，[Maxstep]
        """
        self.missileNum = missilesNum
        self.missileSpeed = missilesSpeed
        self.planeSpeed = planeSpeed
        self.interceptorNum = InterceptorNum
        self.interceptorSpeed = InterceptorSpeed
        self.interceptor_remain = InterceptorNum
        self.missileList = missileList
        self.aircraftList = aircraftList
        self.interceptorList = []
        self.position_scale = 25000.0
        self.At_1 = 0
        # 初始化拦截弹列表
        for _ in range(InterceptorNum):
            self.interceptorList.append(
                Interceptor([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                            self.aircraftList.V, self.aircraftList.Pitch, self.aircraftList.Heading))

        self.observation_planes = np.zeros((1, 6), dtype=np.float32)  # 只有一个飞机
        self.observation_missiles = np.zeros((self.missileNum, 6), dtype=np.float32)
        self.observation_interceptors = np.zeros((self.interceptorNum, 6), dtype=np.float32)
        self.StateShape = self.observation_planes.shape[0] + self.observation_missiles.shape[0] + \
                          self.observation_interceptors.shape[0]
        self.D0 = np.empty((missilesNum,), dtype=np.float32)
        for i in range(missilesNum):
            self.D0[i] = CalDistance([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                                     [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z])
        self.t = 0
        self.lanchTime = 0

    """飞机动作库，11种机动动作，利用三个控制量来控制。

            输入参数：

            ----------

            commandNum:动作指令序号

            Returns

            -----------

            控制量nx,nz,roll和俯仰角限制标识                        
        """

    def AirCraftActions(self, commandNum: int, interceptor_goal):
        # 动作库字典：[nx, ny, roll, Pitch]
        action = self.action_dep[commandNum][0:4]
        # interceptor_goal = int(self.action_dep[commandNum][4])
        LauchFlag = self.LanchPolicy(interceptor_goal)
        if LauchFlag:
            interceptor_goal = interceptor_goal
        else:
            interceptor_goal = -1

        return action, interceptor_goal

    def constraint_obs(self, act: List, speedFlag):

        #   取出机动动作
        # nx = aircraftList.nx
        # ny = aircraftList.ny
        # roll = aircraftList.roll
        nx, ny, roll, Pitch = act
        if speedFlag:
            nx = m.sin(self.aircraftList.Pitch)
        else:
            ny = m.cos(self.aircraftList.Pitch) / m.cos(self.aircraftList.roll)
            roll = self.aircraftList.roll
            Pitch = -1
        info = 'Go on Combating...'
        escapeFlag = -1  # 是否逃离标志，0未逃离，1机动逃离，2拦截完毕

        # 检查是否到了逃逸空间，如果比最大步数多，则机动逃逸
        if self.t >= self.spaceSize:
            escapeFlag = 1
            self.escapeFlag = escapeFlag
            info = 'Maneuver Success'
            return np.concatenate(
                (self.observation_planes, self.observation_missiles, self.observation_interceptors)), escapeFlag, info

        tx, ty, tz = self.aircraftList.AircraftPostition(None, nx, ny, roll, Pitch)  # 目前只有一个飞机
        self.observation_planes[0] = np.array(
            [tx, ty, tz, self.aircraftList.Pitch, self.aircraftList.Heading, self.aircraftList.roll])
        x_a, y_a, z_a, v_a, Pitch_a, Heading_a = self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z, self.aircraftList.V, \
                                                 self.aircraftList.Pitch, self.aircraftList.Heading  # 获取飞机位置等信息
        ac_list = [x_a, y_a, z_a]  # 飞机位置列表，用于导弹传参

        # 检查来袭导弹是否拦截完毕，全部导弹不在飞行则逃逸成功
        does_inter_over = 0
        for j in range(len(self.missileList)):
            if not self.missileList[j].attacking:
                does_inter_over += 1
        if does_inter_over == len(self.missileList):
            escapeFlag = 2
            self.escapeFlag = escapeFlag
            info = 'Intercept Success'
            return np.concatenate(
                (self.observation_planes, self.observation_missiles,
                 self.observation_interceptors)), self.escapeFlag, info

        # 根据飞机信息更新导弹
        for i in range(len(self.missileList)):
            if self.missileList[i].attacking:
                # 获取其中一个导弹的x,y,z坐标
                mx, my, mz = self.missileList[i].MissilePosition(ac_list, v_a, Pitch_a, Heading_a)
                # 添加到导弹状态数组中
                self.observation_missiles[i] = np.array(
                    [mx, my, mz, self.missileList[i].Pitch, self.missileList[i].Heading, 0])
                # 计算导弹与飞机的距离
                dist = m.sqrt((x_a - mx) ** 2 + (y_a - my) ** 2 + (z_a - mz) ** 2)

                """
                    进行拦截计算：
                    对每一个拦截弹来说，其锁定的目标来袭导弹是一定的，根据锁定目标的索引来计算彼此的距离，
                    如果小于阈值，则拦截成功。置被拦截目标导弹和拦截弹的飞行状态为False。
                """

                # 拦截弹如果没有发射，随飞机一起飞行
                for j in range(len(self.interceptorList)):
                    if self.interceptorList[j].T_i == -1:
                        self.interceptorList[j].sync_with_aircraft(
                            [x_a, y_a, z_a], Pitch_a, Heading_a, self.aircraftList.V
                        )
                        self.observation_interceptors[j] = np.array([x_a, y_a, z_a, Pitch_a, Heading_a, 0])
                    ix, iy, iz = self.interceptorList[j].X_i, self.interceptorList[j].Y_i, self.interceptorList[j].Z_i

                    # 拦截弹发射成功，根据起拦截目标更新位置
                    if self.interceptorList[j].T_i == i and self.interceptorList[j].attacking != 1:
                        ix, iy, iz = self.interceptorList[j].InterceptorPosition([mx, my, mz], self.missileList[i].V,
                                                                                 self.missileList[i].Pitch,
                                                                                 self.missileList[i].Heading)
                        self.observation_interceptors[j] = np.array(
                            [ix, iy, iz, self.interceptorList[j].Pitch_i, self.interceptorList[j].Heading_i, 0])
                        dist_im = m.sqrt((ix - mx) ** 2 + (iy - my) ** 2 + (iz - mz) ** 2)
                        # 拦截弹拦截成功
                        if dist_im < INTERCEPT_SUCCESS_DISTANCE:
                            self.missileList[i].attacking = False  # 拦截使来袭导弹失效
                            self.interceptorList[j].attacking = 1  # 拦截导弹牺牲

                # 被i导弹打中
                if dist < MISSILE_HIT_DISTANCE:
                    escapeFlag = 0  # 被导弹击中，未逃离
                    self.escapeFlag = escapeFlag
                    info = 'Hit on! Escape Fail!!'
                    return np.concatenate((self.observation_planes, self.observation_missiles,
                                           self.observation_interceptors)), self.escapeFlag, info
        self.escapeFlag = escapeFlag
        return np.concatenate(
            (self.observation_planes, self.observation_missiles, self.observation_interceptors)), self.escapeFlag, info

    def generate_obs(self, act: List):

        #   取出机动动作
        nx, ny, roll, Pitch = act
        #
        # if self.aircraftList.V > 3.4 * 1.2:
        #     nx = 0

        escapeFlag = -1  # 是否逃离标志，0未逃离，1机动逃离，2拦截完毕
        info = 'Go on Combating...'

        # # 检查导弹是否触底，触底则失效
        # for i in range(len(self.missileList)):
        #     if self.missileList[i].Y <= 0:
        #         self.missileList[i].attacking = False

        # 检查是否到了逃逸空间，如果比最大步数多，则机动逃逸
        if self.t >= self.spaceSize:
            escapeFlag = 1
            self.escapeFlag = escapeFlag
            info = 'Maneuver Success'
            return np.concatenate(
                (self.observation_planes, self.observation_missiles, self.observation_interceptors)), escapeFlag, info

        tx, ty, tz = self.aircraftList.AircraftPostition(None, nx, ny, roll, Pitch)  # 目前只有一个飞机
        self.observation_planes[0] = np.array(
            [tx, ty, tz, self.aircraftList.Pitch, self.aircraftList.Heading, self.aircraftList.roll])
        x_a, y_a, z_a, v_a, Pitch_a, Heading_a = self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z, self.aircraftList.V, \
                                                 self.aircraftList.Pitch, self.aircraftList.Heading  # 获取飞机位置等信息
        ac_list = [x_a, y_a, z_a]  # 飞机位置列表，用于导弹传参

        # 检查来袭导弹是否拦截完毕，全部导弹不在飞行则逃逸成功
        does_inter_over = 0
        for j in range(len(self.missileList)):
            if not self.missileList[j].attacking:
                does_inter_over += 1
        if does_inter_over == len(self.missileList):
            escapeFlag = 2
            self.escapeFlag = escapeFlag
            info = 'Intercept Success'
            return np.concatenate(
                (self.observation_planes, self.observation_missiles, self.observation_interceptors)), escapeFlag, info

        # 根据飞机信息更新导弹
        for i in range(len(self.missileList)):
            if self.missileList[i].attacking:
                # 获取其中一个导弹的x,y,z坐标
                mx, my, mz = self.missileList[i].MissilePosition(ac_list, v_a, Pitch_a, Heading_a)
                # 添加到导弹状态数组中
                self.observation_missiles[i] = np.array(
                    [mx, my, mz, self.missileList[i].Pitch, self.missileList[i].Heading, 0])
                # 计算导弹与飞机的距离
                dist = m.sqrt((x_a - mx) ** 2 + (y_a - my) ** 2 + (z_a - mz) ** 2)

                """
                    进行拦截计算：
                    对每一个拦截弹来说，其锁定的目标来袭导弹是一定的，根据锁定目标的索引来计算彼此的距离，
                    如果小于阈值，则拦截成功。置被拦截目标导弹和拦截弹的飞行状态为False。
                """

                # 拦截弹如果没有发射，随飞机一起飞行
                for j in range(len(self.interceptorList)):
                    if self.interceptorList[j].T_i == -1:
                        self.interceptorList[j].X_i = x_a
                        self.interceptorList[j].Y_i = y_a
                        self.interceptorList[j].Z_i = z_a
                        self.interceptorList[j].Pitch_i = Pitch_a
                        self.interceptorList[j].Heading_i = Heading_a
                        self.observation_interceptors[j] = np.array([x_a, y_a, z_a, Pitch_a, Heading_a, 0])
                    ix, iy, iz = self.interceptorList[j].X_i, self.interceptorList[j].Y_i, self.interceptorList[j].Z_i

                    # 拦截弹发射成功，根据起拦截目标更新位置
                    if self.interceptorList[j].T_i == i and self.interceptorList[j].attacking != 1:
                        ix, iy, iz = self.interceptorList[j].InterceptorPosition([mx, my, mz], self.missileList[i].V,
                                                                                 self.missileList[i].Pitch,
                                                                                 self.missileList[i].Heading)
                        self.observation_interceptors[j] = np.array(
                            [ix, iy, iz, self.interceptorList[j].Pitch_i, self.interceptorList[j].Heading_i, 0])
                        dist_im = m.sqrt((ix - mx) ** 2 + (iy - my) ** 2 + (iz - mz) ** 2)
                        # 拦截弹拦截成功
                        if dist_im < INTERCEPT_SUCCESS_DISTANCE:
                            self.missileList[i].attacking = False  # 拦截使来袭导弹失效
                            self.interceptorList[j].attacking = 1  # 拦截导弹牺牲

                # 被i导弹打中
                if dist < MISSILE_HIT_DISTANCE:
                    escapeFlag = 0  # 被导弹击中，未逃离
                    self.escapeFlag = escapeFlag
                    info = 'Hit on! Escape Fail!!'
                    return np.concatenate((self.observation_planes, self.observation_missiles,
                                           self.observation_interceptors)), escapeFlag, info
        self.escapeFlag = escapeFlag
        return np.concatenate(
            (self.observation_planes, self.observation_missiles, self.observation_interceptors)), self.escapeFlag, info

    """
            动作生成函数，输入机动策略序号与拦截目标导弹序号

            ---------------
            Returns

            机动动作列表[nx, ny, roll, Pitch]，并改变了类内interceptors的attacking标志和导引目标T_i
        """

    def _gen_action(self, c, goal=None):
        if goal == None:
            return

        return self.AirCraftActions(c, goal)

    def _get_obs(self):
        obs = np.concatenate((self.observation_planes, self.observation_missiles, self.observation_interceptors))
        return obs

    def _get_actSpace(self):
        a = self.action_space.n
        return a

    def _get_stateSpace(self):
        s = self._get_obs()
        s = s.flatten()
        return s.shape

    # 获取拉直的向量
    def _get_flattenstate(self, s):
        s = s.flatten()
        return s

    # 将拉直的向量重构为环境内计算的向量格式
    def _resize_flattenState(self, flattenState):
        obs = np.resize(flattenState, (self.StateShape, 6))
        return obs

    def render(self):
        pass


    """
        拦截弹锁定目标的上限
    """
    def LockConstraint(self, intceptor_goal):
        if intceptor_goal < 0:
            return False

        locked_Num = 0  # 打击第i个导弹的拦截弹个数
        for j in range(len(self.interceptorList)):
            Att_Num = self.interceptorList[j].T_i  # 拦截弹的锁定目标
            # 当拦截弹锁定目标为第i个导弹时，给变量+1
            if Att_Num == intceptor_goal:
                locked_Num += 1

        active_missiles = self.getRemainMissileNum()
        if active_missiles <= 0:
            return False

        base_limit = max(1, m.ceil(self.interceptorNum / max(1, self.missileNum)))
        focus_bonus = max(0, self.missileNum - active_missiles)
        max_lock = min(self.interceptorNum, base_limit + focus_bonus)

        if locked_Num >= max_lock:
            return False
        return True

    """ 
        飞机高度奖励：
        输入：飞机高度
        输出：奖励值：[0,1]
    
        """

    def heightReward(self, h):

        safe_min = 8000
        safe_max = 12000
        tolerance = 1000
        hard_min = safe_min - tolerance
        hard_max = safe_max + tolerance

        if h < hard_min or h > hard_max:
            self.escapeFlag = 0  # 撞地结束或高度过高失速结束
            return -1.5

        if h < safe_min:
            ratio = (h - hard_min) / (safe_min - hard_min)
            return -1.0 + 2.0 * ratio
        if h > safe_max:
            ratio = (hard_max - h) / (hard_max - safe_max)
            return -1.0 + 2.0 * ratio

        center = (safe_min + safe_max) / 2.0
        span = (safe_max - safe_min) / 2.0
        offset = (h - center) / span
        return 1.0 - offset ** 2

    """
        距离奖励：
        输入：导弹向量
        输出：奖励值【-1.608， 1】
    """
    def distanceReward(self, missileState, planeState):

        rd = 0
        rdMin = 10e8
        engagement_range = 25000.0
        for i in range(missileState.shape[0]):
            if self.missileList[i].attacking:
                D = abs(np.linalg.norm((missileState[i] - planeState)))
                D = max(D, 1.0)
                scale = engagement_range / max(DANGER_DISTANCE, 1.0)
                if scale <= 1:
                    rd = -1.5
                else:
                    rd = m.log(D / DANGER_DISTANCE) / m.log(scale)
                rd = max(min(rd, 1.0), -1.5)
        # 计算奖励最小的导弹
            if rd < rdMin:
                rdMin = rd
        return rdMin




    """
        成型奖励：输入全局状态，输出成型奖励值

        ---------------
        Returns

        成型奖励值
    """

    def commonReward(self, state, interceptor_goal):
        planeState = state[0]
        missileState = state[1:self.missileNum + 1]
        rd = 0
        # 距离最近的导弹距离 和其本身的索引
        dist, missile_index = self.getClosetMissileDist()
        # 危险标志 此标志为TRUE时 将距离奖励变大 拦截奖励变大 拦截惩罚变小
        dangerFlag = False
        if dist <= DANGER_DISTANCE:
            dangerFlag = True

        """
            飞机高度奖励，防止撞地
        """

        C1 = 1.2  # 飞机高度奖励的系数
        h = planeState[1]
        rh = self.heightReward(h)
        rd += C1 * rh

        """
            距离大时发射拦截弹的惩罚
            [-3, 0]
            """
        for i in range(len(self.interceptorList)):
            target_index = self.interceptorList[i].T_i
            # 发射了拦截弹的目标
            if target_index != -1:
                # 与发射目标的原始距离
                D0 = abs(self.D0[target_index])
                # 与发射目标的实时距离
                D = abs(np.linalg.norm((missileState[target_index] - planeState)))
                # 如果距离大于一半 惩罚
                if D > D0 / 2:
                    rd -= 0.4
                    # 如果当前动作是发射远距离的导弹 这一步视为错误
                    if interceptor_goal == target_index:
                        rd -= ERRACTIONSCALE





        """ 
                    导弹与飞机相对距离奖励
                    [0,1]
                """
        C2 = 1
        rd_o = 0
        for i in range(missileState.shape[0]):
            D0 = abs(self.D0[i])
            D = abs(np.linalg.norm((missileState[i] - planeState)))
            if D0 < 1e-8:
                rd = - 1
                self.escapeFlag = 0
            else:
                rd_o += C2 * (D / D0)
        rd_o = rd_o / missileState.shape[0]
        rd += C2 * rd_o


        """
                导弹与飞机实时距离标量奖励
                正常下：【-1.608， 1】
                距离近下：【-8,0】
                """

        r_Dd = self.distanceReward(missileState, planeState)
        if dangerFlag:
            C2 = DANGERSCALE
        rd += C2 * r_Dd


        """
                   剩余拦截弹奖励rm
                   正常下：0
                   危机下：【-12,0】
               """
        rm = 0
        if dangerFlag:
            active_missiles = max(1, self.getRemainMissileNum())
            idle_interceptors = sum(1 for itr in self.interceptorList if itr.attacking == -1)
            if self.LockConstraint(missile_index) and idle_interceptors > 0:
                focus_penalty = idle_interceptors / active_missiles
                if interceptor_goal not in (-1, missile_index):
                    focus_penalty += 1 / active_missiles
                rm = -DANGERSCALE * focus_penalty
        rd += rm

        """
                    威胁度奖励
                    【-1,0.25】
                """
        C5 = 1
        Treat = 0
        plane_position = planeState
        v_p = self.aircraftList.V
        for i in range(missileState.shape[0]):
            if self.missileList[i].attacking:
                missile_position = missileState[i]
                v_m = self.missileList[i].V
                if Treat < CalTreat(plane_position, missile_position, v_p, v_m):
                    Treat = CalTreat(plane_position, missile_position, v_p, v_m)
                    if Treat < 0:
                        assert 'TreatCompute Wrong!!'
        if Treat <= self.Treat_t:
            self.Treat_t = Treat
            rd += 0.25 * C5 * Treat
        else:
            self.Treat_t = Treat
            rd += - 1 * C5 * Treat    # 负的要狠
        if rd < -1:
            assert 'TreatCompute Wrong!!'

        """
            拦截弹发射的奖励：
            当载机和来袭弹的距离相差较小时，发射拦截弹加奖励
            当载机和来袭弹的角度合适是，发射拦截弹加奖励
            当二者都合适时，发射拦截弹加大的奖励
        """
        rl = 0

        # 当有一个导弹距离近时
        if dangerFlag:
            engaged_on_threat = 0
            wrong_lock = 0
            for itr in self.interceptorList:
                if itr.attacking != -1 and itr.T_i == missile_index:
                    engaged_on_threat += 1
                elif itr.attacking != -1 and itr.T_i not in (-1, missile_index):
                    wrong_lock += 1

            rl += 1.8 * engaged_on_threat
            can_focus = self.LockConstraint(missile_index)
            if can_focus:
                rl -= ERRACTIONSCALE * wrong_lock
                if interceptor_goal == missile_index:
                    rl += 0.6
                elif interceptor_goal not in (-1, missile_index):
                    rl -= ERRACTIONSCALE * 0.5
        rd += rl

        """
            引导拦截弹拦截奖励
            拦截弹对锁定目标的威胁度越高，奖励越大
            正常情况下：【-0.5，0.5】
            危急情况下：【-2.5，2.5】
        """
        # C6 = 8
        C6 = 1
        ri = 0
        lock_num = 0
        for itr in self.interceptorList:

            # 拦截弹已发射且没有牺牲
            if itr.attacking != -1 and itr.attacking != 1:
                # 锁的当前最近的
                if itr.T_i == missile_index:
                    ri += CalTreat(missileState[itr.T_i], [itr.X_i, itr.Y_i, itr.Z_i, itr.Pitch_i, itr.Heading_i, 0], self.missileList[itr.T_i].V, itr.V_i)

                # 不是当前最近的
                else:
                    ri -= CalTreat(missileState[itr.T_i], [itr.X_i, itr.Y_i, itr.Z_i, itr.Pitch_i, itr.Heading_i, 0], self.missileList[itr.T_i].V, itr.V_i)
                lock_num += 1

        # 防止没有拦截弹发射时导致的除法错误
        if lock_num != 0:
            if dangerFlag:
                C6 = DANGERSCALE
            rd += C6 * ri / lock_num

        # if rd > 51 or rd < -20:
        #     print(rd)
        return rd

    """
        机动不合规惩罚
    """

    def Punish(self):
        rp = 1
        # rp = 12
        return -rp

    """
            稀疏奖励
        """

    def SparseReward(self):
        rd = 0
        C4 = SPARSE_REWARD_SCALE

        # C4 = 100
        if self.escapeFlag == -1:
            dist, _ = self.getClosetMissileDist()
            danger_multiplier = 1.0 if dist <= DANGER_DISTANCE else 0.5
            rd = - 0.06 * C4 * danger_multiplier * self.getRemainMissileNum() # 每一颗存在的导弹都要有惩罚
        elif self.escapeFlag == 0:
            rd = - C4
        elif self.escapeFlag == 1:
            rd = C4
        elif self.escapeFlag == 2:
            rd = C4

        return rd

    """
            奖励函数：输入当前状态，输出奖励值

            ----------
            Returns

            奖励值Reward
        """

    def rewards(self, state, interceptor_goal):
        comReward = self.commonReward(state, interceptor_goal)
        sparseReward = self.SparseReward()
        reward = comReward + sparseReward
        # if reward < -1000 or reward > 1000:
        #     print(reward)
        return reward

    '''获取剩余导弹个数'''
    def getRemainMissileNum(self):
        count = 0
        for missile in self.missileList:
            if missile.attacking:
                count += 1
        return count

    '''获取最近的正在来袭的导弹距离及索引'''
    def getClosetMissileDist(self):
        distMin = 10e8
        planePos = [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z]
        index = 0
        for i in range(len(self.missileList)):
            if self.missileList[i].attacking:
                targetPos = [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z]
                dist = CalDistance(planePos, targetPos)
                if dist < distMin:
                    distMin = dist
                    index = i
        return distMin, index


    """获取动作信息 将动作索引输入 输出三控制量的具体信息和拦截目标"""

    def getActionData(self, action, interceptor_goal=None):
        # 距离最近的导弹距离 和其本身的索引
        dist, missile_index = self.getClosetMissileDist()
        # 危险标志 此标志为TRUE时 将距离奖励变大 拦截奖励变大 拦截惩罚变小
        dangerFlag = False
        if dist <= DANGER_DISTANCE:
            dangerFlag = True
        if dangerFlag:
            goal = missile_index
        else:
            goal = -1

        if interceptor_goal == None:
            [nx, ny, roll, pitch_constraint], interceptor_goal = self._gen_action(action, goal)
        else:
            [nx, ny, roll, pitch_constraint], interceptor_goal = self._gen_action(action, interceptor_goal)
        return [nx, ny, roll, pitch_constraint], interceptor_goal

    def step(self, action):
        state = self._get_obs()
        escapeFlag = -1
        info = ''
        # 第一次进入环境时 将action给at-1
        if self.t == 0:
            self.At_1 = action

        [nx, ny, roll, pitch_constraint], interceptor_goal = self.getActionData(action)
        # 判断动作是否合理
        # 速度合理标识
        speed_flag = True
        lauch_flag = not (self.interceptor_remain == 0 and interceptor_goal != -1)
        if self.aircraftList.action_constraint(pitch_constraint) and self.aircraftList.speed_constraint(nx) and lauch_flag:
            for _ in range(Gostep):
                state, escapeFlag, info = self.generate_obs([nx, ny, roll, pitch_constraint])
            reward = self.rewards(state, interceptor_goal)

            # 好奇心机制
            if action != self.At_1:
                reward += CURIOSITYSCALE


            # # 强制让网络选不同的动作试试
            # if action == self.At_3:
            #     reward = -10000
            # reward = 0
            # if action == 50:
            #     reward +=5000
            self.t += 1
            self.At_1 = action

        else:
            info = 'Constraint!!'
            for _ in range(Gostep):
                state, escapeFlag, info = self.constraint_obs([nx, ny, roll, pitch_constraint], speed_flag)

            reward = self.rewards(state, interceptor_goal)
            reward = self.Punish() + reward
            # 好奇心机制
            if action != self.At_1:
                reward += CURIOSITYSCALE
                # # 强制让网络选不同的动作试试
            # if action == self.At_3:
            #     reward = -10000
            # reward = 0
            # if action == 50:
            #     reward +=5000
            self.t += 1
            self.At_1 = action



                    # state无量纲化
        state = self.normalizeState(state, reverse=False)
        # state[:, 0:3] = state[:, 0:3] / 20000
        # state[:, 3:6] = state[:, 3:6] / m.pi
        state = self._get_flattenstate(state)  # 获取拉直的向量

        state = self._genNewState_()
        return state, reward, self.escapeFlag, info


    """对比实验的策略"""

    def compareTest(self, action):
        state = self._get_obs()
        escapeFlag = -1
        info = ''
        # 第一次进入环境时 将action给at-1
        if self.t == 0:
            self.At_1 = action

        if self.t % 2 == 0:
            interceptor_goal = 0
        else:
            interceptor_goal = 1

        [nx, ny, roll, pitch_constraint], interceptor_goal = self.getActionData(action, interceptor_goal)
        # 判断动作是否合理
        # 速度合理标识
        speed_flag = True
        lauch_flag = not (self.interceptor_remain == 0 and interceptor_goal != -1)
        if self.aircraftList.action_constraint(pitch_constraint) and self.aircraftList.speed_constraint(nx) and lauch_flag:
            for _ in range(Gostep):
                state, escapeFlag, info = self.generate_obs([nx, ny, roll, pitch_constraint])
            reward = self.rewards(state, interceptor_goal)

            # 好奇心机制
            if action != self.At_1:
                reward += CURIOSITYSCALE


            # # 强制让网络选不同的动作试试
            # if action == self.At_3:
            #     reward = -10000
            # reward = 0
            # if action == 50:
            #     reward +=5000
            self.t += 1
            self.At_1 = action

        else:
            info = 'Constraint!!'
            for _ in range(Gostep):
                state, escapeFlag, info = self.constraint_obs([nx, ny, roll, pitch_constraint], speed_flag)

            reward = self.rewards(state, interceptor_goal)
            reward = self.Punish() + reward
            # 好奇心机制
            if action != self.At_1:
                reward += CURIOSITYSCALE
                # # 强制让网络选不同的动作试试
            # if action == self.At_3:
            #     reward = -10000
            # reward = 0
            # if action == 50:
            #     reward +=5000
            self.t += 1
            self.At_1 = action



                    # state无量纲化
        state = self.normalizeState(state, reverse=False)
        # state[:, 0:3] = state[:, 0:3] / 20000
        # state[:, 3:6] = state[:, 3:6] / m.pi
        state = self._get_flattenstate(state)  # 获取拉直的向量

        state = self._genNewState_()
        return state, reward, self.escapeFlag, info

    def reset(self):
        missilesNum = self.missileNum
        self.Treat_t = 0
        self.interceptor_remain = self.interceptorNum
        self.escapeFlag = -1
        info = 'Go on Combating...'
        missileList, aircraftList, planeSpeed, missiles_num, spaceSize, missilesSpeed = reset_para(
            num_missiles=missilesNum)
        self.missileNum = missilesNum
        self.missileSpeed = missilesSpeed
        self.planeSpeed = planeSpeed
        self.missileList = missileList
        self.aircraftList = aircraftList
        self.interceptorList = []
        self.position_scale = 25000.0
        # 初始化拦截弹列表
        for i in range(self.interceptorNum):
            self.interceptorList.append(Interceptor([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                                                    self.aircraftList.V, self.aircraftList.Pitch,
                                                    self.aircraftList.Heading))

        self.observation_planes = np.zeros((1, 6), dtype=np.float32)  # 只有一个飞机

        self.observation_planes[0] = np.array(
            [aircraftList.X, aircraftList.Y, aircraftList.Z, aircraftList.Pitch, aircraftList.Heading,
             aircraftList.roll], dtype=np.float32)

        self.observation_missiles = np.zeros((self.missileNum, 6), dtype=np.float32)

        for i in range(len(missileList)):
            self.observation_missiles[i] = np.array(
                [missileList[i].X, missileList[i].Y, missileList[i].Z, missileList[i].Pitch, missileList[i].Heading, 0],
                dtype=np.float32)
        self.observation_interceptors = np.zeros((self.interceptorNum, 6), dtype=np.float32)

        for i in range(len(self.interceptorList)):
            self.observation_interceptors[i] = np.array(
                [aircraftList.X, aircraftList.Y, aircraftList.Z, aircraftList.Pitch, aircraftList.Heading, 0],
                dtype=np.float32)

        self.StateShape = self.observation_planes.shape[0] + self.observation_missiles.shape[0] + \
                          self.observation_interceptors.shape[0]
        self.D0 = np.empty((missilesNum,), dtype=np.float32)

        for i in range(missilesNum):
            self.D0[i] = CalDistance([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                                     [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z])
        self.t = 0
        state = np.concatenate((self.observation_planes, self.observation_missiles, self.observation_interceptors))

        # 无量纲化
        state = self.normalizeState(state, reverse=False)
        state = self._get_flattenstate(state)
        state = self._genNewState_()
        return state, self.escapeFlag, info

    # 无量纲化
    '''reverse为TRUE时 量钢化'''
    def normalizeState(self, state, reverse=False):
        if reverse:
            state[:, 0:3] = state[:, 0:3] * self.position_scale
            state[:, 3:6] = state[:, 3:6] * m.pi
        else:
            state[:, 0:3] = state[:, 0:3] / self.position_scale
            state[:, 3:6] = state[:, 3:6] / m.pi
        return state


    """新特征向量的大小"""
    def _getNewStateSpace(self):
        state = np.zeros((self.missileNum * self.interceptorNum + 2 * self.missileNum + self.interceptorNum,),
                         dtype=np.float32)
        return state.shape


    """发射策略"""
    def LanchPolicy(self, interceptor_goal):
        # 两次发射间隔步数
        if abs(self.t - self.lanchTime) >= LanchGap:
            # 目标没锁满
                    if self.LockConstraint(interceptor_goal):
                        for i in range(self.interceptorNum):
                            # 待发射的导弹
                            if self.interceptorList[i].T_i == -1:
                                interceptor = self.interceptorList[i]
                                interceptor.sync_with_aircraft(
                                    [self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z],
                                    self.aircraftList.Pitch,
                                    self.aircraftList.Heading,
                                    self.aircraftList.V,
                                )
                                launch_speed = max(self.aircraftList.V, self.interceptorSpeed)
                                interceptor.begin_pursuit(interceptor_goal, launch_speed)
                                self.interceptor_remain -= 1
                                self.lanchTime = self.t
                                return True

        else:
            return False



    """新特征状态"""
    def _genNewState_(self):

        state = np.zeros((self.missileNum * self.interceptorNum + 2 * self.missileNum + self.interceptorNum,), dtype=np.float32)
        missileDist = np.zeros((self.missileNum, ), dtype=np.float32)
        interceptorDist = np.zeros((self.missileNum * self.interceptorNum, ), dtype=np.float32)
        missileStatus = np.zeros((self.missileNum, ), dtype=np.float32)
        interceptorStatus = np.zeros((self.interceptorNum, ), dtype=np.float32)

        for i in range(self.missileNum):
            missileDist[i] = CalDistance([self.aircraftList.X, self.aircraftList.Y, self.aircraftList.Z], [self.missileList[i].X, self.missileList[i].Y, self.missileList[i].Z]) / 10000

        t = 0
        for i in range(self.interceptorNum):
            for j in range(self.missileNum):
                interceptorDist[t] = CalDistance([self.interceptorList[i].X_i, self.interceptorList[i].Y_i, self.interceptorList[i].Z_i], [self.missileList[j].X, self.missileList[j].Y, self.missileList[j].Z]) / 10000
                t += 1

        for i in range(self.missileNum):
            if self.missileList[i].attacking:
                missileStatus[i] = 1
            else:
                missileStatus[i] = -1

        for i in range(self.interceptorNum):
            if self.interceptorList[i].attacking == -1:
                interceptorStatus[i] = -1
            elif self.interceptorList[i].attacking == 1:
                interceptorStatus[i] = 1

            else:
                interceptorStatus[i] = 0

        state = np.concatenate((missileDist, interceptorDist, missileStatus, interceptorStatus))
        return state
