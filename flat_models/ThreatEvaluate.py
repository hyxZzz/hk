import numpy as np
import math as m

from utils.common import CalDistance, ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles
from flat_models.AHP import AHP

"""
    1、角度威胁模型
        初始化参数：

    ----------

    Heading_max: 0 < double类型 < pi/2
           来袭导弹的最大水平攻击角度

    Pitch_max:0 < double类型 < pi/2
              来袭导弹的最大俯仰攻击角度

    omega:0 < double类型 < 1
          水平角度的加权值，越大说明水平角度对角度威胁的影响越大


    Returns

    -----------

    AngleThreat对象
"""


class AngleThreat:
    def __init__(self, Heading_max, Pitch_max, omega):
        self.Heading_max = Heading_max
        self.Pitch_max = Pitch_max
        self.omega = omega

    """
        角度威胁值计算函数

            参数：

            ----------

            Heading:double类型 
                    我方战斗机相对于敌方导弹单位的水平角度

            Pitch:double类型
                  我方导弹相对于敌方反导单位的俯仰角度


            Returns

            -----------

            角度威胁值：0 <= Ta <= 1                       
    """

    def CalTa(self, Heading, Pitch):
        Ta = m.exp(
            max(1 - self.omega * (abs(Heading) / self.Heading_max) - (1 - self.omega) * (abs(Pitch) / self.Pitch_max),
                0))
        return Ta/m.e


"""
    2、距离威胁模型
        初始化参数：

    ----------

    dist_max: 0 < double类型 
           敌方导弹的最大攻击距离

    Returns

    -----------

    DistanceThreat对象
"""


class DistanceThreat:
    def __init__(self, kd, sigma, dist_max):
        self.dist_max = dist_max
        self.kd = kd
        self.sigma = sigma

    """
        距离威胁值计算函数

            参数：

            ----------

            dist:double类型 
                    我方战斗机与敌方来袭导弹之间的距离


            Returns

            -----------

            距离威胁值：0 <= Td <= 1                       
    """

    def CalTd(self, dist):
        # Td = m.exp(max(1 - (abs(dist) / self.dist_max), 0))
        # return Td/m.e
        Td = self.kd * (1 / (abs(dist) + self.sigma) - 1 / self.dist_max) ** 2 * (abs(dist) + self.sigma) ** 2
        return Td


"""
    3、速度威胁模型
        初始化参数：

    ----------

    ve: 0 < double类型 
        敌方导弹的预估速度
    
    v: 0 < double类型
      我方导弹的预估速度
    
    Returns

    -----------

    SpeedThreat对象
"""


# 3、速度威胁模型
class SpeedThreat:
    def __init__(self, ve, v):
        self.ve = ve
        self.v = v

    """
        速度威胁值计算函数

            参数：

            ----------
            无

            Returns

            -----------

            速度威胁值：0 <= Ts <= 1                       
    """

    def CalTs(self):
        Adv_speed = self.ve / self.v
        if Adv_speed <= 0.6:
            return 0.1
        if Adv_speed > 1.5:
            return 1
        else:
            return Adv_speed - 0.5

    def setV(self, ve, v):
        self.ve = ve
        self.v = v


# 4、总体受威胁态势建模
class TreatEva:
    def __init__(self, at, dt, st, rola, rold, rols):
        self.AT = at
        self.DT = dt
        self.ST = st
        self.rola = rola
        self.rold = rold
        self.rols = rols

    def CalT(self):
        T = self.AT.CalTa * self.rola + self.DT.CalTd * self.rold + self.ST.CalTs * self.rols
        return T


"""
    威胁评估的超参数
"""

fi_max, theta_max = m.pi * 0.5, m.pi * 0.5
omega = 0.2
dist_max = 10000
angle_threat = AngleThreat(fi_max, theta_max, omega)
distance_threat = DistanceThreat(kd=1, sigma=1e-8, dist_max=dist_max)
speed_threat = SpeedThreat(100, 200)
# 输入Angle、distance、speed威胁评估矩阵
#           角威胁 速度威胁 距离威胁
# 角威胁
# 速度威胁
# 距离威胁
criteria = np.array([[1, 1 / 2, 1 / 8],
                     [2, 1, 1 / 6],
                     [8, 6, 1]])
a = AHP(criteria).run("calculate_mean_weights")


# 根据飞机和导弹的位置计算威胁度（传入对象）

def CalTreat_obj(plane: Aircraft, missile: Missiles):
    # 计算角度威胁
    plane_position = [plane.X, plane.Y, plane.Z]
    missile_position = [missile.X, missile.Y, missile.Z]
    V_m = missile.V
    V_p = plane.V
    angle_Heading = ComputeHeading(plane_position, missile_position)
    angle_Pitch = ComputePitch(plane_position, missile_position)
    Treat_a = angle_threat.CalTa(angle_Heading, angle_Pitch)

    # 计算距离威胁
    dist = CalDistance(plane_position, missile_position)
    Treat_d = distance_threat.CalTd(dist)

    # 计算速度威胁
    speed_threat.setV(V_m, V_p)
    Treat_v = speed_threat.CalTs()

    # 计算加权后的威胁度
    Treat_tot = np.array([Treat_a, Treat_v, Treat_d])
    Treat = np.dot(Treat_tot, a)

    return Treat


# 根据飞机和导弹的位置计算威胁度

def CalTreat(plane_position, missile_position, V_p, V_m):
    # 计算角度威胁
    angle_Heading = ComputeHeading(plane_position, missile_position)
    angle_Pitch = ComputePitch(plane_position, missile_position)
    Treat_a = angle_threat.CalTa(angle_Heading, angle_Pitch)

    # 计算距离威胁
    dist = CalDistance(plane_position, missile_position)
    Treat_d = distance_threat.CalTd(dist)

    # 计算速度威胁
    speed_threat.setV(V_m, V_p)
    Treat_v = speed_threat.CalTs()

    # 计算加权后的威胁度
    Treat_tot = np.array([Treat_a, Treat_v, Treat_d])
    Treat = np.dot(Treat_tot, a)

    return Treat

# 置信区间输出威胁等级

def intervalEvaluation(Treat):
    low = 0.1
    mid = 0.5
    EvaResult = ''
    if Treat <= low:
        EvaResult = '低危'
    elif low < Treat <= mid:
        EvaResult = '中危'
    else:
        EvaResult = '高危'
    return EvaResult