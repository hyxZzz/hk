import numpy as np
import math as m
from utils.common import ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles


def reset_para(num_missiles=3, StepNum=1200):
    # 飞机的初始位置，x和y为[-10000, 10000], z为服从2000为均值，300为标准差的正态分布

    # 飞行高度8~12Km
    a_x = np.random.uniform(-10000, 10000)
    a_y = np.random.uniform(-10000, 10000)
    a_z = np.random.uniform(8000, 12000)

    # print(a_x, a_y, a_z)

    # 飞行速度0.5~1.2马赫
    a_v = np.random.uniform(0.5, 1.2)
    a_v = a_v * 340

    # 飞机的俯仰角和偏转角
    aPitch = 0
    aHeading = np.random.uniform(-1, 1)
    # aHeading = np.random.uniform(0, 2)
    aHeading = aHeading * m.pi


    # 在水平椭圆区域内生成导弹位置
    angles = np.random.uniform(0, 2 * m.pi, size=num_missiles)
    radial_scale = np.random.uniform(0.85, 1.15, size=num_missiles)
    major_axis = 20000.0
    minor_axis = 15000.0
    missile_x = a_x + major_axis * radial_scale * np.cos(angles)
    missile_y = a_y + minor_axis * radial_scale * np.sin(angles)
    altitude_offsets = np.random.uniform(-3000.0, 3000.0, size=num_missiles)
    missile_z = np.clip(a_z + altitude_offsets, 0.0, None)
    mposList = []
    mHeadingList = []
    mPitchList = []
    for i in range(len(missile_x)):
        mposList.append([missile_x[i], missile_z[i], missile_y[i]])
        mHeadingList.append(ComputeHeading([a_x, a_z, a_y], [missile_x[i], missile_z[i], missile_y[i]]))
        mPitchList.append(ComputePitch([a_x, a_z, a_y], [missile_x[i], missile_z[i], missile_y[i]]))
    # print(mposList)

    # 导弹速度不低于2马赫
    m_v = np.random.uniform(2, 3)
    m_v = m_v * 340

    aircraft_agent = [Aircraft([a_x, a_z, a_y], V=a_v, Pitch=aPitch, Heading=aHeading)]
    missiles_list = []
    for i in range(len(mposList)):
        missiles_list.append(Missiles(mposList[i], V=m_v, Pitch=mPitchList[i], Heading=mHeadingList[i]))


    # """
    #     一致性测试
    # """
    #
    # aPos = [2000, 1000, 2000]
    # # 测试aircraft类
    # plane = Aircraft(aPos, 340, 0, 180 * m.pi / 180, )
    # m1Pos = [1000, 2000, 1000]
    # m2Pos = [1000, 2000, 500]
    # heading1 = ComputeHeading(aPos, m1Pos)
    # ms = Missiles(m1Pos, 680, ComputePitch(aPos, m1Pos), heading1)
    # ms2 = Missiles(m2Pos, 680, ComputePitch(aPos, m2Pos), ComputeHeading(aPos, m2Pos))
    # missiles_list = [ms, ms2]
    # aircraft_agent = [plane]
    # m_v = 680
    # a_v = 340

    return missiles_list, aircraft_agent[0], a_v, num_missiles, StepNum, m_v

