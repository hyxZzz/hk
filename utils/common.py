import math as m
import numpy as np

'''
根据当前位置计算自身指向目标时的偏航角
（相较于坐标轴的角度）
'''


def ComputeHeading(TargetPos, SelfPos):
    m_x = SelfPos[0]
    a_x = TargetPos[0]
    m_z = SelfPos[2]
    a_z = TargetPos[2]

    x = a_x - m_x
    z = a_z - m_z
    Heading = m.atan(abs(z) / abs(x + 10e-8))

    if (x >= 0 and z > 0):
        Heading = -Heading

    elif (x < 0 and z <= 0):
        Heading = m.pi - Heading

    elif (x < 0 and z > 0):
        Heading = Heading - m.pi

    return Heading  # 模型的xz坐标系，当x轴正方向向上↑👆的时候z轴正方向向左←👈


def ComputePitch(TargetPos, SelfPos):
    m_y = SelfPos[1]
    a_y = TargetPos[1]

    m_x = SelfPos[0]
    a_x = TargetPos[0]

    m_z = SelfPos[2]
    a_z = TargetPos[2]

    Pitch = m.atan((a_y - m_y) / m.sqrt(((a_x - m_x) ** 2) + ((a_z - m_z) ** 2) + 10e-8))

    return Pitch


"""
    函数作用：计算自身位置与目标位置的夹角
            
    -----------
    
    输入参数：
            
            
            
    Return
    --------
    
    夹角【0, 180】
    [0, 90]时，目标位置在自身飞行位置的前方
    [90, 180]时，目标在身后
    （两个速度矢量之间的相对角度）
"""


def CalAngle(TargetPos, SelfPos, Heading_Self):
    selfPos = np.array(SelfPos)

    TargetPos = np.array(TargetPos)

    Angle = abs(Heading_Self - ComputeHeading(TargetPos, selfPos))

    Angle = Angle % (2 * m.pi)
    if Angle > m.pi:
        Angle = 2 * m.pi - Angle

    return Angle


"""
    函数功能：计算速度矢量
    
    输入参数：
            Speed：速度
            Heading：偏转角
            Pitch：俯仰角
            
    ----------
    Returns
    
    ----------
    矢量的三个分量
    [vx, vy, vz]
    
"""


def ComputeVelocity(Speed, Heading, Pitch):
    SpeedX = Speed * m.cos(Pitch) * m.cos(Heading)
    SpeedY = Speed * m.sin(Pitch)
    SpeedZ = Speed * m.cos(Pitch) * m.sin(Heading)

    return SpeedX, SpeedY, SpeedZ


"""
    函数功能：计算归一化速度矢量
    
    输入参数：
            Speed：速度
            Heading：偏转角
            Pitch：俯仰角
            
    ----------
    Returns
    
    ----------
    矢量的三个分量
    [vx, vy, vz]
"""


def normalize(SpeedX, SpeedY, SpeedZ):
    length = SpeedX ** 2 + SpeedY ** 2 + SpeedZ ** 2  # 速度模长
    SpeedX = SpeedX / length
    SpeedY = SpeedY / length
    SpeedZ = SpeedZ / length
    return SpeedX, SpeedY, SpeedZ


"""
    函数功能：根据运动学方程计算位置
    
    输入参数：
            SelfPosition：[x, y, z]坐标
            V：标量速度
            Heading：偏转角
            Pitch：俯仰角
            dt: 时间差分
            
    ----------
    Returns
    
    ----------
    位置的三维坐标
    X, Y, Z
"""


def CalSelfPosition(SelfPosition, V, Heading, Pitch, dt=0.01):
    X, Y, Z = SelfPosition
    X = X + V * m.cos(Pitch) * m.cos(Heading) * dt
    Y = Y + V * m.sin(Pitch) * dt
    Z = Z - V * m.cos(Pitch) * m.sin(Heading) * dt
    return X, Y, Z


"""
    函数功能：根据两个点的坐标计算【距离】

    输入参数：
            SelfPosition：[x, y, z]坐标
            TargetPosition: [x, y, z]坐标


    ----------
    Returns

    ----------
    距离：distance： double型
"""


def CalDistance(SelfPosition, TargetPosition):
    SelfPosition = np.array(SelfPosition)
    TargetPosition = np.array(TargetPosition)

    distance = abs(np.linalg.norm(SelfPosition - TargetPosition))

    return distance
