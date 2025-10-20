import matplotlib.pyplot as plt
import numpy as np
import math as m
import copy

from utils.common import ComputeHeading, CalAngle, CalSelfPosition, CalDistance
from matplotlib import style
from typing import List, Optional


MaxInterceptorDist = 10000   # 拦截弹最远距离
MinInterceptorDist = 300    # 拦截弹最近距离
MaxInterceptorAngle = 90 * m.pi / 180
GoStep = 10  # 步长大小
MinV = 170

"""导弹类，包含导弹初始发射状态的初始化、位置的计算。

    初始化参数：
    
    ----------
    
    missile_plist:List类型 [X坐标, Y坐标, Z坐标]
                  导弹位置信息
    
    V:double类型 
        导弹发射初速度
        
    Pitch:double类型 
            导弹发射的初始迎角
    
    Heading:double类型
          导弹发射的初始偏航角
          
    dt:double类型
       仿真步长
       
    g:double类型
       重力加速度
       
    k1:int类型
       比例导引系数
       
    k2:int类型
       比例导引系数
    
    Returns
    
    -----------
    
    Missiles对象                        
"""


class Missiles:
    def __init__(
        self,
        missile_plist: list,
        V,
        Pitch,
        Heading,
        dt=0.01,
        g=9.6,
        k1=7,
        k2=7,
        target_speed: float = 1200.0,
        boost_duration: float = 5.0,
        speed_decay_interval: float = 1.0,
        speed_decay_factor: float = 0.99,
    ):
        self.X, self.Y, self.Z = missile_plist  # 导弹发射时位于的坐标
        self.V = V  # 导弹发射初速度
        self.Pitch, self.Heading = Pitch, Heading  # 导弹发射的初始俯仰角和偏航角

        self.attacking = True  # 导弹是否还在飞行

        # 导弹自身参数
        self.g = g  # 重力加速度
        self.dt = dt  # 时间精度
        self.k1, self.k2 = k1, k2  # 比例系数

        # 两段运动参数
        self.target_speed = target_speed
        self.boost_duration = boost_duration
        self.speed_decay_interval = speed_decay_interval
        self.speed_decay_factor = speed_decay_factor
        self.initial_speed = V
        self.time = 0.0
        self._last_decay_time = boost_duration

    def _advance_time(self):
        next_time = self.time + self.dt
        if next_time <= self.boost_duration:
            progress = next_time / self.boost_duration
            self.V = self.initial_speed + (self.target_speed - self.initial_speed) * progress
            self._last_decay_time = self.boost_duration
        else:
            if self.time < self.boost_duration:
                self.V = self.target_speed
            elapsed_since_decay = next_time - self._last_decay_time
            if elapsed_since_decay >= self.speed_decay_interval:
                decay_steps = int(elapsed_since_decay / self.speed_decay_interval)
                self.V *= self.speed_decay_factor ** decay_steps
                self._last_decay_time += self.speed_decay_interval * decay_steps
        self.time = next_time
        self.V = max(self.V, 200.0)

    """导弹位置计算函数，除了导弹自身的系数，还需传入目标的实时位置、速度，角度等信息。为导弹类的成员函数

        参数：

        ----------

        aircraft_plist:List类型 [X坐标, Y坐标, Z坐标]
                      目标机位置信息

        V:double类型 
            打击机速度

        Pitch:double类型 
                打击机迎角

        Heading:double类型
              打击机偏航角


        Returns

        -----------

        变更了位置X,Y,Z的Missiles对象                        
    """

    def MissilePosition(self, aircraft_plist: list, V_t, theta_t, fea_t):
        self._advance_time()
        # 目标实时位置
        X_m = self.X
        Y_m = self.Y
        Z_m = self.Z

        V_m = self.V
        Heading_m = self.Heading
        Pitch_m = self.Pitch
        g = self.g
        k1 = self.k1
        k2 = self.k2
        dt = self.dt

        X_t, Y_t, Z_t = aircraft_plist

        dX_m = V_m * m.cos(Pitch_m) * m.cos(Heading_m)
        dY_m = V_m * m.sin(Pitch_m)  # V_m * m.cos(Pitch_m) * m.cos(Heading_m)
        dZ_m = - V_m * m.cos(Pitch_m) * m.sin(
            Heading_m)  # dZ_m = - V_m * m.cos(Pitch_m) * m.sin(Heading_m)

        dX_t = V_t * m.cos(theta_t) * m.cos(fea_t)
        dY_t = V_t * m.sin(theta_t)  # V_t * m.cos(theta_t) * m.cos(fea_t)
        dZ_t = - V_t * m.cos(theta_t) * m.sin(fea_t)  # dZ_t = - V_t * m.cos(theta_t) * m.sin(fea_t)
        dist = m.sqrt(
            (X_m - X_t) * (X_m - X_t) + (Y_m - Y_t) * (Y_m - Y_t) + (Z_m - Z_t) * (
                    Z_m - Z_t))
        dR = ((Y_m - Y_t) * (dY_m - dY_t) + (Z_m - Z_t) * (dZ_m - dZ_t) + (X_m - X_t) * (
                dX_m - dX_t)) / dist

        dtheta_L = ((dY_t - dY_m) * m.sqrt((X_t - X_m) ** 2 + (Z_t - Z_m) ** 2) - (Y_t - Y_m) * (
                (X_t - X_m) * (dX_t - dX_m) + (Z_t - Z_m) * (dZ_t - dZ_m)) / m.sqrt(
            (X_t - X_m) ** 2 + (Z_t - Z_m) ** 2)) / (
                           (X_m - X_t) ** 2 + (Y_m - Y_t) ** 2 + (Z_m - Z_t) ** 2)
        ny = k1 * abs(dR) * dtheta_L / g
        dtheta_m = g / V_m * (ny - m.cos(Pitch_m))
        Pitch_m = Pitch_m + dtheta_m * dt

        dfea_L = ((dZ_t - dZ_m) * (X_t - X_m) - (Z_t - Z_m) * (dX_t - dX_m)) / (
                (X_t - X_m) ** 2 + (Z_t - Z_m) ** 2)
        nz = k2 * abs(dR) * dfea_L / g
        dfea_m = - g / (V_m * m.cos(Pitch_m)) * nz
        Heading_m = Heading_m + dfea_m * dt

        X_m = X_m + V_m * m.cos(Pitch_m) * m.cos(Heading_m) * dt
        Y_m = Y_m + V_m * m.sin(Pitch_m) * dt
        Z_m = Z_m - V_m * m.cos(Pitch_m) * m.sin(Heading_m) * dt

        self.X = X_m
        self.Y = Y_m
        self.Z = Z_m
        self.Pitch = Pitch_m
        self.Heading = Heading_m

        # 将偏转角稳定在【-pi, pi】上
        if self.Heading > m.pi:
            self.Heading = 2 * m.pi - self.Heading
        elif self.Heading < -m.pi:
            self.Heading = 2 * m.pi + self.Heading

        return [X_m, Y_m, Z_m]


"""----------------------------------------------------------------------------------------
"""

"""飞机类，包含来袭导弹的位置，初始飞行位置、速度的初始化，其后位置的计算。

    初始化参数：

    ----------

    aircraft_plist:List类型 [X坐标, Y坐标, Z坐标]
                  飞机位置信息

    V:double类型 
        飞机起飞初速度

    Pitch:double类型 
            飞机飞行的初始俯仰角

    Heading:double类型
          飞机飞行的初始偏航角

    dt:double类型
       仿真步长 ： 0.01

    g:double类型
       重力加速度 ：9.6
    
    Returns

    -----------

    Aircraft对象                        
"""


class Aircraft:
    def __init__(self, aircraft_plist: list, V, Pitch, Heading, dt=0.01, g=9.6, MaxV=408):
        self.X, self.Y, self.Z = aircraft_plist  # 飞机飞行时位于的坐标
        self.V = V  # 飞机飞行初速度
        self.Pitch, self.Heading = Pitch, Heading  # 飞机飞行的初始偏转角和偏航角
        self.MaxV = MaxV
        self.MinV = MinV

        # 环境参数
        self.g = g  # 重力加速度
        self.dt = dt  # 时间精度

        # 飞机自身参数
        self.nx = 0  # 切向过载
        self.ny = 1  # 法向过载
        self.roll = 0  # 当前滚转角

    """
        拦截弹发射限制条件
        
        输入参数：
            目标位置
            
        Return:
        ----------
        是否可发射Bool型
        
    """

    def LimitCondition(self, TargetPos: List):

        Angle = CalAngle(TargetPos, [self.X, self.Y, self.Z], self.Heading)
        if not(0 <= Angle <= m.pi):

            print(Angle)



        dist = np.linalg.norm(np.array(TargetPos) - np.array([self.X, self.Y, self.Z]))


        LanchFlag = False

        if MinInterceptorDist < dist < MaxInterceptorDist :
            LanchFlag = True

        return LanchFlag

    """
        动作约束：俯仰角不为零时不能平飞和转弯

        输入参数：
                act向量：[nx, ny, roll, pitchconstraint, AttTarget]
        根据pitchconstraint=-1无约束，若=0则说明进行的水平面运动，不可以在自身pitch!=0时进行

        ----------
        返回：
            constraint_flag=true时，可以进行动作
    """

    def action_constraint(self, PitchConstraint):

        constraint_flag = True

        aPitch = self.Pitch
        if PitchConstraint == 0 and aPitch != 0:
            constraint_flag = False

        if not (-45 * m.pi / 180 <= self.Pitch <= 60 * m.pi / 180):
            constraint_flag = False
        return constraint_flag


    def speed_constraint(self, nx):

        constraint_flag = True

        if self.V >= self.MaxV and nx != 0:
            constraint_flag = False

        if self.V < self.MinV and nx < 0:
            constraint_flag = False

        return constraint_flag


    """
        拦截结果预测函数：假设导弹保持飞行姿态不变！
                    输入来袭导弹位置、速度、偏航角、俯仰角信息，根据飞机转弯的速度和导弹飞行的距离判断是否可以拦截
        
        参数：
        
        ----------
        target_position:List类型 [X坐标, Y坐标, Z坐标]
                      来袭导弹位置信息

        t_V:double类型 
            来袭导弹速度

        t_Pitch:double类型 
                来袭导弹迎角

        t_Heading:double类型
              来袭导弹偏航角
              
        Returns:
        ----------
        
        预测拦截结果Bool型：Result
        机动过载int型：Rny
        滚转角float型：Rroll
        预估转向后距离double型：Rdist
        
    """

    def PredictInceptorResult(self, target_position, t_V, t_Pitch, t_Heading):

        # 预测结果
        Result = False
        Rny = 0
        Rroll = 0
        Rdist = 10e8  # 初始值为极大值
        # 时间步
        t = 0

        # 画图测试数组
        missileArray = np.zeros((300, 3), dtype=np.float32)
        planeArray = np.zeros((300, 3), dtype=np.float32)

        # 取参数
        t_X0, t_Y0, t_Z0 = target_position

        # 虚拟化对象，保证不改变当前对象的属性
        virtual_plane = copy.deepcopy(self)
        Heading_m = t_Heading
        Pitch_m = t_Pitch
        V_m = t_V
        X = virtual_plane.X
        Y = virtual_plane.Y
        Z = virtual_plane.Z
        Pitch = virtual_plane.Pitch
        Heading = virtual_plane.Heading
        V = virtual_plane.V

        # 不需要机动就能拦截
        Angle = CalAngle([t_X0, t_Y0, t_Z0], [X, Y, Z], Heading)
        dist = CalDistance([virtual_plane.X, virtual_plane.Y, virtual_plane.Z], [t_X0, t_Y0, t_Z0])

        if 0 <= Angle <= 90 * m.pi / 180 and dist < MaxInterceptorDist:
            Result = True
            Rdist = dist
            Rny = self.ny
            Rroll = self.roll
            return Result, Rny, Rroll, Rdist, planeArray, missileArray

        """
            遍历转弯动作空间
        """
        # 转弯的目标
        Target_Angle = m.pi + Heading
        direc = ComputeHeading([t_X0, t_Y0, t_Z0], [X, Y, Z])

        # 转弯方向
        if direc > Heading:
            direction = 1
        else:
            direction = -1
        # 遍历动作空间
        for ny in range(2, 5):
            roll = direction * m.acos(1 / ny)

            # 重置导弹坐标
            t_X, t_Y, t_Z = t_X0, t_Y0, t_Z0

            # 重置飞机坐标和角度
            virtual_plane.X, virtual_plane.Y, virtual_plane.Z, virtual_plane.V, virtual_plane.Pitch, virtual_plane.Heading = X, Y, Z, V, Pitch, Heading

            # 重置步数与数组
            t = 0
            missileArray = np.zeros((300, 3), dtype=np.float32)
            planeArray = np.zeros((300, 3), dtype=np.float32)

            while virtual_plane.Heading <= Target_Angle:
                # 时间步推进
                t += 1
                for _ in range(GoStep):
                    virtual_plane.AircraftPostition(ny=ny, roll=roll)
                planeArray[t] = [virtual_plane.X, virtual_plane.Y, virtual_plane.Z]

            i = 0

            for i in range(t):
                for _ in range(GoStep):
                    t_X, t_Y, t_Z = CalSelfPosition([t_X, t_Y, t_Z], V=V_m, Heading=Heading_m, Pitch=Pitch_m,
                                                    dt=self.dt)
                missileArray[i] = t_X, t_Y, t_Z

            dist = CalDistance([virtual_plane.X, virtual_plane.Y, virtual_plane.Z], [t_X, t_Y, t_Z])

            Angle = CalAngle([t_X, t_Y, t_Z], [virtual_plane.X, virtual_plane.Y, virtual_plane.Z],
                             virtual_plane.Heading)

            #   保存最优的机动动作
            if Rdist < dist <= MaxInterceptorDist and 0 <= Angle <= MaxInterceptorAngle:
                Result = True
                Rdist = dist
                Rny = ny
                Rroll = roll

        return Result, Rny, Rroll, Rdist, planeArray, missileArray

    """飞机位置计算函数，除了飞机自身的系数，还需传入导弹的实时位置、速度，角度等信息。为飞机类的成员函数

        参数：

        ----------

        missile_plist:List类型 [X坐标, Y坐标, Z坐标]
                      来袭导弹位置信息

        V:double类型 
            来袭导弹速度

        Pitch:double类型 
                来袭导弹迎角

        Heading:double类型
              来袭导弹偏航角


        Returns

        -----------

        变更了位置X,Y,Z的Aircraft对象                        
    """

    def AircraftPostition(self, missile_plist=None, nx=0, ny=1, roll=0.0, Pitch=0.0):
        if Pitch == -1:
            self.Pitch = self.Pitch
        elif Pitch == 0:
            self.Pitch = 0
        self.roll = roll
        self.ny = ny
        self.nx = nx
        if missile_plist is None:
            missile_plist = []
        _V = self.g * (self.nx - m.sin(self.Pitch))  # 速度标量加速度
        _Pitch = (self.g / self.V) * (self.ny * m.cos(self.roll) - m.cos(self.Pitch))  # 俯仰角速度
        _Heading = self.g * self.ny * m.sin(self.roll) / (self.V * m.cos(self.Pitch))  # 偏航角速度

        self.V += _V * self.dt
        self.Pitch += _Pitch * self.dt
        self.Heading += _Heading * self.dt

        # 将偏转角稳定在【-pi, pi】上
        if self.Heading > m.pi:
            self.Heading = 2 * m.pi - self.Heading
        elif self.Heading < -m.pi:
            self.Heading = 2 * m.pi + self.Heading

        self.X = self.X + self.V * m.cos(self.Pitch) * m.cos(self.Heading) * self.dt
        self.Y = self.Y + self.V * m.sin(self.Pitch) * self.dt
        self.Z = self.Z - self.V * m.cos(self.Pitch) * m.sin(self.Heading) * self.dt

        return self.X, self.Y, self.Z

    """飞机对象属性修改函数，为飞机类的成员函数

        参数：

        ----------

        aircraft_plist:List类型 [X坐标, Y坐标, Z坐标]
                      位置信息

        V:double类型 
            速度

        Pitch:double类型 
                俯仰角

        Heading:double类型
              偏航角


        Returns

        -----------

        变更了属性的Aircraft对象                        
    """

    def setAttr(self, aircraft_plist, V, Pitch, Heading):
        self.X, self.Y, self.Z = aircraft_plist
        self.V = V
        self.Pitch = Pitch
        self.Heading = Heading


"""拦截弹类，包含拦截弹初始发射状态的初始化、位置的计算。

    初始化参数：

    ----------

    interceptor_plist:List类型 [X坐标, Y坐标, Z坐标]
                  拦截弹位置信息

    v_m:double类型 
        拦截弹发射初速度

    Pitch_m:double类型 
            拦截弹发射的初始迎角

    Heading_m:double类型
          拦截弹发射的初始偏航角

    dt:double类型
       仿真步长

    mg:double类型
       重力加速度

    k1:int类型
       比例导引系数

    k2:int类型
       比例导引系数

    Returns

    -----------

    Interceptor对象                        
"""


class Interceptor:
    def __init__(
        self,
        interceptor_plist: list,
        v_i,
        Pitch_i,
        Heading_i,
        dt=0.01,
        mg=9.6,
        k1=6.5,
        k2=6.5,
        target_speed: float = 1000.0,
        boost_duration: float = 5.0,
        speed_decay_interval: float = 1.0,
        speed_decay_factor: float = 0.99,
    ):
        self.X_i, self.Y_i, self.Z_i = interceptor_plist  # 拦截弹发射时位于的坐标
        self.V_i = v_i  # 拦截弹发射初速度
        self.Pitch_i, self.Heading_i = Pitch_i, Heading_i  # 拦截弹发射的初始俯仰角和偏航角

        self.T_i = - 1  # 拦截弹锁定的来袭导弹编号
        self.attacking = -1  # 拦截弹是否就绪或飞行或打中，初始-1为就绪状态

        # 拦截弹自身参数
        self.mg = mg  # 重力加速度
        self.dt = dt  # 时间精度
        self.k1, self.k2 = k1, k2  # 比例系数

        # 两段运动参数
        self.target_speed = target_speed
        self.boost_duration = boost_duration
        self.speed_decay_interval = speed_decay_interval
        self.speed_decay_factor = speed_decay_factor
        self.initial_speed = v_i
        self.time = 0.0
        self._last_decay_time = boost_duration

    def reset_dynamics(self, launch_speed: Optional[float] = None):
        if launch_speed is not None:
            self.initial_speed = launch_speed
        self.V_i = self.initial_speed
        self.time = 0.0
        self._last_decay_time = self.boost_duration

    def begin_pursuit(self, target_index: int, launch_speed: float):
        self.T_i = target_index
        self.attacking = 0
        self.reset_dynamics(launch_speed)

    def sync_with_aircraft(self, position, pitch, heading, speed):
        self.X_i, self.Y_i, self.Z_i = position
        self.Pitch_i = pitch
        self.Heading_i = heading
        if self.attacking == -1:
            self.initial_speed = speed
            self.V_i = speed

    def _advance_time(self):
        next_time = self.time + self.dt
        if next_time <= self.boost_duration:
            progress = next_time / self.boost_duration
            self.V_i = self.initial_speed + (self.target_speed - self.initial_speed) * progress
            self._last_decay_time = self.boost_duration
        else:
            if self.time < self.boost_duration:
                self.V_i = self.target_speed
            elapsed_since_decay = next_time - self._last_decay_time
            if elapsed_since_decay >= self.speed_decay_interval:
                decay_steps = int(elapsed_since_decay / self.speed_decay_interval)
                self.V_i *= self.speed_decay_factor ** decay_steps
                self._last_decay_time += self.speed_decay_interval * decay_steps
        self.time = next_time
        self.V_i = max(self.V_i, 200.0)

    """拦截弹位置计算函数，除了拦截弹自身的系数，还需传入来袭导弹的实时位置、速度，角度等信息。为拦截弹类的成员函数

        参数：

        ----------

        missile_plist:List类型 [X坐标, Y坐标, Z坐标]
                      来袭导弹位置信息

        V_m:double类型 
            来袭导弹速度

        Pitch_m:double类型 
                来袭导弹俯仰角

        Heading_t:double类型
              来袭导弹偏航角


        Returns

        -----------

        变更了位置X,Y,Z的Interceptor对象                        
    """

    def InterceptorPosition(self, missile_plist: list, V_m, Pitch_m, Heading_m):
        self._advance_time()
        # 目标实时位置
        X_m, Y_m, Z_m = missile_plist

        dX_i = self.V_i * m.cos(self.Pitch_i) * m.cos(self.Heading_i)
        dY_i = self.V_i * m.sin(self.Pitch_i)  # self.V_m * m.cos(self.Pitch_m) * m.cos(self.Heading_m)
        dZ_i = - self.V_i * m.cos(self.Pitch_i) * m.sin(
            self.Heading_i)  # dZ_m = - self.V_m * m.cos(self.Pitch_m) * m.sin(self.Heading_m)

        dX_m = V_m * m.cos(Pitch_m) * m.cos(Heading_m)
        dY_m = V_m * m.sin(Pitch_m)  # V_t * m.cos(Pitch_t) * m.cos(Heading_t)
        dZ_m = - V_m * m.cos(Pitch_m) * m.sin(Heading_m)  # dZ_t = - V_t * m.cos(Pitch_t) * m.sin(Heading_t)
        dist = m.sqrt(
            (self.X_i - X_m) * (self.X_i - X_m) + (self.Y_i - Y_m) * (self.Y_i - Y_m) + (self.Z_i - Z_m) * (
                    self.Z_i - Z_m))
        dR = ((self.Y_i - Y_m) * (dY_i - dY_m) + (self.Z_i - Z_m) * (dZ_i - dZ_m) + (self.X_i - X_m) * (
                dX_i - dX_m)) / dist

        dPitch_L = ((dY_m - dY_i) * m.sqrt((X_m - self.X_i) ** 2 + (Z_m - self.Z_i) ** 2) - (Y_m - self.Y_i) * (
                (X_m - self.X_i) * (dX_m - dX_i) + (Z_m - self.Z_i) * (dZ_m - dZ_i)) / m.sqrt(
            (X_m - self.X_i) ** 2 + (Z_m - self.Z_i) ** 2)) / (
                           (self.X_i - X_m) ** 2 + (self.Y_i - Y_m) ** 2 + (self.Z_i - Z_m) ** 2)
        ny = self.k1 * abs(dR) * dPitch_L / self.mg
        dPitch_i = self.mg / self.V_i * (ny - m.cos(self.Pitch_i))
        self.Pitch_i = self.Pitch_i + dPitch_i * self.dt

        dHeading_L = ((dZ_m - dZ_i) * (X_m - self.X_i) - (Z_m - self.Z_i) * (dX_m - dX_i)) / (
                (X_m - self.X_i) ** 2 + (Z_m - self.Z_i) ** 2)
        nz = self.k2 * abs(dR) * dHeading_L / self.mg
        dHeading_i = - self.mg / (self.V_i * m.cos(self.Pitch_i)) * nz
        self.Heading_i = self.Heading_i + dHeading_i * self.dt

        self.X_i = self.X_i + self.V_i * m.cos(self.Pitch_i) * m.cos(self.Heading_i) * self.dt
        self.Y_i = self.Y_i + self.V_i * m.sin(self.Pitch_i) * self.dt
        self.Z_i = self.Z_i - self.V_i * m.cos(self.Pitch_i) * m.sin(self.Heading_i) * self.dt

        # 将偏转角稳定在【-pi, pi】上
        if self.Heading_i > m.pi:
            self.Heading_i = 2 * m.pi - self.Heading_i
        elif self.Heading_i < -m.pi:
            self.Heading_i = 2 * m.pi + self.Heading_i

        return [self.X_i, self.Y_i, self.Z_i]


"""----------------------------------------------------------------------------------------
"""
"""
获取位置列表函数，传入导弹列表和飞机列表，给出导弹追踪时的

    参数：

    ----------

    missile_plist:List[Missiles]类型
                  来袭导弹类信息列表

    target: List[Aircraft]类型 
        飞行器类信息列表


    Returns

    -----------

    导弹的位置列表和飞机的位置列表
"""


# def getPosition(missilesList: List[Missiles], target: List[Aircraft]):
#     missilesLocationList = []  # 用于保存所有导弹绘图数据
#
#     aircraftLocationList = []  # 用于保存所有飞机绘图数据
#
#     for _ in missilesList:
#         missilesLocationList.append([[], [], []])
#
#     for _ in target:
#         aircraftLocationList.append([[], [], []])
#     moveFlag = True  # 飞机终止移动标志，击中后不更新飞机位置
#     finalFlag = False  # 终止标志，所有导弹均击中
#     hitNum = [0 for _ in range(len(missilesList))]  # 导弹终止标志，导弹击中飞机后不再移动
#     tx, ty, tz = target[0].AircraftPostition()  # 目前只有一个飞机
#     while 1:
#         X, Y, Z, V, Pitch, Heading = target[0].X, target[0].Y, target[0].Z, target[0].V, target[
#             0].Pitch, target[0].Heading  # 获取飞机位置等信息
#         ac_list = [X, Y, Z]
#         for i in range(len(missilesList)):
#             if hitNum[i] == 1:
#                 continue
#             mx, my, mz = missilesList[i].MissilePosition(ac_list, V, Pitch, Heading)
#             missilesLocationList[i][0].append(mx)  # 添加X轴坐标
#             missilesLocationList[i][1].append(my)  # 添加Y轴坐标
#             missilesLocationList[i][2].append(mz)  # 添加Z轴坐标
#             dist = m.sqrt((tx - mx) ** 2 + (ty - my) ** 2 + (tz - mz) ** 2)
#             if dist < 10:
#                 moveFlag = False
#                 hitNum[i] = 1
#         if hitNum.count(1) == len(missilesList):
#             finalFlag = True
#             break
#         # 计算飞机位置信息
#         if moveFlag:
#             tx, ty, tz = target[0].AircraftPostition()
#
#         # 将飞机计算出的坐标画图
#         aircraftLocationList[0][0].append(tx)
#         aircraftLocationList[0][1].append(ty)
#         aircraftLocationList[0][2].append(tz)
#
#         if finalFlag:
#             break
#     return missilesLocationList, aircraftLocationList
# 利用数组进行来加入时间维度
# def getPosition(missilesList: List[Missiles], target: List[Aircraft]):
#     missilesLocationList = []  # 用于保存所有导弹绘图数据
#     missilesLocationArray = np.zeros((len(missilesList), Maxstep, 4), dtype=np.float32)
#     # print(missilesLocationArray[0][50])
#     aircraftLocationList = []  # 用于保存所有飞机绘图数据
#     aircraftLocationArray = np.zeros((len(target), Maxstep, 4), dtype=np.float32)
#
#     for _ in missilesList:
#         missilesLocationList.append([[], [], []])
#
#     for _ in target:
#         aircraftLocationList.append([[], [], []])
#     moveFlag = True  # 飞机终止移动标志，击中后不更新飞机位置
#     finalFlag = False  # 终止标志，所有导弹均击中
#     hitNum = [0 for _ in range(missilesNum)]  # 导弹终止标志，导弹击中飞机后不再移动
#     tx, ty, tz = target[0].AircraftPostition()  # 目前只有一个飞机
#     t = 0
#     while t < Maxstep:
#         X, Y, Z, V, Pitch, Heading = target[0].X, target[0].Y, target[0].Z, target[0].V, target[
#             0].Pitch, target[0].Heading  # 获取飞机位置等信息
#         ac_list = [X, Y, Z]
#         for i in range(len(missilesList)):
#             if hitNum[i] == 1:
#                 missilesLocationArray[i, t] = missilesLocationArray[i, t - 1]
#                 continue
#             mx, my, mz = missilesList[i].MissilePosition(ac_list, V, Pitch, Heading)
#             missilesLocationArray[i][t] = np.array([t, mx, my, mz])
#             # print(aircraftLocationArray[0][t])
#             dist = m.sqrt((tx - mx) ** 2 + (ty - my) ** 2 + (tz - mz) ** 2)
#             if dist < 10:
#                 moveFlag = False
#                 hitNum[i] = 1
#                 return 'hit on！', t, missilesLocationArray, aircraftLocationArray
#         if hitNum.count(1) == len(missilesList):
#             finalFlag = True
#             break
#             # 计算飞机位置信息
#         if moveFlag:
#             tx, ty, tz = target[0].AircraftPostition()
#         aircraftLocationArray[0][t] = np.array([t, tx, ty, tz])
#
#         if finalFlag:
#             break
#         t += 1
#     return 'escape！', Maxstep, missilesLocationArray, aircraftLocationArray
    # while 1:
    #     X, Y, Z, V, Pitch, Heading = target[0].X, target[0].Y, target[0].Z, target[0].V, target[
    #         0].Pitch, target[0].Heading  # 获取飞机位置等信息
    #     ac_list = [X, Y, Z]
    #     for i in range(len(missilesList)):
    #         if hitNum[i] == 1:
    #             continue
    #         mx, my, mz = missilesList[i].MissilePosition(ac_list, V, Pitch, Heading)
    #         missilesLocationList[i][0].append(mx)  # 添加X轴坐标
    #         missilesLocationList[i][1].append(my)  # 添加Y轴坐标
    #         missilesLocationList[i][2].append(mz)  # 添加Z轴坐标
    #         dist = m.sqrt((tx - mx) ** 2 + (ty - my) ** 2 + (tz - mz) ** 2)
    #         if dist < 10:
    #             moveFlag = False
    #             hitNum[i] = 1
    #     if hitNum.count(1) == len(missilesList):
    #         finalFlag = True
    #         break
    #     # 计算飞机位置信息
    #     if moveFlag:
    #         tx, ty, tz = target[0].AircraftPostition()
    #
    #     # 将飞机计算出的坐标画图
    #     aircraftLocationList[0][0].append(tx)
    #     aircraftLocationList[0][1].append(ty)
    #     aircraftLocationList[0][2].append(tz)
    #
    #     if finalFlag:
    #         break
    # return missilesLocationList, aircraftLocationList


"""----------------------------------------------------------------------------------------
"""
"""
根据位置信息列表打印图表的函数

    参数：

    ----------

    missilesLocationList:List类型
                  来袭导弹位置信息列表

    target: List类型 
        飞行器位置信息列表


    Returns

    -----------

    无返回值，画图plt.show()
"""


#
# def printPlot(missilesLocationList, target):
#     style.use('ggplot')
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, projection='3d')
#
#     for mis in missilesLocationList:
#         ax1.plot(mis[0], mis[2], mis[1], c='g')
#     ax1.plot(target[0][0], target[0][2], target[0][1])
#
#     # 定义标签
#     ax1.set_xlabel('x-axis')
#     ax1.set_ylabel('y-axis')
#     ax1.set_zlabel('z-axis')
# 3
#     # 出图
#     plt.show()


def printPlot(missilesLocationList, target, step):
    style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    for i in range(missilesLocationList.shape[0]):
        finalStep = np.max(missilesLocationList[i][1], axis=0)
        ax1.plot(missilesLocationList[i, 0:step - 1, 1], missilesLocationList[i, 0:step - 1, 3],
                 missilesLocationList[i, 0:step - 1, 2], c='g')
        # ax1.plot(mis[0], mis[2], mis[1], c='g')
    finalStep = np.max(target[0][1], axis=0)
    # print(finalStep, 'hjghjghj')
    ax1.plot(target[0, 0:step - 1, 1], target[0, 0:step - 1, 3], target[0, 0:step - 1, 2], c='r')

    # 定义标签
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')
    ax1.set_zlabel('z-axis')

    # 出图
    plt.show()

# # 测试missile类
# ms = Missiles([0, 1000, 0], 260, 45 * m.pi / 180, 10 * m.pi / 180)
# ms2 = Missiles([0, 1000, 500], 260, 45 * m.pi / 180, 10 * m.pi / 180)
# ms3 = Missiles([0, 1000, 1000], 260, 45 * m.pi / 180, 10 * m.pi / 180)
# # 测试aircraft类
# plane = Aircraft([1000, 2000, 2000], 150, 0, 30 * m.pi / 180)
#
# # 声明图表参数
# style.use('ggplot')
#
# missilesList = [ms, ms2, ms3]
#
# aircraftList = [plane]
#
# info, step, missilesLocationArray, targetLocationArray = getPosition(missilesList, aircraftList)

# print(info)

# print(missilesLocationList[0][0])
# print(missilesLocationArray)
# printPlot(missilesLocationArray, targetLocationArray, step)
