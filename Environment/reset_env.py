import math as m
from typing import List

import numpy as np

from utils.common import ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles


def reset_para(num_missiles: int = 3, StepNum: int = 1200, num_planes: int = 1):
    aircraft_list: List[Aircraft] = []

    for _ in range(num_planes):
        a_x = np.random.uniform(-10000, 10000)
        a_y = np.random.uniform(-10000, 10000)
        a_z = np.random.uniform(8000, 12000)

        a_v = np.random.uniform(0.5, 1.2) * 340
        a_pitch = 0.0
        a_heading = np.random.uniform(-1, 1) * m.pi

        aircraft_list.append(Aircraft([a_x, a_z, a_y], V=a_v, Pitch=a_pitch, Heading=a_heading))

    center = np.mean([[plane.X, plane.Z, plane.Y] for plane in aircraft_list], axis=0)
    a_x, a_z, a_y = center

    angles = np.random.uniform(0, 2 * m.pi, size=num_missiles)
    radial_scale = np.random.uniform(0.85, 1.15, size=num_missiles)
    major_axis = 20000.0
    minor_axis = 15000.0
    missile_x = a_x + major_axis * radial_scale * np.cos(angles)
    missile_y = a_y + minor_axis * radial_scale * np.sin(angles)
    altitude_offsets = np.random.uniform(-3000.0, 3000.0, size=num_missiles)
    missile_z = np.clip(a_z + altitude_offsets, 0.0, None)

    missiles_list: List[Missiles] = []
    for mx, my, mz in zip(missile_x, missile_y, missile_z):
        heading = ComputeHeading([a_x, a_z, a_y], [mx, mz, my])
        pitch = ComputePitch([a_x, a_z, a_y], [mx, mz, my])
        missiles_list.append(Missiles([mx, mz, my], V=0.0, Pitch=pitch, Heading=heading))

    missile_speed = np.random.uniform(2, 3) * 340
    for missile in missiles_list:
        missile.V = missile_speed

    plane_speeds = [plane.V for plane in aircraft_list]

    return missiles_list, aircraft_list, plane_speeds, num_missiles, StepNum, missile_speed
