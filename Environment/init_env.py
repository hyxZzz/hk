from typing import List, Tuple

from Environment.env import ManeuverEnv
from Environment.reset_env import reset_para
from flat_models.trajectory import Aircraft, Missiles


def init_env(
    num_missiles: int = 4,
    StepNum: int = 1000,
    interceptor_num: int = 6,
    num_planes: int = 2,
) -> Tuple[ManeuverEnv, List[Aircraft], List[Missiles]]:
    missiles_list, aircraft_list, plane_speeds, _, _, missile_speed = reset_para(
        num_missiles=num_missiles,
        StepNum=StepNum,
        num_planes=num_planes,
    )

    env = ManeuverEnv(
        missiles_list,
        aircraft_list,
        planeSpeed=sum(plane_speeds) / max(len(plane_speeds), 1),
        missilesNum=num_missiles,
        spaceSize=StepNum,
        missilesSpeed=missile_speed,
        InterceptorNum=interceptor_num,
    )

    return env, list(aircraft_list), list(missiles_list)
