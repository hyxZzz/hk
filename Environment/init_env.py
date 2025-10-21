from typing import List, Tuple

from typing import List, Tuple

from Environment.env import ManeuverEnv, SOFT_CONSTRAINT_DEFAULT
from Environment.reset_env import reset_para
from flat_models.trajectory import Aircraft, Missiles


def init_env(
    num_missiles: int = 4,
    StepNum: int = 1000,
    interceptor_num: int = 6,
    num_planes: int = 2,
    soft_penalty_scale: float = SOFT_CONSTRAINT_DEFAULT,
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
        soft_penalty_scale=soft_penalty_scale,
    )

    return env, list(aircraft_list), list(missiles_list)
