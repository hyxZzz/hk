from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math as m
import numpy as np
from gym import spaces

from Environment.ActionDepository import getNewActionDepository
from Environment.reset_env import reset_para
from flat_models.ThreatEvaluate import CalTreat
from flat_models.trajectory import (
    Aircraft,
    Interceptor,
    Missiles,
    MaxInterceptorAngle,
    MaxInterceptorDist,
    MinInterceptorDist,
)
from utils.common import CalAngle, CalDistance


act_num = 29
Gostep = 1

INTERCEPT_SUCCESS_DISTANCE = 20.0
MISSILE_HIT_DISTANCE = 20.0
SPARSE_REWARD_SCALE = 5.0
DANGER_DISTANCE = 3000.0
LanchGap = 70
ERRACTIONSCALE = 2
DANGERSCALE = 1.5
CURIOSITYSCALE = 0.5
SAFE_DISTANCE_THRESHOLD = 8000.0
SOFT_CONSTRAINT_DEFAULT = 0.05
COOPERATIVE_OWNER_REWARD = 2.5
COOPERATIVE_TARGET_REWARD = 2.5
UNPROTECTED_PENALTY = 0.3
REPEATED_LOCK_PENALTY = 0.1
TIMEOUT_REWARD = -0.25
THREAT_DISENGAGE_REWARD = 2.0
APPROACH_REWARD_SCALE = 0.0005
INTERCEPT_APPROACH_SCALE = 0.002
LOCK_SUPPORT_REWARD = 0.05
LOCK_MAINTENANCE_REWARD = 0.03


@dataclass
class InterceptEvent:
    owner_plane: int
    missile_index: int
    target_plane: int
    lock_count: int = 0
    cooperative: bool = False
    owner_lock_counts: Optional[Dict[int, int]] = None


class ManeuverEnv:
    """Two-aircraft cooperative interception environment."""

    def __init__(
        self,
        missileList: List[Missiles],
        aircraftList: Union[Sequence[Aircraft], Aircraft],
        planeSpeed: float = 170.0,
        missilesNum: int = 4,
        spaceSize: int = 5000,
        missilesSpeed: float = 680.0,
        InterceptorNum: int = 6,
        InterceptorSpeed: float = 540.0,
        soft_penalty_scale: float = SOFT_CONSTRAINT_DEFAULT,
    ) -> None:
        if isinstance(aircraftList, Aircraft):
            aircrafts: List[Aircraft] = [aircraftList]
        else:
            aircrafts = list(aircraftList)

        self.action_space = spaces.Discrete(act_num * (missilesNum + 1))
        self.maneuver_count = act_num
        self.action_dep = getNewActionDepository(act_num)

        self.missileNum = missilesNum
        self.missileSpeed = missilesSpeed
        self.planeSpeed = planeSpeed
        self.spaceSize = spaceSize
        self.interceptor_per_plane = max(1, int(InterceptorNum))
        self.interceptorSpeed = InterceptorSpeed
        self.soft_penalty_scale = soft_penalty_scale

        self.position_scale = 25000.0
        self.speed_scale = 1000.0

        self._init_static_entities(missileList, aircrafts)

    # ------------------------------------------------------------------
    # core environment lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        missile_list, aircraft_list, _, _, _, missile_speed = reset_para(
            num_missiles=self.missileNum,
            StepNum=self.spaceSize,
            num_planes=self.num_planes,
        )
        self.missileSpeed = missile_speed
        self._init_static_entities(missile_list, aircraft_list)

        self.escapeFlag = -1
        self.info = "Go on Combating..."
        self.t = 0

        return self._build_state_dict(), self.escapeFlag, self.info

    def step(self, actions: Union[Dict[int, int], Sequence[int]]):
        action_map = self._canonicalize_actions(actions)
        self.escapeFlag = -1
        self.info = "Go on Combating..."

        for plane_id in range(self.num_planes):
            action_idx = action_map.get(plane_id, 0)
            self._apply_plane_action(plane_id, action_idx)

        intercept_events = self._advance_entities()
        rewards = self._compute_rewards(intercept_events)

        self.t += 1
        if self.t >= self.spaceSize and self.escapeFlag == -1:
            self.escapeFlag = 1
            self.info = "Maneuver Success"

        state = self._build_state_dict()
        info: Dict[str, Any] = {"intercepts": intercept_events}
        if self.escapeFlag != -1:
            info["metrics"] = self.collect_episode_metrics()
        return state, rewards, self.escapeFlag, info

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------
    def _init_static_entities(self, missile_list: List[Missiles], aircrafts: Sequence[Aircraft]):
        self.missileList: List[Missiles] = missile_list
        self.aircrafts: List[Aircraft] = list(aircrafts)
        self.num_planes = len(self.aircrafts)
        self.total_interceptors = self.interceptor_per_plane * self.num_planes

        self.observation_planes = np.zeros((self.num_planes, 7), dtype=np.float32)
        self.observation_missiles = np.zeros((self.missileNum, 7), dtype=np.float32)
        self.observation_interceptors = np.zeros((self.total_interceptors, 8), dtype=np.float32)

        self.aircraft_alive = [True for _ in range(self.num_planes)]

        self.interceptorList: List[Interceptor] = []
        self.interceptor_owners: List[int] = []
        self.interceptor_targets = np.full((self.total_interceptors,), -1, dtype=np.int32)

        for plane_id, plane in enumerate(self.aircrafts):
            for _ in range(self.interceptor_per_plane):
                interceptor = Interceptor(
                    [plane.X, plane.Y, plane.Z],
                    plane.V,
                    plane.Pitch,
                    plane.Heading,
                )
                interceptor.owner_plane_id = plane_id
                self.interceptorList.append(interceptor)
                self.interceptor_owners.append(plane_id)

        self.interceptorList = list(self.interceptorList)

        self.interceptor_remain = np.full((self.num_planes,), self.interceptor_per_plane, dtype=np.int32)
        self.lanchTime = np.zeros((self.num_planes,), dtype=np.int32)
        self.missile_targets = np.random.randint(0, self.num_planes, size=self.missileNum)
        for idx, target in enumerate(self.missile_targets):
            self.missileList[idx].target_plane_id = int(target)
        self.missile_lock_counts = np.zeros((self.missileNum,), dtype=np.int32)

        self.escapeFlag = -1
        self.info = "Go on Combating..."
        self.t = 0
        self.At_1 = np.full((self.num_planes,), -1, dtype=np.int32)

        self.last_missile_distances = np.zeros((self.num_planes, self.missileNum), dtype=np.float32)
        self.prev_missile_distances = np.zeros((self.num_planes, self.missileNum), dtype=np.float32)
        self.last_interceptor_distances = np.zeros((self.total_interceptors,), dtype=np.float32)
        self.prev_interceptor_distances = np.zeros((self.total_interceptors,), dtype=np.float32)
        self._refresh_all_observations()

        self.soft_penalty_accumulator = np.zeros((self.num_planes,), dtype=np.float32)
        self._reset_episode_metrics()

        sample_obs = self._build_plane_obs(0)
        self.single_obs_size = sample_obs.shape[0]

    def _reset_episode_metrics(self) -> None:
        self.episode_metrics: Dict[str, Any] = {
            "launch_attempts": 0,
            "invalid_launches": 0,
            "successful_launches": 0,
            "invalid_reasons": defaultdict(int),
            "cooperative_intercepts": 0,
            "intercept_lock_counts": [],
            "unprotected_threat_steps": 0,
            "episode_length": 0,
            "min_threat_distance": float("inf"),
            "plane_survival_time": [-1.0 for _ in range(self.num_planes)],
            "plane_destroyed": [False for _ in range(self.num_planes)],
            "plane_launch_counts": [0 for _ in range(self.num_planes)],
        }
        self.metrics_finalized = False

    def _finalize_episode_metrics(self) -> None:
        if self.metrics_finalized:
            return
        for plane_id in range(self.num_planes):
            if self.episode_metrics["plane_survival_time"][plane_id] < 0:
                self.episode_metrics["plane_survival_time"][plane_id] = float(self.t)
        self.episode_metrics["episode_length"] = int(self.t)
        if self.episode_metrics["min_threat_distance"] == float("inf"):
            self.episode_metrics["min_threat_distance"] = 0.0
        self.metrics_finalized = True

    def collect_episode_metrics(self) -> Dict[str, Any]:
        self._finalize_episode_metrics()
        metrics_copy: Dict[str, Any] = {
            "launch_attempts": int(self.episode_metrics["launch_attempts"]),
            "invalid_launches": int(self.episode_metrics["invalid_launches"]),
            "successful_launches": int(self.episode_metrics["successful_launches"]),
            "invalid_reasons": dict(self.episode_metrics["invalid_reasons"]),
            "cooperative_intercepts": int(self.episode_metrics["cooperative_intercepts"]),
            "intercept_lock_counts": list(self.episode_metrics["intercept_lock_counts"]),
            "unprotected_threat_steps": int(self.episode_metrics["unprotected_threat_steps"]),
            "episode_length": int(self.episode_metrics["episode_length"]),
            "min_threat_distance": float(self.episode_metrics["min_threat_distance"]),
            "plane_survival_time": list(self.episode_metrics["plane_survival_time"]),
            "plane_launch_counts": list(self.episode_metrics["plane_launch_counts"]),
        }
        return metrics_copy

    # ------------------------------------------------------------------
    # action decoding and execution
    # ------------------------------------------------------------------
    def _canonicalize_actions(self, actions: Union[Dict[int, int], Sequence[int]]):
        if isinstance(actions, dict):
            return {int(k): int(v) for k, v in actions.items()}
        try:
            return {idx: int(value) for idx, value in enumerate(actions)}
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("Unsupported action container") from exc

    def valid_action_mask(self, plane_id: int) -> np.ndarray:
        mask = np.ones((self.action_space.n,), dtype=np.bool_)
        for action_idx in range(self.action_space.n):
            launch_idx = action_idx % (self.missileNum + 1)
            if launch_idx >= self.missileNum:
                continue
            can_launch, _ = self._can_launch_interceptor(plane_id, launch_idx)
            if not can_launch:
                mask[action_idx] = False
        return mask

    def _is_geometrically_feasible(self, plane_id: int, missile_idx: int) -> bool:
        if missile_idx < 0 or missile_idx >= self.missileNum:
            return False
        plane = self.aircrafts[plane_id]
        missile = self.missileList[missile_idx]
        missile_pos = [missile.X, missile.Y, missile.Z]
        if not plane.LimitCondition(missile_pos):
            return False
        plane_pos = [plane.X, plane.Y, plane.Z]
        angle = CalAngle(missile_pos, plane_pos, plane.Heading)
        if angle > MaxInterceptorAngle:
            return False
        distance = CalDistance(plane_pos, missile_pos)
        if distance < MinInterceptorDist or distance > MaxInterceptorDist:
            return False
        return True

    def _can_launch_interceptor(self, plane_id: int, missile_idx: int) -> Tuple[bool, Optional[str]]:
        if missile_idx < 0 or missile_idx >= self.missileNum:
            return False, "invalid_index"

        missile = self.missileList[missile_idx]
        if not missile.attacking:
            return False, "inactive"

        target_plane = self.missile_targets[missile_idx]
        if target_plane < 0 or not self.aircraft_alive[int(target_plane)]:
            return False, "invalid_target"

        if self.interceptor_remain[plane_id] <= 0:
            return False, "empty"

        if abs(self.t - self.lanchTime[plane_id]) < LanchGap:
            return False, "cooldown"

        ready_interceptor = any(
            self.interceptor_owners[idx] == plane_id and interceptor.attacking == -1
            for idx, interceptor in enumerate(self.interceptorList)
        )
        if not ready_interceptor:
            return False, "no_ready"

        if not self.LockConstraint(missile_idx):
            return False, "lock_limit"

        if not self._is_geometrically_feasible(plane_id, missile_idx):
            return False, "geometry"

        return True, None

    def _count_active_locks_for_missile(self, missile_idx: int) -> Tuple[int, Dict[int, int]]:
        owner_counts: Dict[int, int] = defaultdict(int)
        total = 0
        for idx, target in enumerate(self.interceptor_targets):
            if target != missile_idx:
                continue
            interceptor = self.interceptorList[idx]
            if interceptor.attacking != 0:
                continue
            owner = self.interceptor_owners[idx]
            owner_counts[owner] += 1
            total += 1
        return total, owner_counts

    def _update_missile_lock_counts(self) -> None:
        self.missile_lock_counts.fill(0)
        for idx, target in enumerate(self.interceptor_targets):
            if target < 0 or target >= self.missileNum:
                continue
            interceptor = self.interceptorList[idx]
            if interceptor.attacking != 0:
                continue
            self.missile_lock_counts[target] += 1

    def _apply_plane_action(self, plane_id: int, action_idx: int) -> None:
        plane = self.aircrafts[plane_id]
        action_idx = int(np.clip(action_idx, 0, self.action_space.n - 1))
        maneuver_idx = action_idx // (self.missileNum + 1)
        launch_idx = action_idx % (self.missileNum + 1)
        interceptor_goal = launch_idx if launch_idx < self.missileNum else -1

        nx, ny, roll, pitch_constraint = self.action_dep[maneuver_idx]

        if not plane.speed_constraint(nx):
            nx = m.sin(plane.Pitch)

        if not plane.action_constraint(pitch_constraint):
            pitch_constraint = -1

        plane.AircraftPostition(None, nx, ny, roll, pitch_constraint)
        self.observation_planes[plane_id] = np.array(
            [plane.X, plane.Y, plane.Z, plane.Pitch, plane.Heading, plane.roll, plane.V],
            dtype=np.float32,
        )

        if interceptor_goal >= 0:
            self.LanchPolicy(plane_id, interceptor_goal)

    # ------------------------------------------------------------------
    # entity updates
    # ------------------------------------------------------------------
    def _advance_entities(self) -> List[InterceptEvent]:
        intercept_events: List[InterceptEvent] = []
        min_threat_distance = float("inf")

        for missile_idx, missile in enumerate(self.missileList):
            if not missile.attacking:
                missile.target_plane_id = -1
                self.observation_missiles[missile_idx] = np.array(
                    [missile.X, missile.Y, missile.Z, missile.Pitch, missile.Heading, -1.0, -1.0],
                    dtype=np.float32,
                )
                self.last_missile_distances[:, missile_idx] = 0.0
                self.prev_missile_distances[:, missile_idx] = 0.0
                continue

            target_plane = self.missile_targets[missile_idx]
            if target_plane < 0 or not self.aircraft_alive[target_plane]:
                missile.target_plane_id = -1
                self.missile_targets[missile_idx] = -1
                self.observation_missiles[missile_idx] = np.array(
                    [missile.X, missile.Y, missile.Z, missile.Pitch, missile.Heading, -1.0, -1.0],
                    dtype=np.float32,
                )
                self.last_missile_distances[:, missile_idx] = 0.0
                self.prev_missile_distances[:, missile_idx] = 0.0
                continue

            plane = self.aircrafts[target_plane]
            plane_pos = [plane.X, plane.Y, plane.Z]
            missile_pos = missile.MissilePosition(plane_pos, plane.V, plane.Pitch, plane.Heading)
            self.observation_missiles[missile_idx] = np.array(
                [missile_pos[0], missile_pos[1], missile_pos[2], missile.Pitch, missile.Heading, 1.0, float(target_plane)],
                dtype=np.float32,
            )
            missile.target_plane_id = int(target_plane)

            dist = CalDistance(plane_pos, missile_pos)
            previous = self.last_missile_distances[target_plane, missile_idx]
            if previous > 0:
                self.prev_missile_distances[target_plane, missile_idx] = previous
            else:
                self.prev_missile_distances[target_plane, missile_idx] = dist
            self.last_missile_distances[target_plane, missile_idx] = dist

            if dist < min_threat_distance:
                min_threat_distance = dist

            if dist < MISSILE_HIT_DISTANCE:
                self.escapeFlag = 0
                self.info = "Hit on! Escape Fail!!"
                self.aircraft_alive[target_plane] = False
                missile.attacking = False
                missile.target_plane_id = -1
                self.missile_targets[missile_idx] = -1
                self.last_missile_distances[:, missile_idx] = 0.0
                self.prev_missile_distances[:, missile_idx] = 0.0
                self.episode_metrics["plane_destroyed"][target_plane] = True
                self.episode_metrics["plane_survival_time"][target_plane] = float(self.t)

        for interceptor_idx, interceptor in enumerate(self.interceptorList):
            owner = self.interceptor_owners[interceptor_idx]
            plane = self.aircrafts[owner]
            target_idx = self.interceptor_targets[interceptor_idx]

            if interceptor.attacking == -1:
                interceptor.sync_with_aircraft(
                    [plane.X, plane.Y, plane.Z],
                    plane.Pitch,
                    plane.Heading,
                    plane.V,
                )
                self.observation_interceptors[interceptor_idx] = np.array(
                    [interceptor.X_i, interceptor.Y_i, interceptor.Z_i, interceptor.Pitch_i, interceptor.Heading_i, -1.0, float(owner), -1.0],
                    dtype=np.float32,
                )
                self.last_interceptor_distances[interceptor_idx] = 0.0
                self.prev_interceptor_distances[interceptor_idx] = 0.0
                continue

            if interceptor.attacking == 1:
                self.observation_interceptors[interceptor_idx] = np.array(
                    [interceptor.X_i, interceptor.Y_i, interceptor.Z_i, interceptor.Pitch_i, interceptor.Heading_i, 1.0, float(owner), float(target_idx)],
                    dtype=np.float32,
                )
                self.last_interceptor_distances[interceptor_idx] = 0.0
                self.prev_interceptor_distances[interceptor_idx] = 0.0
                continue

            if target_idx < 0 or target_idx >= self.missileNum:
                interceptor.attacking = 1
                self.observation_interceptors[interceptor_idx] = np.array(
                    [interceptor.X_i, interceptor.Y_i, interceptor.Z_i, interceptor.Pitch_i, interceptor.Heading_i, 1.0, float(owner), -1.0],
                    dtype=np.float32,
                )
                self.last_interceptor_distances[interceptor_idx] = 0.0
                self.prev_interceptor_distances[interceptor_idx] = 0.0
                continue

            missile = self.missileList[target_idx]
            if not missile.attacking:
                interceptor.attacking = 1
                self.observation_interceptors[interceptor_idx] = np.array(
                    [interceptor.X_i, interceptor.Y_i, interceptor.Z_i, interceptor.Pitch_i, interceptor.Heading_i, 1.0, float(owner), float(target_idx)],
                    dtype=np.float32,
                )
                self.last_interceptor_distances[interceptor_idx] = 0.0
                self.prev_interceptor_distances[interceptor_idx] = 0.0
                continue

            missile_pos = [missile.X, missile.Y, missile.Z]
            interceptor_pos = interceptor.InterceptorPosition(
                missile_pos,
                missile.V,
                missile.Pitch,
                missile.Heading,
            )
            self.observation_interceptors[interceptor_idx] = np.array(
                [interceptor_pos[0], interceptor_pos[1], interceptor_pos[2], interceptor.Pitch_i, interceptor.Heading_i, 0.0, float(owner), float(target_idx)],
                dtype=np.float32,
            )

            dist = CalDistance(interceptor_pos, missile_pos)
            previous_dist = self.last_interceptor_distances[interceptor_idx]
            if previous_dist > 0:
                self.prev_interceptor_distances[interceptor_idx] = previous_dist
            else:
                self.prev_interceptor_distances[interceptor_idx] = dist
            self.last_interceptor_distances[interceptor_idx] = dist
            if dist < INTERCEPT_SUCCESS_DISTANCE:
                lock_total, lock_by_owner = self._count_active_locks_for_missile(target_idx)
                target_plane_cached = self._cached_target_plane(target_idx)
                missile.attacking = False
                missile.target_plane_id = -1
                interceptor.attacking = 1
                self.interceptor_targets[interceptor_idx] = -1
                self.missile_targets[target_idx] = -1
                self.last_missile_distances[:, target_idx] = 0.0
                self.prev_missile_distances[:, target_idx] = 0.0
                self.last_interceptor_distances[interceptor_idx] = 0.0
                self.prev_interceptor_distances[interceptor_idx] = 0.0
                intercept_event = InterceptEvent(
                    owner,
                    target_idx,
                    target_plane=target_plane_cached,
                    lock_count=lock_total,
                    cooperative=target_plane_cached >= 0 and target_plane_cached != owner,
                    owner_lock_counts=dict(lock_by_owner),
                )
                intercept_events.append(intercept_event)
                self.episode_metrics["intercept_lock_counts"].append(lock_total)
                if intercept_event.cooperative:
                    self.episode_metrics["cooperative_intercepts"] += 1

        if all(not missile.attacking for missile in self.missileList):
            if self.escapeFlag == -1:
                self.escapeFlag = 2
                self.info = "Intercept Success"

        self._update_missile_lock_counts()

        if min_threat_distance < float("inf"):
            self.episode_metrics["min_threat_distance"] = min(
                self.episode_metrics["min_threat_distance"], min_threat_distance
            )
            if min_threat_distance > SAFE_DISTANCE_THRESHOLD and self.escapeFlag == -1:
                self.escapeFlag = 2
                self.info = "Threat Disengaged"

        unprotected = 0
        for missile_idx, missile in enumerate(self.missileList):
            if not missile.attacking:
                continue
            if self.missile_lock_counts[missile_idx] <= 0:
                unprotected += 1
        self.episode_metrics["unprotected_threat_steps"] += unprotected

        return intercept_events

    def _cached_target_plane(self, missile_idx: int) -> int:
        target = self.missile_targets[missile_idx]
        return int(target) if target >= 0 else -1

    # ------------------------------------------------------------------
    # reward computation
    # ------------------------------------------------------------------
    def _compute_rewards(self, intercept_events: Sequence[InterceptEvent]):
        rewards = {plane_id: 0.0 for plane_id in range(self.num_planes)}

        lock_counts_per_plane = [0 for _ in range(self.num_planes)]
        for idx, target_idx in enumerate(self.interceptor_targets):
            if target_idx < 0 or target_idx >= self.missileNum:
                continue
            interceptor = self.interceptorList[idx]
            if interceptor.attacking != 0:
                continue
            owner = self.interceptor_owners[idx]
            lock_counts_per_plane[owner] += 1
            prev_dist = float(self.prev_interceptor_distances[idx])
            curr_dist = float(self.last_interceptor_distances[idx])
            if prev_dist > 0 and curr_dist > 0:
                delta = prev_dist - curr_dist
                if delta != 0:
                    rewards[owner] += INTERCEPT_APPROACH_SCALE * delta

        for plane_id, plane in enumerate(self.aircrafts):
            if not self.aircraft_alive[plane_id]:
                rewards[plane_id] -= 10.0
                continue

            rewards[plane_id] += 0.2 * self.heightReward(plane.Y)

            for missile_idx, missile in enumerate(self.missileList):
                if not missile.attacking:
                    continue
                target_plane = self.missile_targets[missile_idx]
                plane_pos = [plane.X, plane.Y, plane.Z]
                missile_pos = [missile.X, missile.Y, missile.Z]
                dist = CalDistance(plane_pos, missile_pos)

                if target_plane == plane_id:
                    prev = float(self.prev_missile_distances[plane_id, missile_idx])
                    if prev > 0:
                        delta = prev - dist
                        if delta != 0:
                            rewards[plane_id] += APPROACH_REWARD_SCALE * delta
                    danger_scale = DANGERSCALE
                    lock_count = self.missile_lock_counts[missile_idx]
                    if lock_count > 0:
                        danger_scale *= 0.5
                        rewards[plane_id] += LOCK_SUPPORT_REWARD * lock_count
                    if dist < DANGER_DISTANCE:
                        rewards[plane_id] -= danger_scale * (1.0 - dist / DANGER_DISTANCE)
                    if lock_count <= 0:
                        rewards[plane_id] -= UNPROTECTED_PENALTY
                else:
                    if dist < DANGER_DISTANCE:
                        rewards[plane_id] -= 0.1 * (1.0 - dist / DANGER_DISTANCE)

        for plane_id, lock_count in enumerate(lock_counts_per_plane):
            if lock_count > 0 and self.aircraft_alive[plane_id]:
                rewards[plane_id] += LOCK_MAINTENANCE_REWARD * lock_count

        for event in intercept_events:
            owner = event.owner_plane
            target_plane = event.target_plane
            if event.cooperative and target_plane >= 0:
                rewards[owner] += COOPERATIVE_OWNER_REWARD
                rewards[target_plane] += COOPERATIVE_TARGET_REWARD
            elif target_plane == owner:
                rewards[owner] += 3.0
            elif target_plane >= 0:
                rewards[owner] += 2.0
                rewards[target_plane] += 1.0
            else:
                rewards[owner] += 1.0

            if event.owner_lock_counts:
                for owner_id, count in event.owner_lock_counts.items():
                    if count > 1:
                        rewards[owner_id] -= REPEATED_LOCK_PENALTY * (count - 1)

        for missile_idx, missile in enumerate(self.missileList):
            if not missile.attacking:
                continue
            lock_total, owner_counts = self._count_active_locks_for_missile(missile_idx)
            for owner_id, count in owner_counts.items():
                if count > 1:
                    rewards[owner_id] -= REPEATED_LOCK_PENALTY * (count - 1)

        if self.escapeFlag == 2:
            if self.info == "Intercept Success":
                bonus = SPARSE_REWARD_SCALE
            else:
                bonus = THREAT_DISENGAGE_REWARD
            for plane_id in rewards:
                rewards[plane_id] += bonus
        elif self.escapeFlag == 1:
            for plane_id in rewards:
                rewards[plane_id] += TIMEOUT_REWARD
        elif self.escapeFlag == 0:
            for plane_id, alive in enumerate(self.aircraft_alive):
                if not alive:
                    rewards[plane_id] -= 5.0

        for plane_id in range(self.num_planes):
            penalty = float(self.soft_penalty_accumulator[plane_id])
            if penalty > 0:
                rewards[plane_id] -= penalty
                self.soft_penalty_accumulator[plane_id] = 0.0

        return rewards

    # ------------------------------------------------------------------
    # launch policy and lock constraints
    # ------------------------------------------------------------------
    def LanchPolicy(self, plane_id: int, interceptor_goal: int) -> bool:
        if interceptor_goal < 0 or interceptor_goal >= self.missileNum:
            return False

        self.episode_metrics["launch_attempts"] += 1

        can_launch, reason = self._can_launch_interceptor(plane_id, interceptor_goal)
        if not can_launch:
            self.episode_metrics["invalid_launches"] += 1
            if reason is not None:
                self.episode_metrics["invalid_reasons"][reason] += 1
            if reason in {"cooldown", "lock_limit"}:
                self.soft_penalty_accumulator[plane_id] += self.soft_penalty_scale
            return False

        owner_interceptors = [
            (idx, interceptor)
            for idx, interceptor in enumerate(self.interceptorList)
            if self.interceptor_owners[idx] == plane_id
        ]
        for idx, interceptor in owner_interceptors:
            if interceptor.attacking == -1:
                plane = self.aircrafts[plane_id]
                interceptor.sync_with_aircraft(
                    [plane.X, plane.Y, plane.Z],
                    plane.Pitch,
                    plane.Heading,
                    plane.V,
                )
                launch_speed = max(plane.V, self.interceptorSpeed)
                interceptor.begin_pursuit(interceptor_goal, launch_speed)
                self.interceptor_targets[idx] = interceptor_goal
                self.interceptor_remain[plane_id] -= 1
                self.lanchTime[plane_id] = self.t
                self.episode_metrics["successful_launches"] += 1
                self.episode_metrics["plane_launch_counts"][plane_id] += 1
                return True
        return False

    def LockConstraint(self, missile_idx: int) -> bool:
        if missile_idx < 0 or missile_idx >= self.missileNum:
            return False

        missile = self.missileList[missile_idx]
        if not missile.attacking:
            return False

        locked_num = 0
        for interceptor, target in zip(self.interceptorList, self.interceptor_targets):
            if interceptor.attacking == 1:
                continue
            if target == missile_idx and interceptor.attacking != -1:
                locked_num += 1

        active_missiles = self.getRemainMissileNum()
        if active_missiles <= 0:
            return False

        base_limit = max(1, m.ceil(self.total_interceptors / max(1, self.missileNum)))
        focus_bonus = max(0, self.missileNum - active_missiles)
        max_lock = min(self.total_interceptors, base_limit + focus_bonus)

        return locked_num < max_lock

    def getRemainMissileNum(self) -> int:
        return sum(1 for missile in self.missileList if missile.attacking)

    # ------------------------------------------------------------------
    # observation helpers
    # ------------------------------------------------------------------
    def _refresh_all_observations(self) -> None:
        for plane_id, plane in enumerate(self.aircrafts):
            self.observation_planes[plane_id] = np.array(
                [plane.X, plane.Y, plane.Z, plane.Pitch, plane.Heading, plane.roll, plane.V],
                dtype=np.float32,
            )

        for missile_idx, missile in enumerate(self.missileList):
            target = float(self.missile_targets[missile_idx]) if missile.attacking else -1.0
            active = 1.0 if missile.attacking else -1.0
            self.observation_missiles[missile_idx] = np.array(
                [missile.X, missile.Y, missile.Z, missile.Pitch, missile.Heading, active, target],
                dtype=np.float32,
            )

        for idx, interceptor in enumerate(self.interceptorList):
            owner = float(self.interceptor_owners[idx])
            target = float(self.interceptor_targets[idx])
            status = float(interceptor.attacking)
            self.observation_interceptors[idx] = np.array(
                [interceptor.X_i, interceptor.Y_i, interceptor.Z_i, interceptor.Pitch_i, interceptor.Heading_i, status, owner, target],
                dtype=np.float32,
            )

    def _build_state_dict(self) -> Dict[int, np.ndarray]:
        state_dict: Dict[int, np.ndarray] = {}
        for plane_id in range(self.num_planes):
            state_dict[plane_id] = self._build_plane_obs(plane_id)
        return state_dict

    def _build_plane_obs(self, plane_id: int) -> np.ndarray:
        features: List[float] = []

        features.extend(self._encode_plane_state(plane_id))
        for other_id in range(self.num_planes):
            if other_id == plane_id:
                continue
            features.extend(self._encode_plane_state(other_id))

        for missile_idx in range(self.missileNum):
            features.extend(self._encode_missile_state(missile_idx, plane_id))

        for interceptor_idx in range(self.total_interceptors):
            features.extend(self._encode_interceptor_state(interceptor_idx, plane_id))

        features.append(self.interceptor_remain[plane_id] / max(1, self.interceptor_per_plane))
        cooldown = min((self.t - self.lanchTime[plane_id]) / max(1, LanchGap), 1.0)
        features.append(cooldown)

        agent_one_hot = [0.0] * self.num_planes
        agent_one_hot[plane_id] = 1.0
        features.extend(agent_one_hot)

        return np.array(features, dtype=np.float32)

    def _encode_plane_state(self, plane_id: int) -> List[float]:
        plane_vec = self.observation_planes[plane_id]
        x, y, z, pitch, heading, roll, speed = plane_vec
        return [
            x / self.position_scale,
            y / self.position_scale,
            z / self.position_scale,
            pitch / m.pi,
            heading / m.pi,
            roll / m.pi,
            speed / self.speed_scale,
            1.0 if self.aircraft_alive[plane_id] else -1.0,
        ]

    def _encode_missile_state(self, missile_idx: int, plane_id: int) -> List[float]:
        missile_vec = self.observation_missiles[missile_idx]
        x, y, z, pitch, heading, active, target = missile_vec
        active_flag = 1.0 if active > 0 else -1.0
        target_plane = int(target)
        target_one_hot = [0.0] * self.num_planes
        if 0 <= target_plane < self.num_planes:
            target_one_hot[target_plane] = 1.0
        lock_count = float(self.missile_lock_counts[missile_idx]) / max(1, self.total_interceptors)
        locked_flag = 1.0 if self.missile_lock_counts[missile_idx] > 0 else -1.0
        return [
            x / self.position_scale,
            y / self.position_scale,
            z / self.position_scale,
            pitch / m.pi,
            heading / m.pi,
            active_flag,
            *target_one_hot,
            lock_count,
            locked_flag,
        ]

    def _encode_interceptor_state(self, interceptor_idx: int, plane_id: int) -> List[float]:
        interceptor_vec = self.observation_interceptors[interceptor_idx]
        x, y, z, pitch, heading, status, owner, target = interceptor_vec
        owner_id = int(owner)
        owner_one_hot = [0.0] * self.num_planes
        if 0 <= owner_id < self.num_planes:
            owner_one_hot[owner_id] = 1.0
        missile_one_hot = [0.0] * (self.missileNum + 1)
        target_idx = int(target)
        if 0 <= target_idx < self.missileNum:
            missile_one_hot[target_idx] = 1.0
        else:
            missile_one_hot[-1] = 1.0

        status_norm = -1.0 if status < 0 else (1.0 if status > 0 else 0.0)

        return [
            x / self.position_scale,
            y / self.position_scale,
            z / self.position_scale,
            pitch / m.pi,
            heading / m.pi,
            status_norm,
            *owner_one_hot,
            *missile_one_hot,
        ]

    # ------------------------------------------------------------------
    # utility API retained for compatibility
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            (self.observation_planes, self.observation_missiles, self.observation_interceptors)
        )

    def _get_actSpace(self) -> int:
        return self.action_space.n

    def _get_stateSpace(self):  # pragma: no cover - legacy
        obs = self._get_obs()
        obs = obs.flatten()
        return obs.shape

    def _getNewStateSpace(self):
        return (self.single_obs_size,)

    def normalizeState(self, state, reverse=False):  # pragma: no cover - compatibility hook
        if reverse:
            state[:, 0:3] = state[:, 0:3] * self.position_scale
            state[:, 3:6] = state[:, 3:6] * m.pi
        else:
            state[:, 0:3] = state[:, 0:3] / self.position_scale
            state[:, 3:6] = state[:, 3:6] / m.pi
        return state

    def heightReward(self, h: float) -> float:
        safe_min = 8000
        safe_max = 12000
        tolerance = 1000
        hard_min = safe_min - tolerance
        hard_max = safe_max + tolerance

        if h < hard_min or h > hard_max:
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

    def TreatReward(self):  # pragma: no cover - compatibility
        treat = 0
        for i in range(self.missileNum):
            missile = self.missileList[i]
            if not missile.attacking:
                continue
            target = self.missile_targets[i]
            if target < 0 or not self.aircraft_alive[target]:
                continue
            plane = self.aircrafts[target]
            treat += CalTreat(
                [plane.X, plane.Y, plane.Z],
                [missile.X, missile.Y, missile.Z],
                plane.V,
                missile.V,
            )
        return treat

    def render(self):  # pragma: no cover - not implemented
        pass

