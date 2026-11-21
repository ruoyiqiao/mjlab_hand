"""Palm-up spinning tasks on every hand: Baoding balls and plate spinning on the fingertips.

* ``Baoding-<Hand>``: two balls rest in the palm-up hand; the policy rotates the pair about the palm normal in a
  commanded direction, sampled per episode (clockwise or counter-clockwise). ``Baoding-Sharpa-Tilt`` pitches the hand
  fingers-down by ``hand_tilt``; ``Baoding-Sharpa-TiltThumb`` also rolls it toward the thumb by ``hand_roll``.
* ``PlateSpin-<Hand>``: a dessert plate (flat well, sloped rim, foot ring) rests on the fingertips and the policy spins
  it about its own normal in the commanded direction. Only the last phalanges may touch the plate: other hand links
  are penalised and palm contact ends the episode. The spin reward is scaled by the share of a three-finger support.
  The start state is measured per hand (tools/allegro4/plate_probe.py -> hand_params/plate.json): a finger pose whose
  fingertips alone carry the plate, and the pose the plate settles to on them, so an episode starts at rest.

Both: the hand is the palm-up floating hand of the juggling tasks (measured anchor per hand,
juggling/juggle_env_cfg.py), fixed in space with its wrist locked, so the objects are moved by the fingers only.
Fingers use delta actions (zero action holds the start pose). Reward = signed rotation rate in the commanded direction
(capped); an episode ends when an object leaves the hand; metric = turns per episode (``Metrics/spin/turns``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_hand.juggling.juggle_env_cfg import (
    ANCHOR,
    CUP_POSE,
    PALM_UP,
    hand_juggle_params,
    make_ball_spec,
)
from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT
from mjlab_hand.tasks_common import mdp as tmdp
from mjlab_hand.tasks_common.env_builder import (
    hand_joint_cfg,
    make_action_cfg,
    make_hand_entity_cfg,
    make_sim_cfg,
)
from mjlab_hand.tasks_common.hand_profile import get_profile
from mjlab_hand.tasks_common.hand_tasks import HAND_SUFFIX
from mjlab_hand.tasks_common.rl import make_ppo_cfg
from mjlab_hand.tasks_common.workspace import BASE_JOINTS, HandRootWorkspace

HANDS = ("allegro", "leap", "wuji", "sharpa", "xhand", "shadow")
ALLEGRO_CUP_REL_PALM = (-0.0475, 0.0, 0.0371)  # juggling command's palm-face rule for Allegro
STIFF_BASE_GAINS = (2000.0, 90.0, 300.0)  # the juggling root PD: holds the locked base in place
PLATE_PARAMS = Path(__file__).resolve().parent / "tasks_common/hand_params/plate.json"
# Allegro's last phalanges: the distal links and their jointless tip bodies
ALLEGRO_TIP_BODIES = ("ff_distal", "mf_distal", "rf_distal", "th_distal")
ALLEGRO_TIP_BODIES += tuple(b.replace("distal", "tip") for b in ALLEGRO_TIP_BODIES)


@dataclass
class PalmTaskSpec:
    task: str  # "baoding" | "plate"
    hand: str = "allegro"
    episode_s: float = 10.0
    hand_tilt: float = 0.0  # rad, the palm-up anchor pitched fingers-down
    hand_roll: float = 0.0  # rad, the anchor rolled about the finger axis (> 0 lifts the +y side)
    ball_radius: float = 0.02
    ball_mass: float = 0.04
    ball_half_sep: float = 0.022  # balls start at the cup point +- this along ball_axis_deg
    ball_axis_deg: float = 90.0  # pair direction in the palm plane: 0 = along the fingers
    # dessert plate: flat well, rim rising to the edge, foot ring underneath (m, kg)
    plate_radius: float = 0.09
    plate_well_radius: float = 0.055
    plate_rim_rise: float = 0.012
    plate_thickness: float = 0.003
    plate_foot_radius: float = 0.045
    plate_foot_height: float = 0.003
    plate_mass: float = 0.15
    max_rate: float = 6.0  # rad/s cap of the rewarded rotation rate
    reward_spin: float = 0.05
    reward_center: float = -0.5  # Baoding: distance of the ball centroid from the cup point (m)
    reward_tilt: float = -0.2  # plate: 1 - cos(plate tilt)
    reward_nontip: float = -1.0  # plate: a hand link other than a last phalanx touches the plate
    reward_support: float = 0.01  # plate: min(fingers carrying it, 3) / 3
    reward_drop: float = -5.0
    reward_alive: float = 0.01
    reward_action_rate: float = -1e-3


# ---------------------------------------------------------------------- per-hand geometry
def plate_params(hand: str) -> dict | None:
    """Measured fingertip-support start state (tools/allegro4/plate_probe.py); None if not measured."""
    if not PLATE_PARAMS.exists():
        return None
    return json.loads(PLATE_PARAMS.read_text()).get(hand)


def distal_bodies(hand: str) -> tuple[str, ...]:
    """One body per finger whose subtree is that finger's last phalanx (counts the fingers carrying
    the plate). Allegro's and XHand's tip bodies are jointless children of the distal link."""
    if hand == "allegro":
        return ALLEGRO_TIP_BODIES[:4]
    tips = tuple(get_profile(hand).tip_bodies)
    return tuple(t.replace("_tip", "_link2") for t in tips) if hand == "xhand" else tips


def fingertip_bodies(hand: str) -> tuple[str, ...]:
    """Every body of the last phalanges (allowed to touch the plate). XHand's distal link ``*_link2``
    carries a tip pad ``*_tip`` and a rigid back cover ``*back_link2``."""
    if hand == "allegro":
        return ALLEGRO_TIP_BODIES
    if hand == "xhand":
        tips = tuple(get_profile(hand).tip_bodies)
        return tips + tuple(t.replace("_tip", s) for t in tips for s in ("_link2", "back_link2"))
    return distal_bodies(hand)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def tilt_matrix(tilt: float, roll: float = 0.0) -> np.ndarray:
    """World rotation of the tilted anchor: ``roll`` about the finger axis (+x), then ``tilt`` about
    +y, so the fingers point down by ``tilt``."""
    c, s = math.cos(tilt), math.sin(tilt)
    cr, sr = math.cos(roll), math.sin(roll)
    pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return pitch @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])


def tilt_quat(tilt: float, roll: float = 0.0) -> tuple[float, float, float, float]:
    """wxyz quaternion of tilt_matrix(tilt, roll)."""
    return _qmul(
        (math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0),
        (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0),
    )


def rest_rel_palm(sp: PalmTaskSpec) -> tuple[float, float, float]:
    """Object rest point relative to the palm origin (world axes): the ball cup or the plate centre."""
    if sp.task == "plate":
        pp = plate_params(sp.hand)
        if pp is not None:
            return tuple(pp["plate_rel_palm"])
    if sp.hand == "allegro":
        cup = ALLEGRO_CUP_REL_PALM
    else:
        cup = hand_juggle_params(sp.hand)["cup_rel_palm"]
    return tuple(float(x) for x in tilt_matrix(sp.hand_tilt, sp.hand_roll) @ np.asarray(cup))


# ---------------------------------------------------------------------- command: direction + rotation tracking
@dataclass(kw_only=True)
class SpinCommandCfg(CommandTermCfg):
    spec: PalmTaskSpec
    palm_body: str = "palm"
    object_names: tuple[str, ...] = ("ball_a", "ball_b")
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False

    def build(self, env):
        return SpinCommand(self, env)


class SpinCommand(CommandTerm):
    """Samples the spin direction per episode, places the objects and measures their signed rotation."""

    cfg: SpinCommandCfg

    def __init__(self, cfg: SpinCommandCfg, env):
        super().__init__(cfg, env)
        sp, n, dev = cfg.spec, self.num_envs, self.device
        self.sp = sp
        self.hand = env.scene["hand"]
        self.objs = [env.scene[o] for o in cfg.object_names]
        self.palm_id = self.hand.find_bodies(cfg.palm_body)[0][0]
        m = env.sim.mj_model
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand/{cfg.palm_body}")
        self.rest_offset = torch.tensor(rest_rel_palm(sp), dtype=torch.float32, device=dev)
        self.rest_local = torch.tensor(d.xpos[pid], dtype=torch.float32, device=dev)
        self.rest_local += self.rest_offset
        # palm frame of the (tilted) anchor: e1 along the fingers, e2 lateral, up = palm normal
        R = torch.tensor(tilt_matrix(sp.hand_tilt, sp.hand_roll), dtype=torch.float32, device=dev)
        self.e1, self.e2, self.up = R[:, 0], R[:, 1], R[:, 2]
        pp = plate_params(sp.hand) if sp.task == "plate" else None
        self.plate_quat = torch.tensor(pp["plate_quat"] if pp else (1.0, 0.0, 0.0, 0.0), device=dev)
        self.dt = float(env.step_dt)
        self.direction = torch.ones(n, device=dev)
        self.prev_angle = torch.zeros(n, device=dev)
        self.progress = torch.zeros(n, device=dev)  # rad this step, in the commanded direction
        self.turns = torch.zeros(n, device=dev)
        self.metrics["turns"] = self.turns
        self.metrics["rate"] = torch.zeros(n, device=dev)
        if sp.task == "plate":
            self.metrics["nontip_contact"] = torch.zeros(n, device=dev)
            self.metrics["tips_in_contact"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self.direction.unsqueeze(-1)

    # object state -----------------------------------------------------------------------------------------------
    def obj_pos(self) -> torch.Tensor:
        """(N, K, 3) env-local object positions."""
        o = self._env.scene.env_origins
        return torch.stack([b.data.root_link_pos_w - o for b in self.objs], 1)

    def rest_point(self) -> torch.Tensor:
        """(N, 3) env-local rest point of the objects."""
        palm = self.hand.data.body_link_pos_w[:, self.palm_id] - self._env.scene.env_origins
        return palm + self.rest_offset

    def plate_normal(self) -> torch.Tensor:
        """(N, 3) world z axis of the plate body."""
        w, x, y, z = self.objs[0].data.root_link_quat_w.unbind(-1)
        return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)

    def angle(self) -> torch.Tensor:
        """Baoding: heading of ball_a -> ball_b in the palm plane."""
        p = self.obj_pos()
        v = p[:, 1] - p[:, 0]
        return torch.atan2((v * self.e2).sum(-1), (v * self.e1).sum(-1))

    def _found(self, sensor: str) -> torch.Tensor:
        return self._env.scene[sensor].data.found.reshape(self.num_envs, -1) > 0

    def tip_contacts(self) -> torch.Tensor:
        """(N, fingers) the last phalanx of each finger touches the plate."""
        return self._found("plate_tip_contact")

    def nontip_contact(self) -> torch.Tensor:
        return self._found("plate_nontip_contact").any(-1)

    def palm_contact(self) -> torch.Tensor:
        return self._found("plate_palm_contact").any(-1)

    def dropped(self) -> torch.Tensor:
        p = self.obj_pos()
        rest = self.rest_point()
        far = (torch.linalg.norm(p[..., :2] - rest[:, None, :2], dim=-1) > 0.12).any(-1)
        if self.sp.task == "plate":
            low = (p[..., 2] < rest[:, None, 2] - 0.04).any(-1)
            tipped = self.plate_normal()[:, 2] < 0.5
            return low | far | tipped | self.palm_contact()
        low = (p[..., 2] < rest[:, None, 2] - 0.06).any(-1)
        return low | far

    # manager hooks ----------------------------------------------------------------------------------------------
    def _update_metrics(self) -> None:
        if self.sp.task == "baoding":
            a = self.angle()
            d = torch.remainder(a - self.prev_angle + math.pi, 2 * math.pi) - math.pi
            self.prev_angle = a
        else:
            w = self.objs[0].data.root_link_ang_vel_w
            d = (w * self.plate_normal()).sum(-1) * self.dt
            # running means over the episode (logged at reset)
            k = self._env.episode_length_buf.float().clamp(min=1.0)
            for key, v in (
                ("nontip_contact", self.nontip_contact().float()),
                ("tips_in_contact", self.tip_contacts().float().sum(-1)),
            ):
                self.metrics[key] += (v - self.metrics[key]) / k
        self.progress = self.direction * d
        self.turns += self.progress / (2 * math.pi)
        self.metrics["turns"] = self.turns
        self.metrics["rate"] = self.progress / self.dt

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        ccw = torch.rand(len(env_ids), device=self.device) < 0.5
        self.direction[env_ids] = torch.where(ccw, 1.0, -1.0)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def reset_objects(self, env_ids: torch.Tensor) -> None:
        n, sp = len(env_ids), self.sp
        origin = self._env.scene.env_origins[env_ids]
        q = torch.zeros(n, 4, device=self.device)
        q[:, 0] = 1.0
        zero = torch.zeros(n, 6, device=self.device)
        if sp.task == "baoding":
            psi = math.radians(sp.ball_axis_deg)
            u = math.cos(psi) * self.e1 + math.sin(psi) * self.e2
            centre = self.rest_local + 0.004 * self.up
            for k, b in enumerate(self.objs):
                p = origin + centre + (2 * k - 1) * sp.ball_half_sep * u
                b.write_root_link_pose_to_sim(torch.cat([p, q], -1), env_ids=env_ids)
                b.write_root_link_velocity_to_sim(zero, env_ids=env_ids)
            self.prev_angle[env_ids] = psi  # heading of ball_a -> ball_b
        else:  # the measured settled pose, 1 mm higher: the fingers start unloaded at their targets
            p = origin + self.rest_local + torch.tensor([0.0, 0.0, 0.001], device=self.device)
            pose = torch.cat([p, self.plate_quat.expand(n, 4)], -1)
            self.objs[0].write_root_link_pose_to_sim(pose, env_ids=env_ids)
            self.objs[0].write_root_link_velocity_to_sim(zero, env_ids=env_ids)
        # turns is not zeroed here: reset events run before the command manager logs and zeroes it
        self.progress[env_ids] = 0.0


def get_spin(env, name: str = "spin") -> SpinCommand:
    return env.command_manager.get_term(name)


def reset_spin(env, env_ids, command_name: str = "spin"):
    get_spin(env, command_name).reset_objects(env_ids)


# ---------------------------------------------------------------------- observations / rewards / terminations
def obs_objects(env, command_name: str = "spin"):
    t = get_spin(env, command_name)
    parts = [(t.obj_pos() - t.rest_point()[:, None]).flatten(1)]
    for b in t.objs:
        parts += [
            b.data.root_link_quat_w,
            0.2 * b.data.root_link_lin_vel_w,
            0.2 * b.data.root_link_ang_vel_w,
        ]
    return torch.cat(parts, -1)


def obs_tip_contacts(env, command_name: str = "spin"):
    return get_spin(env, command_name).tip_contacts().float()


def obs_direction(env, command_name: str = "spin"):
    return get_spin(env, command_name).command


def reward_support(env, command_name: str = "spin"):
    return get_spin(env, command_name).tip_contacts().float().sum(-1).clamp(max=3.0) / 3.0


def reward_spin(env, command_name: str = "spin"):
    """Rotation rate in the commanded direction; for the plate, only while fingertips carry it."""
    t = get_spin(env, command_name)
    r = (t.progress / t.dt).clamp(-t.sp.max_rate, t.sp.max_rate)
    if t.sp.task == "plate":
        r = r * reward_support(env, command_name)
    return r


def reward_center(env, command_name: str = "spin"):
    t = get_spin(env, command_name)
    return torch.linalg.norm(t.obj_pos().mean(1) - t.rest_point(), dim=-1)


def reward_tilt(env, command_name: str = "spin"):
    return 1.0 - get_spin(env, command_name).plate_normal()[:, 2].clamp(-1.0, 1.0)


def reward_nontip(env, command_name: str = "spin"):
    return get_spin(env, command_name).nontip_contact().float()


def reward_alive(env):
    return torch.ones(env.num_envs, device=env.device)


def object_dropped(env, command_name: str = "spin"):
    return get_spin(env, command_name).dropped()


# ---------------------------------------------------------------------- objects
def _revolve(profile: list[tuple[float, float]], n: int = 48) -> tuple[list, list]:
    """Closed (r, z) profile revolved about z: vertices and triangles (r = 0 points: one vertex)."""
    verts, idx = [], []
    for r, z in profile:
        if r == 0.0:
            idx.append([len(verts)] * n)
            verts.append((0.0, 0.0, z))
        else:
            idx.append(list(range(len(verts), len(verts) + n)))
            verts += [
                (r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z)
                for k in range(n)
            ]
    faces = []
    for a, b in zip(idx, idx[1:] + idx[:1]):
        for k in range(n):
            k2 = (k + 1) % n
            for tri in ((a[k], b[k], b[k2]), (a[k], b[k2], a[k2])):
                if len(set(tri)) == 3:
                    faces.append(tri)
    return verts, faces


def make_plate_spec(sp: PalmTaskSpec) -> mujoco.MjSpec:
    """Dessert plate. Collision: well disc + 16 sloped rim boxes + 16 foot-ring boxes (convex
    primitives; a plate is concave). Visual: the revolved profile (group 1, no collision)."""
    R, rw, rise, t = sp.plate_radius, sp.plate_well_radius, sp.plate_rim_rise, sp.plate_thickness
    rf, hf = sp.plate_foot_radius, sp.plate_foot_height
    h = t / 2
    prof = [
        (0.0, h),
        (rw, h),
        (R, h + rise),
        (R, h + rise - t),
        (rw + 0.002, -h),
        (rf + 0.003, -h),
        (rf + 0.003, -h - hf),
        (rf - 0.003, -h - hf),
        (rf - 0.003, -h),
        (0.0, -h),
    ]
    verts, faces = _revolve(prof)
    vtxt = " ".join(f"{c:.5f}" for v in verts for c in v)
    ftxt = " ".join(str(i) for f in faces for i in f)
    geoms = [f'<geom name="plate_collision" type="cylinder" size="{rw} {h}" {{c}}/>']
    n = 16
    slope = math.atan2(rise, R - rw)
    half_len = 0.5 * math.hypot(R - rw, rise)
    rm = 0.5 * (R + rw)
    for k in range(n):
        phi = 2 * math.pi * k / n
        qz = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))
        q = _qmul(qz, (math.cos(-slope / 2), 0.0, math.sin(-slope / 2), 0.0))  # +x rises outward
        qs = " ".join(f"{c:.6f}" for c in q)
        w = 1.15 * math.pi * rm / n
        geoms.append(
            f'<geom name="plate_rim{k}_collision" type="box" size="{half_len:.5f} {w:.5f} {h}" '
            f'pos="{rm * math.cos(phi):.5f} {rm * math.sin(phi):.5f} {rise / 2:.5f}" quat="{qs}" '
            "{c}/>"
        )
        wf = 1.15 * math.pi * rf / n
        geoms.append(
            f'<geom name="plate_foot{k}_collision" type="box" size="0.003 {wf:.5f} {hf / 2}" '
            f'pos="{rf * math.cos(phi):.5f} {rf * math.sin(phi):.5f} {-h - hf / 2:.5f}" '
            f'quat="{qz[0]:.6f} 0 0 {qz[3]:.6f}" {{c}}/>'
        )
    col = (
        'rgba="0.22 0.45 0.85 1" group="3" condim="4" friction="1.0 0.005 0.0005" priority="1" '
        'mass="0"'
    )
    body = "\n      ".join(g.format(c=col) for g in geoms)
    m = sp.plate_mass
    i_z = 0.85 * m * R * R / 2  # thin disc, mass slightly toward the centre (well + foot)
    xml = f"""
<mujoco model="plate">
  <asset><mesh name="plate_mesh" vertex="{vtxt}" face="{ftxt}"/></asset>
  <worldbody>
    <body name="plate">
      <freejoint name="plate_joint"/>
      <inertial pos="0 0 0.003" mass="{m}" diaginertia="{i_z / 2} {i_z / 2} {i_z}"/>
      <geom name="plate_visual" type="mesh" mesh="plate_mesh" rgba="0.22 0.45 0.85 1" group="1"
            contype="0" conaffinity="0" mass="0"/>
      {body}
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def plate_sensors(sp: PalmTaskSpec, prof) -> tuple[ContactSensorCfg, ...]:
    tips = fingertip_bodies(sp.hand)
    palm = prof.palm_body if prof else "palm"
    plate = ContactMatch(mode="subtree", pattern="plate", entity="plate")
    kw = dict(fields=("found",), reduce="netforce", num_slots=1)
    return (
        ContactSensorCfg(  # one primary per finger: the subtree of its last phalanx
            name="plate_tip_contact",
            secondary=plate,
            primary=ContactMatch(mode="subtree", pattern=distal_bodies(sp.hand), entity="hand"),
            **kw,
        ),
        ContactSensorCfg(
            name="plate_nontip_contact",
            secondary=plate,
            primary=ContactMatch(mode="body", pattern=".*", entity="hand", exclude=tips),
            **kw,
        ),
        ContactSensorCfg(
            name="plate_palm_contact",
            secondary=plate,
            primary=ContactMatch(mode="body", pattern=palm, entity="hand"),
            **kw,
        ),
    )


# ---------------------------------------------------------------------- env config
def _workspace(sp: PalmTaskSpec) -> HandRootWorkspace:
    if sp.hand == "allegro":
        side, rot = "left", PALM_UP
    else:
        prof = get_profile(sp.hand)
        side, rot = prof.side, prof.anchor_rot(down=False, fingers_px=True)
    if sp.hand_tilt or sp.hand_roll:
        rot = _qmul(tilt_quat(sp.hand_tilt, sp.hand_roll), rot)
    pos = ANCHOR if sp.hand == "allegro" else get_profile(sp.hand).anchor_pos(ANCHOR, rot)
    lock = (
        0.002  # the workspace needs non-empty bounds: +-2 mm / mrad hold every base joint in place
    )
    return HandRootWorkspace(
        side=side,
        entity_name="hand",
        anchor_pos=pos,
        anchor_rot=rot,
        joint_bounds={n: (-lock, lock) for n in BASE_JOINTS},
        lin_vel_max=1.0,
        ang_vel_max=3.0,
        reset_noise={n: (0.0, 0.0) for n in BASE_JOINTS},
        notes=f"palm-up {sp.hand} hand, {sp.task}, tilt {sp.hand_tilt} roll {sp.hand_roll}",
    )


def start_finger_pose(sp: PalmTaskSpec, prof) -> dict[str, float]:
    """Baoding: the measured juggling cup. Plate: the measured fingertip-support pose."""
    if sp.task == "baoding":
        return dict(CUP_POSE) if prof is None else dict(hand_juggle_params(sp.hand)["cup_pose"])
    pp = plate_params(sp.hand)
    if pp is not None:
        return dict(pp["finger_pose"])
    names = list(CUP_POSE) if prof is None else [*prof.finger_joints, *prof.coupled_joints]
    return {j: 0.0 for j in names}  # unmeasured hand (plate_probe.py): flat fingers


def make_palm_task_env_cfg(sp: PalmTaskSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
    prof = None if sp.hand == "allegro" else get_profile(sp.hand)
    ws = _workspace(sp)
    plate = sp.task == "plate"
    obj_names = ("plate",) if plate else ("ball_a", "ball_b")
    cmd = SpinCommandCfg(
        spec=sp, palm_body=prof.palm_body if prof else "palm", object_names=obj_names
    )
    jc = hand_joint_cfg("hand", prof)
    obs = {
        "joint_pos": ObservationTermCfg(func=tmdp.hand_joint_pos_unit, params={"asset_cfg": jc}),
        "joint_vel": ObservationTermCfg(func=tmdp.hand_joint_vel, params={"asset_cfg": jc}),
        "objects": ObservationTermCfg(func=obs_objects),
        "direction": ObservationTermCfg(func=obs_direction),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    if plate:
        obs["tip_contacts"] = ObservationTermCfg(func=obs_tip_contacts)
    observations = {
        "actor": ObservationGroupCfg(dict(obs), enable_corruption=False),
        "critic": ObservationGroupCfg(dict(obs), enable_corruption=False),
    }
    rewards = {
        "spin": RewardTermCfg(func=reward_spin, weight=sp.reward_spin),
        "drop": RewardTermCfg(func=object_dropped, weight=sp.reward_drop),
        "alive": RewardTermCfg(func=reward_alive, weight=sp.reward_alive),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=sp.reward_action_rate),
    }
    if plate:
        rewards["tilt"] = RewardTermCfg(func=reward_tilt, weight=sp.reward_tilt)
        rewards["nontip"] = RewardTermCfg(func=reward_nontip, weight=sp.reward_nontip)
        rewards["support"] = RewardTermCfg(func=reward_support, weight=sp.reward_support)
    else:
        rewards["center"] = RewardTermCfg(func=reward_center, weight=sp.reward_center)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.0,
            sensors=plate_sensors(sp, prof) if plate else (),
        ),
        observations=observations,
        actions={
            "hand": make_action_cfg(
                "hand", ws, finger_mode="delta", finger_ema=1.0, base_ema=1.0, profile=prof
            )
        },
        commands={"spin": cmd},
        events={
            "reset_scene_to_default": EventTermCfg(
                func=envs_mdp.reset_scene_to_default, mode="reset"
            ),
            "reset_spin": EventTermCfg(func=reset_spin, mode="reset"),
        },
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "dropped": TerminationTermCfg(func=object_dropped),
            "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.55),
            distance=0.7,
            elevation=-30.0,
            azimuth=150.0,
        ),
        sim=make_sim_cfg(nconmax=200 if plate else 150, njmax=1000 if plate else 800),
        decimation=4,
        episode_length_s=sp.episode_s,
        scale_rewards_by_dt=False,
        is_finite_horizon=False,
    )
    spec = ALLEGRO_HAND_LEFT if prof is None else prof.spec
    hand = make_hand_entity_cfg(spec, ws, base_gains=STIFF_BASE_GAINS, profile=prof)
    init = dict(hand.init_state.joint_pos or {})
    tendons = set(prof.tendon_targets) if prof else set()
    init.update({j: v for j, v in start_finger_pose(sp, prof).items() if j not in tendons})
    if prof is not None:
        init.update(dict(prof.fixed_joints))
    hand.init_state.joint_pos = init
    ents = {"hand": hand}
    if plate:
        ents["plate"] = EntityCfg(spec_fn=lambda: make_plate_spec(sp))
    else:
        for name, rgba in (
            ("ball_a", (0.22, 0.45, 0.85, 1.0)),
            ("ball_b", (0.93, 0.45, 0.38, 1.0)),
        ):
            ents[name] = EntityCfg(
                spec_fn=lambda rgba=rgba: make_ball_spec(
                    sp.ball_radius, sp.ball_mass, (0.01, 1.0), (1.0, 0.005, 0.0005), rgba=rgba
                )
            )
    cfg.scene.entities = ents
    return cfg


def _register(task_id: str, sp: PalmTaskSpec) -> None:
    register_mjlab_task(
        task_id=task_id,
        env_cfg=make_palm_task_env_cfg(sp),
        play_env_cfg=make_palm_task_env_cfg(sp, play=True),
        rl_cfg=make_ppo_cfg(
            f"{sp.hand}_{sp.task}" + ("_tilt" if sp.hand_tilt else ""),
            max_iterations=15000 if sp.task == "plate" else 10000,
            init_std=0.5,
        ),
        runner_cls=None,
    )


for _h in HANDS:
    _register(f"Baoding-{HAND_SUFFIX[_h]}", PalmTaskSpec(task="baoding", hand=_h))
    _register(f"PlateSpin-{HAND_SUFFIX[_h]}", PalmTaskSpec(task="plate", hand=_h))
# Fingers 0.3 rad down. The pair starts along the fingers: side by side, the outer ball rolls out of
# the open side of the Sharpa cup.
_register(
    "Baoding-Sharpa-Tilt",
    PalmTaskSpec(task="baoding", hand="sharpa", hand_tilt=0.3, ball_axis_deg=0.0),
)
# Also rolled 0.25 rad toward the thumb (the hand's +y side): the balls rest toward thumb and index.
_register(
    "Baoding-Sharpa-TiltThumb",
    PalmTaskSpec(task="baoding", hand="sharpa", hand_tilt=0.3, hand_roll=-0.25, ball_axis_deg=0.0),
)
