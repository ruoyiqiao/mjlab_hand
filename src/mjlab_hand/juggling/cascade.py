"""Three-ball cascade with two floating Allegro hands (Lee et al., arXiv 2608.26800, Fig. 1, Sec. 2A/4, Tables S1-S3).

* Schedule: never delayed. Beat 0.35 s; ball k mod 3 is thrown on beat k (right hand on even beats), flight 0.70 s,
  caught by the other hand, held one beat (dwell), thrown again.
* Throw skill (Table S2): command = landing point; fixed release point (0.075 m inward of the hand home |y| = 0.131 m),
  v_rel = (u - p_rel) / F + (0, 0, g F / 2), hand acceleration at release = (0, 0, -g); quintic plan (continuous p, v, a).
* Catch skill: closed loop, target = ballistic prediction of the incoming ball at the catch time, replanned each step
  until 0.1 s before the catch. Deviation: the hand arrives with kappa = 0.5 of the ball velocity (an open palm cannot
  catch at rest: the following dip would need < -g).
* Memory-based learner (Sec. 4B, Eq. 17-21): optional, adapts the landing command per throw (``MemoryLearner``).
  It has no effect on this prior (landing barely follows the command: the ball rolls in the grip).
* Root: position actuators with ctrl = q_ref + (kv q'_ref + f_ff) / kp, evaluated every physics substep.
* RL: the action is a residual (6 root + 16 finger per hand) on the designed reference; zero action = prior.
Start (deviation): a steady-state snapshot at a fixed phase instead of the paper's two balls in one hand.

Other hands (``CascadeSpec.hand``): the same schedule, skills and rewards; per hand only the measured quantities change
(tools/allegro4/juggle3_probe.py -> tasks_common/hand_params/juggle3.json): palm-up anchor from the hand profile,
cup / grip / open finger poses from the fingertip-role rule of the Juggle1/2 cup probe, the ball rest point in the grip
(per side), and the hand spacing ``hand_y`` (the smallest value >= Allegro's 0.131 m that keeps both hands apart at the
release points). The residual has 6 + the hand's finger targets (Shadow: joints + J0 tendons) per hand.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_hand.juggling.juggle_env_cfg import CUP_POSE, make_ball_spec
from mjlab_hand.robot.allegro import ALLEGRO_HAND_LEFT, ALLEGRO_HAND_RIGHT
from mjlab_hand.tasks_common import mdp as tmdp
from mjlab_hand.tasks_common.actions import ALLEGRO_FINGER_JOINTS, actuator_of
from mjlab_hand.tasks_common.env_builder import hand_joint_cfg, make_hand_entity_cfg
from mjlab_hand.tasks_common.hand_profile import get_profile
from mjlab_hand.tasks_common.workspace import BASE_JOINTS, HandRootWorkspace

G = 9.81
HANDS = ("hand_r", "hand_l")
SIDES = ("right", "left")
_PARAMS3 = Path(__file__).resolve().parents[1] / "tasks_common/hand_params/juggle3.json"


def hand_cascade_params(hand: str) -> dict:
    """Measured Juggle3 parameters of a non-Allegro hand (tools/allegro4/juggle3_probe.py)."""
    return json.loads(_PARAMS3.read_text())[hand]


def finger_pose(kind: str) -> dict:
    """Finger joint targets (rad) by name: cup / shallow / open / grip / fip (index+pinky open, thumb closed)."""
    p = dict(CUP_POSE)
    if kind == "shallow":
        for f in ("ff", "mf", "rf"):
            p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.2, 0.35, 0.3
    elif kind == "open":
        for f in ("ff", "mf", "rf"):
            p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.0, 0.1, 0.1
    elif kind == "grip":
        for f in ("ff", "mf", "rf"):
            p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.6, 1.0, 0.8
        p["thj0"], p["thj1"], p["thj2"], p["thj3"] = 1.3, 0.5, 0.8, 0.8
    elif kind == "fip":
        for f in ("ff", "rf"):
            p[f + "j1"], p[f + "j2"], p[f + "j3"] = 0.0, 0.1, 0.1
    elif kind != "cup":
        raise ValueError(kind)
    return p


BALLS = ("ball_0", "ball_1", "ball_2")
BALL_RGBA = ((0.86, 0.47, 0.42, 1.0), (0.88, 0.72, 0.36, 1.0), (0.33, 0.62, 0.64, 1.0))


@dataclass
class CascadeSpec:
    # Paper geometry / timing (Table S1, S3) and ball (Sec. 2A).
    hand_y: float = 0.131  # hand home |y|: midpoint of release (0.056) and catch (0.206)
    lat: float = 0.075  # release 0.075 m inward, catch 0.075 m outward of home
    beat_s: float = 0.35
    flight_s: float = 0.70
    ball_radius: float = 0.03175  # 63.5 mm
    ball_mass: float = 0.130
    ball_solref: tuple[float, float] = (
        0.02,
        1.5,
    )  # soft, little rebound (drop test in juggling_plan.md)
    anchor_z: float = 0.60
    # Designed prior.
    kappa: float = 0.5  # cushioned catch (deviation, see module docstring)
    # Finger schedule of the prior: hold pose during the dwell, release pose from release_lead_s before the release
    # for release_hold_s, then back to the hold pose (the next ball is caught into the formed grip).
    hold_pose: str = "grip"  # keeps the ball centred (1.4 vs 3-4 cm in a cup); see juggling_plan.md
    release_pose: str = (
        "open"  # opening 0.03 s before release cuts the landing error 0.20 -> 0.08 m
    )
    release_lead_s: float = 0.03
    release_hold_s: float = 0.15
    grip_ramp_s: float = 0.0
    replan_cutoff_s: float = 0.10  # paper: replanning stops 0.1 s before the catch
    catch_box: tuple[float, float, float] = (
        0.12,
        0.18,
        0.15,
    )  # clamp of catch targets (x, lateral, z) around home
    # Learner (Sec. 4B); h_x, h_y and k are not given in the paper.
    learner: bool = False
    gamma: float = 1e-3
    eta: float = 0.3
    k_nn: int = 16
    h_x: float = 0.10  # kernel widths are not given in the paper; chosen at the landing-error scale
    h_y: float = 0.30
    memory: int = 256
    # Episode / rules.
    episode_s: float = 11.0  # 5 cycles = 30 caught throws = 10.5 s, plus margin
    catch_check_s: float = (
        0.08  # the ball must be held by the scheduled hand this long after the catch time
    )
    drop_below: float = 0.40
    start_steps: int = 16  # start 0.16 s after a release: the released ball is clear of the hand
    random_phase: bool = False  # training: random beat parity and phase in [start_steps, beat)
    # Control.
    timestep: float = 0.002
    decimation: int = 5  # 100 Hz control; the reference is evaluated every 2 ms substep
    slide_gains: tuple[float, float, float] = (5000.0, 140.0, 400.0)
    rot_gains: tuple[float, float, float] = (300.0, 12.0, 80.0)
    # RL residual: action in [-1, 1] -> residual.
    res_slide: float = 0.03
    res_rot: float = 0.15
    res_finger: float = 0.3
    reward_residual: float = -0.01
    finger_action_rate: float = 0.0
    """Weight (<= 0) of the finger residual action-rate penalty (finger smoothing, tasks_common.mdp); 0 disables."""
    hand: str = "allegro"
    """Hand model; non-Allegro hands read their measured parameters from hand_params/juggle3.json."""


# ---------------------------------------------------------------------- quintic helpers (batched)
def quintic_coeffs(p0, v0, a0, p1, v1, a1, T):
    """Coefficients c0..c5 (stacked on a new last dim) of the quintic matching (p, v, a) at 0 and T.
    p*, v*, a*: (..., 3); T: (...,) -> (..., 3, 6)."""
    T = T.clamp(min=1e-3).unsqueeze(-1)
    T2, T3, T4, T5 = T * T, T**3, T**4, T**5
    b0 = p1 - (p0 + v0 * T + 0.5 * a0 * T2)
    b1 = v1 - (v0 + a0 * T)
    b2 = a1 - a0
    c3 = (10 * b0 - 4 * b1 * T + 0.5 * b2 * T2) / T3
    c4 = (-15 * b0 + 7 * b1 * T - b2 * T2) / T4
    c5 = (6 * b0 - 3 * b1 * T + 0.5 * b2 * T2) / T5
    return torch.stack([p0, v0, 0.5 * a0, c3, c4, c5], -1)


def quintic_eval(c, s):
    """c: (..., 3, 6), s: (...,) local time -> pos, vel, acc (..., 3)."""
    s = s.unsqueeze(-1)
    p = c[..., 0] + s * (
        c[..., 1] + s * (c[..., 2] + s * (c[..., 3] + s * (c[..., 4] + s * c[..., 5])))
    )
    v = c[..., 1] + s * (
        2 * c[..., 2] + s * (3 * c[..., 3] + s * (4 * c[..., 4] + s * 5 * c[..., 5]))
    )
    a = 2 * c[..., 2] + s * (6 * c[..., 3] + s * (12 * c[..., 4] + s * 20 * c[..., 5]))
    return p, v, a


def ballistic(p, v, dt):
    """Ball state after dt (broadcast): pos, vel."""
    dt = dt.unsqueeze(-1)
    g = torch.zeros_like(p)
    g[..., 2] = -G
    return p + v * dt + 0.5 * g * dt * dt, v + g * dt


def _cup_point(model, data, hand: str, r: float) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{hand}/palm_collision")
    assert gid >= 0, hand
    R = data.geom_xmat[gid].reshape(3, 3)
    n = R[:, 0]
    if n[2] < 0.5:
        raise ValueError(f"{hand}: palm must face up, normal={n}")
    return data.geom_xpos[gid] + n * (model.geom_size[gid][0] + r + 0.001)


# ---------------------------------------------------------------------- memory-based learner (Sec. 4B)
class MemoryLearner:
    """Batched regularized memory-based learner: one memory per (env, skill); identity prior f0(x, u) = u.

    Eq. 17: u_bar = sum w u / sum w over the kNN in (x, y) space, w = exp(-|x_i-x|^2/h_x^2 - |y_i-y_d|^2/h_y^2).
    Eq. 18: (C, D, d) = argmin sum w |y_i - C dx_i - D du_i - d|^2 + gamma (|C - 0|^2 + |D - I|^2 + |d - u_bar|^2).
    Eq. 21: u* = u_bar + (D^T D + eta I)^-1 D^T (y_d - d).
    With no stored experience, u* = y_d (prior)."""

    def __init__(
        self,
        n: int,
        skills: int,
        cap: int,
        k: int,
        hx: float,
        hy: float,
        gamma: float,
        eta: float,
        device,
    ):
        self.n, self.S, self.cap, self.k = n, skills, cap, k
        self.hx2, self.hy2, self.gamma, self.eta = hx * hx, hy * hy, gamma, eta
        self.X = torch.zeros(n, skills, cap, 6, device=device)
        self.U = torch.zeros(n, skills, cap, 3, device=device)
        self.Y = torch.zeros(n, skills, cap, 3, device=device)
        self.count = torch.zeros(n, skills, dtype=torch.long, device=device)
        self.device = device

    def clear(self, env_ids=None):
        if env_ids is None:
            self.count.zero_()
        else:
            self.count[env_ids] = 0

    def add(self, env_ids, skill, x, u, y):
        """env_ids (m,), skill (m,), x (m,6), u (m,3), y (m,3); ring buffer per (env, skill)."""
        if env_ids.numel() == 0:
            return
        slot = self.count[env_ids, skill] % self.cap
        self.X[env_ids, skill, slot] = x
        self.U[env_ids, skill, slot] = u
        self.Y[env_ids, skill, slot] = y
        self.count[env_ids, skill] += 1

    def query(self, env_ids, skill, x, yd):
        """-> u* (m,3) for the given envs/skills."""
        m = env_ids.numel()
        if m == 0:
            return yd
        X, U, Y = (
            self.X[env_ids, skill],
            self.U[env_ids, skill],
            self.Y[env_ids, skill],
        )  # (m,cap,*)
        cnt = self.count[env_ids, skill].clamp(max=self.cap)
        valid = torch.arange(self.cap, device=self.device).view(1, -1) < cnt.view(-1, 1)
        d2 = ((X - x.unsqueeze(1)) ** 2).sum(-1) / self.hx2 + ((Y - yd.unsqueeze(1)) ** 2).sum(
            -1
        ) / self.hy2
        d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
        k = min(self.k, self.cap)
        d2k, idx = torch.topk(d2, k, dim=1, largest=False)
        w = torch.exp(-d2k)  # (m,k), 0 for invalid (inf)
        Xk = torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, 6))
        Uk = torch.gather(U, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        Yk = torch.gather(Y, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        sw = w.sum(1, keepdim=True)
        has = cnt > 0
        ubar = torch.where(
            has.unsqueeze(-1), (w.unsqueeze(-1) * Uk).sum(1) / sw.clamp(min=1e-8), yd
        )
        # Local linear fit, Theta = [C (3x6) | D (3x3) | d (3x1)], Z_i = [dx_i; du_i; 1].
        Z = torch.cat(
            [Xk - x.unsqueeze(1), Uk - ubar.unsqueeze(1), torch.ones_like(w).unsqueeze(-1)], -1
        )  # (m,k,10)
        Wz = w.unsqueeze(-1) * Z
        A = Z.transpose(1, 2) @ Wz + self.gamma * torch.eye(10, device=self.device)  # (m,10,10)
        th0 = torch.zeros(m, 3, 10, device=self.device)
        th0[:, :, 6:9] = torch.eye(3, device=self.device)
        th0[:, :, 9] = ubar  # d0 = f0(x, ubar) = ubar
        B = Yk.transpose(1, 2) @ Wz + self.gamma * th0  # (m,3,10)
        theta = torch.linalg.solve(A.transpose(1, 2), B.transpose(1, 2)).transpose(1, 2)  # B A^-1
        D = theta[:, :, 6:9]
        d = theta[:, :, 9]
        H = D.transpose(1, 2) @ D + self.eta * torch.eye(3, device=self.device)
        step = torch.linalg.solve(H, (D.transpose(1, 2) @ (yd - d).unsqueeze(-1))).squeeze(-1)
        return torch.where(has.unsqueeze(-1), ubar + step, yd)


# ---------------------------------------------------------------------- command term: orchestrator + skills
@dataclass(kw_only=True)
class CascadeCommandCfg(CommandTermCfg):
    spec: CascadeSpec = field(default_factory=CascadeSpec)
    hand_names: tuple[str, ...] = HANDS
    ball_names: tuple[str, ...] = BALLS
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    palm_body: str | tuple[str, ...] = "palm"
    """Palm body name, or one per hand (sides name it differently, e.g. lh_palm / rh_palm)."""
    cup_rel_palm: tuple[tuple[float, float, float], ...] | None = None
    """Per hand: cup point minus palm origin at home (measured); None = Allegro's palm_collision face."""

    def build(self, env):
        return CascadeCommand(self, env)


class CascadeCommand(CommandTerm):
    cfg: CascadeCommandCfg

    def __init__(self, cfg: CascadeCommandCfg, env):
        super().__init__(cfg, env)
        sp = cfg.spec
        n, dev = self.num_envs, self.device
        self.sp = sp
        self.hands = [env.scene[h] for h in cfg.hand_names]
        self.balls = [env.scene[b] for b in cfg.ball_names]
        pbs = (
            (cfg.palm_body,) * len(cfg.hand_names)
            if isinstance(cfg.palm_body, str)
            else tuple(cfg.palm_body)
        )
        self.palm_body_names = pbs
        self.palm_ids = [h.find_bodies(pb)[0][0] for h, pb in zip(self.hands, pbs)]
        m = env.sim.mj_model
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        f = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=dev)  # noqa: E731
        cups, palms = [], []
        for i, h in enumerate(cfg.hand_names):
            palm = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{h}/{pbs[i]}")].copy()
            palms.append(palm)
            cups.append(
                palm + np.asarray(cfg.cup_rel_palm[i])
                if cfg.cup_rel_palm is not None
                else _cup_point(m, d, h, sp.ball_radius)
            )
        self.cup_home = f(cups)  # (2,3) env-local (env origin at 0 for the CPU model)
        self.cup_offset = f(cups) - f(palms)
        self.outward = f([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])  # right hand y < 0
        self.p_rel = self.cup_home - self.outward * sp.lat  # release points (2,3)
        self.p_catch_nom = self.cup_home + self.outward * sp.lat  # nominal catch points (2,3)
        # Slide Jacobian per hand (world cup displacement per unit base_x/y/z), finite differences.
        self.slide_ids, Jl, Jinv = [], [], []
        for h, pb in zip(cfg.hand_names, pbs):
            pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{h}/{pb}")
            cols = []
            for jn in ("base_x", "base_y", "base_z"):
                adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{h}/{jn}")]
                dd = mujoco.MjData(m)
                mujoco.mj_kinematics(m, dd)
                p0 = dd.xpos[pid].copy()
                dd.qpos[adr] += 0.01
                mujoco.mj_kinematics(m, dd)
                cols.append((dd.xpos[pid] - p0) / 0.01)
            J = np.stack(cols, 1)
            Jl.append(J)
            Jinv.append(np.linalg.inv(J))
            ent = env.scene[h]
            self.slide_ids.append(
                [ent.joint_names.index(j) for j in ("base_x", "base_y", "base_z")]
            )
        self.slide_j = f(Jl)
        self.slide_jinv = f(Jinv)
        # Effective translational mass of each hand root (mj_crb diagonal of the slide dofs).
        mujoco.mj_forward(m, d)
        mujoco.mj_crb(m, d)
        masses = []
        for h in cfg.hand_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{h}/base_x")
            dof = m.jnt_dofadr[jid]
            e = np.zeros(m.nv)
            e[dof] = 1.0
            r = np.zeros(m.nv)
            mujoco.mj_mulM(m, d, r, e)
            masses.append(float(r[dof]))
        self.mass = f(masses)
        self.slide_damping = float(
            m.dof_damping[
                m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_r/base_x")]
            ]
        )
        # Timing on the control grid.
        self.dt = float(env.step_dt)
        self.beat_steps = int(round(sp.beat_s / self.dt))
        assert abs(self.beat_steps * self.dt - sp.beat_s) < 1e-6, (
            "beat must be a multiple of the control step"
        )
        self.flight_steps = int(round(sp.flight_s / self.dt))
        self.cut_steps = int(round(sp.replan_cutoff_s / self.dt))
        self.check_steps = int(round(sp.catch_check_s / self.dt))
        # Per-env state.
        z = torch.zeros
        self.step_idx = z(n, dtype=torch.long, device=dev)  # schedule step (time = step_idx * dt)
        self.ep_steps = z(n, dtype=torch.long, device=dev)
        self.coef = z(n, 2, 3, 6, device=dev)  # cup reference per hand (env-local)
        self.seg_start = z(
            n, 2, dtype=torch.long, device=dev
        )  # schedule step where the segment starts
        self.u_cmd = z(n, 2, 3, device=dev)  # current throw command per hand
        self.x_state = z(n, 2, 6, device=dev)  # learner state of the current throw
        self.catch_cmd = self.p_catch_nom.expand(n, -1, -1).clone()  # last catch target per hand
        self.land_est = z(n, 3, 3, device=dev)  # latest landing estimate per ball
        self.pending_y = z(
            n, 3, dtype=torch.bool, device=dev
        )  # throw of ball b awaits its landing measurement
        self.pend_hand = z(n, 3, dtype=torch.long, device=dev)
        self.pend_rel_step = z(n, 3, dtype=torch.long, device=dev)
        self.contact = z(n, 2, 3, dtype=torch.bool, device=dev)
        self.last_ct = torch.full(
            (n, 2, 3), -1000, dtype=torch.long, device=dev
        )  # last step with contact
        self.caught = z(n, device=dev)  # consecutive caught throws (this episode)
        self.throws = z(n, device=dev)
        self.failed = z(n, dtype=torch.bool, device=dev)
        self.fail_code = z(
            n, dtype=torch.long, device=dev
        )  # 1 missed, 2 dropped, 3 wrong hand, 4 not released
        self.catch_now = z(n, device=dev)
        self.land_err_now = z(n, device=dev)
        self.throw_now = z(n, dtype=torch.bool, device=dev)
        self.just_reset = torch.ones(n, dtype=torch.bool, device=dev)
        self.log_land = []  # (env, hand, err xyz) per measured throw
        self.record = False
        self.learner = MemoryLearner(
            n, 2, sp.memory, sp.k_nn, sp.h_x, sp.h_y, sp.gamma, sp.eta, dev
        )
        self.use_learner = bool(sp.learner)
        for k in ("caught", "throws", "land_err"):
            self.metrics[k] = z(n, device=dev)

    # ------------------------------------------------------------------ schedule
    @property
    def command(self) -> torch.Tensor:
        return self.phase_features()

    def beat_of(self, step):
        return torch.div(step, self.beat_steps, rounding_mode="floor")

    def hand_in_dwell(self, beat):
        """(N,) beat -> (N,2) bool: right holds on odd beats (caught at odd, throws at the next even beat)."""
        odd = (beat % 2) == 1
        return torch.stack([odd, ~odd], -1)

    def held_ball(self, beat):
        """Ball held during a dwell that starts at beat (thrown at beat + 1)."""
        return (beat + 1) % 3

    def incoming_ball(self, beat):
        """Ball caught at the end of a vacant segment that starts at beat (thrown at beat - 1)."""
        return (beat - 1) % 3

    # ------------------------------------------------------------------ state helpers
    def ball_pos(self):
        return torch.stack(
            [b.data.root_link_pos_w for b in self.balls], 1
        ) - self._env.scene.env_origins.unsqueeze(1)

    def ball_vel(self):
        return torch.stack([b.data.root_link_lin_vel_w for b in self.balls], 1)

    def cup_pos(self):
        p = torch.stack(
            [h.data.body_link_pos_w[:, i] for h, i in zip(self.hands, self.palm_ids)], 1
        )
        return p - self._env.scene.env_origins.unsqueeze(1) + self.cup_offset.unsqueeze(0)

    def cup_vel(self):
        return torch.stack(
            [h.data.body_link_lin_vel_w[:, i] for h, i in zip(self.hands, self.palm_ids)], 1
        )

    def ref_at(self, t_steps: torch.Tensor, frac: float = 0.0):
        """Reference cup (pos, vel, acc) per hand at schedule time (t_steps + frac) * dt: (N,2,3)."""
        s = (t_steps.view(-1, 1) - self.seg_start).float() * self.dt + frac * self.dt
        return quintic_eval(self.coef, s)

    def throw_velocity(self, u, h):
        """Release velocity for landing command u (N,3) from hand h's release point."""
        F = self.sp.flight_s
        v = (u - self.p_rel[h]) / F
        v[:, 2] = (u[:, 2] - self.p_rel[h, 2]) / F + 0.5 * G * F
        return v

    def plan_throw(self, ids, h, start_step):
        """Plan hand h's throw (dwell) for envs ids starting at start_step from its current reference state."""
        if ids.numel() == 0:
            return
        sp = self.sp
        p0, v0, a0 = quintic_eval(
            self.coef[ids, h], (start_step[ids] - self.seg_start[ids, h]).float() * self.dt
        )
        yd = self.p_catch_nom[1 - h].expand(ids.numel(), -1)
        x = torch.cat(
            [
                self.catch_cmd[ids, h],
                self.land_est[ids, self.held_ball(self.beat_of(start_step[ids]))],
            ],
            -1,
        )
        u = (
            self.learner.query(ids, torch.full_like(ids, h), x, yd)
            if self.use_learner
            else yd.clone()
        )
        self.u_cmd[ids, h] = u
        self.x_state[ids, h] = x
        v1 = self.throw_velocity(u, h)
        a1 = torch.zeros_like(v1)
        a1[:, 2] = -G
        p1 = self.p_rel[h].expand(ids.numel(), -1)
        T = torch.full((ids.numel(),), sp.beat_s, device=self.device)
        self.coef[ids, h] = quintic_coeffs(p0, v0, a0, p1, v1, a1, T)
        self.seg_start[ids, h] = start_step[ids]

    def catch_target(self, ids, h, t_catch_step):
        """Predicted state of the incoming ball at the catch time (true ball state; nominal if not free)."""
        sp = self.sp
        b = self.incoming_ball(self.beat_of(t_catch_step[ids]) - 1)
        bp = self.ball_pos()[ids, b] if ids.numel() else None
        bv = self.ball_vel()[ids, b]
        free = ~self.contact[ids, :, b].any(-1)
        dt = (t_catch_step[ids] - self.step_idx[ids]).float() * self.dt
        pc, vc = ballistic(bp, bv, dt)
        # Nominal fallback: the ball is still in contact with a hand (not yet thrown).
        nom_p = self.p_catch_nom[h].expand(ids.numel(), -1)
        nom_v = self.incoming_velocity(h).expand(ids.numel(), -1)
        pc = torch.where(free.unsqueeze(-1), pc, nom_p)
        vc = torch.where(free.unsqueeze(-1), vc, nom_v)
        lo = self.cup_home[h] - torch.tensor(sp.catch_box, device=self.device)
        hi = self.cup_home[h] + torch.tensor(sp.catch_box, device=self.device)
        pc_cl = torch.max(torch.min(pc, hi), lo)
        self.land_est[ids, b] = pc
        return pc_cl, vc, b

    def plan_catch(self, ids, h, now_step, t_catch_step):
        if ids.numel() == 0:
            return
        p0, v0, a0 = quintic_eval(
            self.coef[ids, h], (now_step[ids] - self.seg_start[ids, h]).float() * self.dt
        )
        pc, vc, _ = self.catch_target(ids, h, t_catch_step)
        v1 = self.sp.kappa * vc
        a1 = torch.zeros_like(v1)
        T = (t_catch_step[ids] - now_step[ids]).float() * self.dt
        self.coef[ids, h] = quintic_coeffs(p0, v0, a0, pc, v1, a1, T)
        self.seg_start[ids, h] = now_step[ids]
        self.catch_cmd[ids, h] = pc

    # ------------------------------------------------------------------ nominal initial plans
    def init_plans(self, ids, start_step):
        """Set both hands' current segments to the nominal (prior) plans around start_step."""
        if ids.numel() == 0:
            return
        sp = self.sp
        m = ids.numel()
        beat = self.beat_of(start_step[ids])
        b0 = beat * self.beat_steps  # segment start step
        dwell = self.hand_in_dwell(beat)
        T = torch.full((m,), sp.beat_s, device=self.device)
        for h in (0, 1):
            pc = self.p_catch_nom[h].expand(m, -1)
            vc = self.incoming_velocity(h).expand(m, -1)
            u = self.p_catch_nom[1 - h].expand(m, -1).clone()
            v_rel = self.throw_velocity(u.clone(), h)
            a_rel = torch.zeros(m, 3, device=self.device)
            a_rel[:, 2] = -G
            p_rel = self.p_rel[h].expand(m, -1)
            c_dwell = quintic_coeffs(
                pc, sp.kappa * vc, torch.zeros_like(vc), p_rel, v_rel, a_rel, T
            )
            c_vac = quintic_coeffs(p_rel, v_rel, a_rel, pc, sp.kappa * vc, torch.zeros_like(vc), T)
            dw = dwell[:, h].view(-1, 1, 1)
            self.coef[ids, h] = torch.where(dw, c_dwell, c_vac)
            self.seg_start[ids, h] = b0
            self.u_cmd[ids, h] = u
            self.catch_cmd[ids, h] = pc
            self.x_state[ids, h] = torch.cat([pc, pc], -1)

    def incoming_velocity(self, h):
        """Nominal velocity (3,) of a ball arriving at hand h's catch point from the other hand."""
        u = self.p_catch_nom[h].view(1, 3).clone()
        v = self.throw_velocity(u, 1 - h)[0]
        v[2] -= G * self.sp.flight_s
        return v

    def nominal_ball_state(self, ids, start_step):
        """Exact steady-state ball positions / velocities at start_step (prior plans): (m,3,3) x2."""
        sp = self.sp
        m = ids.numel()
        t = start_step[ids]
        pos = torch.zeros(m, 3, 3, device=self.device)
        vel = torch.zeros(m, 3, 3, device=self.device)
        beat = self.beat_of(t)
        rp, rv, _ = quintic_eval(
            self.coef[ids], (t.view(-1, 1) - self.seg_start[ids]).float() * self.dt
        )  # (m,2,3)
        for b in range(3):
            # last throw of ball b at or before t: beats k <= beat with k % 3 == b
            k = beat - ((beat - b) % 3)
            h_thr = k % 2  # right on even beats
            t_rel = k * self.beat_steps
            dt = (t - t_rel).float() * self.dt
            in_flight = dt < sp.flight_s - 1e-6
            # flight from the release point of h_thr to the catch point of the other hand
            p_r = self.p_rel[h_thr]
            u = self.p_catch_nom[1 - h_thr]
            F = sp.flight_s
            v_r = (u - p_r) / F
            v_r[:, 2] = (u[:, 2] - p_r[:, 2]) / F + 0.5 * G * F
            pf, vf = ballistic(p_r, v_r, dt)
            hold_h = 1 - h_thr
            ph = torch.gather(rp, 1, hold_h.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            vh = torch.gather(rv, 1, hold_h.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            pos[:, b] = torch.where(in_flight.unsqueeze(-1), pf, ph)
            vel[:, b] = torch.where(in_flight.unsqueeze(-1), vf, vh)
        return pos, vel

    # ------------------------------------------------------------------ per-step update (after physics)
    def _update_metrics(self):
        self.metrics["caught"] = self.caught.clone()
        self.metrics["throws"] = self.throws.clone()

    def _resample_command(self, env_ids):
        pass

    def _update_command(self, env_ids=None):
        if (
            env_ids is not None
        ):  # explicit reset path (env.reset()): the next update follows physics
            self.just_reset[env_ids] = False
            return
        self.update()

    def update(self):
        sp = self.sp
        n = self.num_envs
        dev = self.device
        env = self._env
        live = ~self.just_reset
        self.just_reset[:] = False
        self.catch_now.zero_()
        self.land_err_now.zero_()
        self.throw_now.zero_()
        for hi, h in enumerate(self.cfg.hand_names):
            for bi, b in enumerate(self.cfg.ball_names):
                s = env.scene[f"{h}_{b}_contact"]
                self.contact[:, hi, bi] = (s.data.found.reshape(n, -1) > 0).any(-1)
        # Advance the clock for live envs.
        self.step_idx = torch.where(live, self.step_idx + 1, self.step_idx)
        self.last_ct = torch.where(self.contact, self.step_idx.view(-1, 1, 1), self.last_ct)
        self.ep_steps = torch.where(live, self.ep_steps + 1, self.ep_steps)
        now = self.step_idx
        beat = self.beat_of(now)
        r = now - beat * self.beat_steps
        dwell = self.hand_in_dwell(beat)
        p = self.ball_pos()
        v = self.ball_vel()
        # 1) landing measurement of pending throws: first free step >= 2 steps after the release.
        free_b = ~self.contact.any(1)  # (N,3)
        since = now.view(-1, 1) - self.pend_rel_step
        meas = self.pending_y & live.view(-1, 1) & free_b & (since >= 2)
        late = self.pending_y & live.view(-1, 1) & ~free_b & (since >= 8)
        if meas.any():
            e, b = meas.nonzero(as_tuple=True)
            t_land = self.pend_rel_step[e, b] + self.flight_steps
            y, _ = ballistic(p[e, b], v[e, b], (t_land - now[e]).float() * self.dt)
            h = self.pend_hand[e, b]
            u = self.u_cmd[e, h]
            yd = self.p_catch_nom[1 - h]
            err = (y - yd).norm(dim=-1)
            self.land_err_now.index_put_((e,), err, accumulate=True)
            self.metrics["land_err"][e] = err
            if self.use_learner:
                self.learner.add(e, h, self.x_state[e, h], u, y)
            if self.record:
                self.log_land.append(
                    torch.cat(
                        [e.float().unsqueeze(-1), h.float().unsqueeze(-1), y - yd, u - yd], -1
                    ).cpu()
                )
            self.pending_y[e, b] = False
        if late.any():
            e, b = late.nonzero(as_tuple=True)
            self.failed[e] = True
            self.fail_code[e] = torch.where(
                self.fail_code[e] == 0, torch.full_like(self.fail_code[e], 4), self.fail_code[e]
            )
            self.pending_y[e, b] = False
        # 2) catch checks: check_steps after each catch time (catch at the start of a dwell).
        chk = live & (r == self.check_steps)
        for h in (0, 1):
            ids = (chk & dwell[:, h]).nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                continue
            b = self.held_ball(beat[ids])
            # caught = touched by the scheduled hand within the last 3 steps and resting near its cup
            # (a soft ball bounces in the grip, so instantaneous contact flickers)
            near = (p[ids, b] - self.cup_pos()[ids, h]).norm(dim=-1) < 0.07
            ok = (now[ids] - self.last_ct[ids, h, b] <= 3) & near
            wrong = (now[ids] - self.last_ct[ids, 1 - h, b] <= 3) & ~ok
            self.caught[ids] += ok.float()
            self.catch_now[ids] += ok.float()
            bad = ~ok
            self.failed[ids] |= bad
            code = torch.where(wrong, torch.full_like(b, 3), torch.full_like(b, 1))
            self.fail_code[ids] = torch.where(
                bad & (self.fail_code[ids] == 0), code, self.fail_code[ids]
            )
        # 3) drops.
        dropped = (p[..., 2] < self.cup_home[0, 2] - sp.drop_below).any(-1) & live
        self.fail_code = torch.where(
            dropped & (self.fail_code == 0), torch.full_like(self.fail_code, 2), self.fail_code
        )
        self.failed |= dropped
        # 4) planning for the next control step (segments start at beat boundaries).
        boundary = live & (r == 0)
        for h in (0, 1):
            # throw (dwell) starts: plan once from the reference state
            ids = (boundary & dwell[:, h]).nonzero(as_tuple=True)[0]
            self.plan_throw(ids, h, now)
            # release (vacant starts): register the landing measurement, plan the catch
            ids = (boundary & ~dwell[:, h]).nonzero(as_tuple=True)[0]
            if ids.numel():
                b_thr = (beat[ids]) % 3  # ball thrown at this beat
                self.pending_y[ids, b_thr] = True
                self.pend_hand[ids, b_thr] = h
                self.pend_rel_step[ids, b_thr] = now[ids]
                self.throws[ids] += 1.0
                self.throw_now[ids] = True
                t_c = (beat[ids] + 1) * self.beat_steps
                tc_full = torch.zeros(n, dtype=torch.long, device=dev)
                tc_full[ids] = t_c
                self.plan_catch(ids, h, now, tc_full)
            # continuous catch replanning until the cutoff
            ids = (
                live & ~boundary & ~dwell[:, h] & (r <= self.beat_steps - self.cut_steps)
            ).nonzero(as_tuple=True)[0]
            if ids.numel():
                tc_full = torch.zeros(n, dtype=torch.long, device=dev)
                tc_full[ids] = (beat[ids] + 1) * self.beat_steps
                self.plan_catch(ids, h, now, tc_full)

    # ------------------------------------------------------------------ features for policies
    def phase_features(self):
        """Per hand: dwell flag, time to segment end / T, plus held / incoming ball one-hot (2 x 2 x ... )."""
        beat = self.beat_of(self.step_idx)
        r = (self.step_idx - beat * self.beat_steps).float() / self.beat_steps
        dwell = self.hand_in_dwell(beat).float()
        held = torch.nn.functional.one_hot(self.held_ball(beat), 3).float()
        inc = torch.nn.functional.one_hot(self.incoming_ball(beat), 3).float()
        return torch.cat([dwell, (1 - r).unsqueeze(-1), held, inc], -1)

    def reset(self, env_ids=None):
        return super().reset(env_ids)


def get_juggler(env, name: str = "juggler") -> CascadeCommand:
    return env.command_manager.get_term(name)


# ---------------------------------------------------------------------- reset event
def reset_cascade(env, env_ids, command_name: str = "juggler"):
    t = get_juggler(env, command_name)
    sp = t.sp
    n = len(env_ids)
    if n == 0:
        return
    dev = env.device
    if sp.random_phase:
        start = torch.randint(0, 2, (n,), device=dev) * t.beat_steps + torch.randint(
            sp.start_steps, t.beat_steps, (n,), device=dev
        )
    else:
        start = torch.full((n,), sp.start_steps, dtype=torch.long, device=dev)
    t.step_idx[env_ids] = start
    t.ep_steps[env_ids] = 0
    t.caught[env_ids] = 0.0
    t.throws[env_ids] = 0.0
    t.failed[env_ids] = False
    t.fail_code[env_ids] = 0
    t.pending_y[env_ids] = False
    t.last_ct[env_ids] = -1000
    t.just_reset[env_ids] = True
    t.init_plans(env_ids, t.step_idx)
    rp, rv, _ = t.ref_at(t.step_idx)
    for hi, h in enumerate(t.hands):
        dq = torch.einsum("ij,nj->ni", t.slide_jinv[hi], rp[env_ids, hi] - t.cup_home[hi])
        dv = torch.einsum("ij,nj->ni", t.slide_jinv[hi], rv[env_ids, hi])
        h.write_joint_state_to_sim(dq, dv, joint_ids=t.slide_ids[hi], env_ids=env_ids)
    bp, bv = t.nominal_ball_state(env_ids, t.step_idx)
    o = env.scene.env_origins[env_ids]
    q = torch.zeros(n, 4, device=dev)
    q[:, 0] = 1.0
    for b, ball in enumerate(t.balls):
        vel = torch.zeros(n, 6, device=dev)
        vel[:, :3] = bv[:, b]
        ball.write_root_link_pose_to_sim(torch.cat([bp[:, b] + o, q], -1), env_ids=env_ids)
        ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    # Land estimates = nominal catch point of each ball's next catching hand (right catches on odd beats).
    beat = t.beat_of(t.step_idx[env_ids])
    for b in range(3):
        k = beat - ((beat - b) % 3)  # last throw beat of ball b
        catcher = 1 - (k % 2)
        t.land_est[env_ids, b] = t.p_catch_nom[catcher]


# ---------------------------------------------------------------------- action: reference + residual
@dataclass(kw_only=True)
class ResidualActionCfg(BaseActionCfg):
    command_name: str = "juggler"
    hand_index: int = 0
    actuator_names: tuple[str, ...] | list[str] = (".*",)
    finger_names: tuple[str, ...] | None = None
    """Finger joint targets (None = Allegro's 16); the residual is 6 root + fingers + tendons."""
    tendon_names: tuple[str, ...] = ()
    held_joint_pos: dict = field(default_factory=dict)
    poses: dict | None = None
    """Finger poses by name (cup / hold / release): {name: {target: value}} (joints and tendons); None = finger_pose()."""

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT

    def build(self, env):
        return ResidualAction(self, env)


class ResidualAction(BaseAction):
    """Residual in [-1, 1] (6 root + the hand's finger targets; Allegro 22-D) on top of the command's reference."""

    cfg: ResidualActionCfg

    def __init__(self, cfg: ResidualActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        names = list(self._target_names)
        fingers = list(cfg.finger_names or ALLEGRO_FINGER_JOINTS)
        canonical = list(BASE_JOINTS) + fingers
        held = list(cfg.held_joint_pos)
        assert sorted(names) == sorted(canonical + held), names
        perm = [names.index(nm) for nm in canonical]
        self._jids = self._target_ids[torch.tensor(perm, device=self.device)]
        self._fingers = fingers
        self._nf = len(fingers)
        self._nt = len(cfg.tendon_names)
        dim = 6 + self._nf + self._nt
        self._action_dim = dim
        self._raw_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._env = env
        self._sub = 0
        self._poses = None
        self._target = torch.zeros(self.num_envs, 6 + self._nf, device=self.device)
        self._gravity = torch.tensor([0.0, 0.0, G], device=self.device)
        m = env.sim.mj_model
        # Root PD gains of the compiled model (Allegro: = the spec's slide gains; inertia-matched hands: scaled).
        slide_acts = [actuator_of(m, cfg.entity_name, jn) for jn in ("base_x", "base_y", "base_z")]
        self._kp = torch.tensor(
            [float(m.actuator_gainprm[a, 0]) for a in slide_acts], device=self.device
        )
        self._kv = torch.tensor(
            [-float(m.actuator_biasprm[a, 2]) for a in slide_acts], device=self.device
        )
        self._held_ids = (
            self._target_ids[torch.tensor([names.index(n) for n in held], device=self.device)]
            if held
            else None
        )
        self._held_pos = (
            torch.tensor([cfg.held_joint_pos[n] for n in held], device=self.device).expand(
                self.num_envs, -1
            )
            if held
            else None
        )
        if self._nt:
            tids, tnames = self._entity.find_tendons(list(cfg.tendon_names), preserve_order=True)
            assert list(tnames) == list(cfg.tendon_names), tnames
            self._tendon_ids = torch.tensor(tids, device=self.device, dtype=torch.long)
            lo_hi = np.array(
                [
                    m.actuator_ctrlrange[actuator_of(m, cfg.entity_name, tn, "tendon")]
                    for tn in cfg.tendon_names
                ]
            )
            lo_hi = torch.tensor(lo_hi, dtype=torch.float32, device=self.device)
            self._t_lo, self._t_hi = lo_hi[:, 0], lo_hi[:, 1]
            # A tendon target is the sum of two joints (J0 = J1 + J2): its residual covers the same angle fraction.
            self._t_res = (self._t_hi - self._t_lo) / np.pi

    def _pose(self, kind: str) -> torch.Tensor:
        """Finger pose (fingers then tendons)."""
        if self.cfg.poses is None:
            p = finger_pose(kind)
            return torch.tensor([p[j] for j in self._fingers], device=self.device)
        p = self.cfg.poses[kind]
        return torch.tensor(
            [p[j] for j in (*self._fingers, *self.cfg.tendon_names)], device=self.device
        )

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._sub = 0

    def apply_actions(self):
        t = get_juggler(self._env, self.cfg.command_name)
        sp = t.sp
        h = self.cfg.hand_index
        a = self._raw_actions.clamp(-1.0, 1.0)
        frac = (self._sub + 1) / self._env.cfg.decimation
        self._sub += 1
        rp, rv, ra = t.ref_at(t.step_idx, frac)
        p, v, acc = rp[:, h], rv[:, h], ra[:, h]
        jinv = t.slide_jinv[h]
        q_ref = torch.einsum("ij,nj->ni", jinv, p - t.cup_home[h])
        qd_ref = torch.einsum("ij,nj->ni", jinv, v)
        # Feed-forward force (world) m (a + g z) mapped to the slides with J^T, plus joint damping.
        f_w = t.mass[h] * (acc + self._gravity)
        f_j = torch.einsum("ji,nj->ni", t.slide_j[h], f_w) + t.slide_damping * qd_ref
        ctrl_slide = q_ref + (self._kv * qd_ref + f_j) / self._kp + a[:, :3] * sp.res_slide
        rot = a[:, 3:6] * sp.res_rot
        prior = self._prior_fingers(t, h, frac)
        nf = self._nf
        fing = prior[:, :nf] + a[:, 6 : 6 + nf] * sp.res_finger
        self._target = torch.cat([ctrl_slide, rot, fing], -1)
        self._entity.set_joint_position_target(self._target, joint_ids=self._jids)
        if self._held_ids is not None:
            self._entity.set_joint_position_target(self._held_pos, joint_ids=self._held_ids)
        if self._nt:
            tt = prior[:, nf:] + a[:, 6 + nf :] * sp.res_finger * self._t_res
            tt = torch.minimum(torch.maximum(tt, self._t_lo), self._t_hi)
            self._entity.set_tendon_len_target(tt, tendon_ids=self._tendon_ids)

    def _prior_fingers(self, t, h, frac):
        """Designed finger schedule (CascadeSpec.hold_pose / release_pose) for hand h at the current substep."""
        sp = t.sp
        if self._poses is None:
            self._poses = {k: self._pose(k) for k in {"cup", sp.hold_pose, sp.release_pose}}
        cup, hold, rel = self._poses["cup"], self._poses[sp.hold_pose], self._poses[sp.release_pose]
        beat = t.beat_of(t.step_idx)
        tin = (
            (t.step_idx - beat * t.beat_steps).float() + frac
        ) * t.dt  # time since the segment start
        dwell = t.hand_in_dwell(beat)[:, h]
        T = sp.beat_s
        # dwell: hold pose, release pose from T - lead; vacant: release pose for hold_s, then hold pose
        w = (tin / max(sp.grip_ramp_s, 1e-3)).clamp(0.0, 1.0).unsqueeze(-1)
        in_dwell = (1 - w) * cup + w * hold
        in_dwell = torch.where(
            (tin >= T - sp.release_lead_s).unsqueeze(-1), rel.expand_as(in_dwell), in_dwell
        )
        # vacant: release pose, then back to the hold pose (the ball is caught into the formed grip)
        in_vac = torch.where(
            (tin < sp.release_hold_s).unsqueeze(-1),
            rel.expand(tin.shape[0], -1),
            hold.expand(tin.shape[0], -1),
        )
        return torch.where(dwell.unsqueeze(-1), in_dwell, in_vac)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


# ---------------------------------------------------------------------- observations / rewards / terminations
def obs_hands(env, command_name="juggler"):
    t = get_juggler(env, command_name)
    rp, rv, _ = t.ref_at(t.step_idx)
    cp, cv = t.cup_pos(), t.cup_vel()
    return torch.cat(
        [
            (cp - t.cup_home.unsqueeze(0)).flatten(1),
            cv.flatten(1),
            (cp - rp).flatten(1),
            (cv - rv).flatten(1),
            (rp - t.cup_home.unsqueeze(0)).flatten(1),
            rv.flatten(1),
        ],
        -1,
    )


def obs_balls(env, command_name="juggler"):
    t = get_juggler(env, command_name)
    bp, bv = t.ball_pos(), t.ball_vel()
    cp = t.cup_pos()
    rel = bp.unsqueeze(1) - cp.unsqueeze(2)  # (N,2,3,3)
    return torch.cat(
        [
            (bp - t.cup_home[0].view(1, 1, 3)).flatten(1),
            bv.flatten(1),
            rel.flatten(1),
            t.contact.float().flatten(1),
        ],
        -1,
    )


def obs_phase(env, command_name="juggler"):
    t = get_juggler(env, command_name)
    beat = t.beat_of(t.step_idx)
    r = (t.step_idx - beat * t.beat_steps).float() / t.beat_steps
    dwell = t.hand_in_dwell(beat).float()
    held = torch.nn.functional.one_hot(t.held_ball(beat), 3).float()
    inc = torch.nn.functional.one_hot(t.incoming_ball(beat), 3).float()
    # throw command and catch target relative to the nominal points
    return torch.cat(
        [
            dwell,
            r.unsqueeze(-1),
            held,
            inc,
            (t.u_cmd - t.p_catch_nom.flip(0).unsqueeze(0)).flatten(1),
            (t.catch_cmd - t.p_catch_nom.unsqueeze(0)).flatten(1),
        ],
        -1,
    )


def reward_catch(env, command_name="juggler"):
    return get_juggler(env, command_name).catch_now


def reward_land(env, command_name="juggler", sigma: float = 0.05):
    t = get_juggler(env, command_name)
    return torch.where(
        t.land_err_now > 0, torch.exp(-t.land_err_now / sigma), torch.zeros_like(t.land_err_now)
    )


def reward_alive(env, command_name="juggler"):
    return (~get_juggler(env, command_name).failed).float()


def reward_failure(env, command_name="juggler"):
    return get_juggler(env, command_name).failed.float()


def terminated_failed(env, command_name="juggler"):
    return get_juggler(env, command_name).failed


def reward_residual(env):
    return env.action_manager.action.square().mean(-1)


# ---------------------------------------------------------------------- env config
def _ws(hand, side, y, sp: CascadeSpec, prof=None) -> HandRootWorkspace:
    if prof is None:
        pos, rot = (0.0, y, sp.anchor_z), (1.0, 0.0, 0.0, 0.0)
    else:  # palm up, fingers +x (canonical base axes, hand_profile.py); palm origin at the Allegro anchor
        rot = prof.anchor_rot(down=False, fingers_px=True)
        pos, side = prof.anchor_pos((0.0, y, sp.anchor_z), rot), prof.side
    return HandRootWorkspace(
        side=side,
        entity_name=hand,
        anchor_pos=pos,
        anchor_rot=rot,
        joint_bounds={
            "base_x": (-0.45, 0.45),
            "base_y": (-0.2, 0.2),
            "base_z": (-0.2, 0.2),
            "base_roll": (-0.3, 0.3),
            "base_pitch": (-0.3, 0.3),
            "base_yaw": (-0.3, 0.3),
        },
        lin_vel_max=10.0,
        ang_vel_max=10.0,
        reset_noise={n: (0.0, 0.0) for n in BASE_JOINTS},
        notes=f"{side} palm-up hand, cascade",
    )


def _contact(name, a, pa, b, pb):
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="subtree", pattern=pa, entity=a),
        secondary=ContactMatch(mode="subtree", pattern=pb, entity=b),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )


@dataclass
class _HandParts:
    """Everything in the env config that depends on the hand model (Allegro or a measured profile)."""

    command: CascadeCommandCfg
    actions: dict
    joint_cfgs: dict
    root_bodies: dict
    entities: dict


def _allegro_parts(sp: CascadeSpec) -> _HandParts:
    wsr, wsl = _ws("hand_r", "right", -sp.hand_y, sp), _ws("hand_l", "left", sp.hand_y, sp)
    hr = make_hand_entity_cfg(
        ALLEGRO_HAND_RIGHT, wsr, base_gains=sp.slide_gains, rot_gains=sp.rot_gains
    )
    hl = make_hand_entity_cfg(
        ALLEGRO_HAND_LEFT, wsl, base_gains=sp.slide_gains, rot_gains=sp.rot_gains
    )
    hr.init_state.joint_pos = finger_pose(sp.hold_pose)
    hl.init_state.joint_pos = finger_pose(sp.hold_pose)
    return _HandParts(
        command=CascadeCommandCfg(spec=sp),
        actions={h: ResidualActionCfg(entity_name=h, hand_index=i) for i, h in enumerate(HANDS)},
        joint_cfgs={h: hand_joint_cfg(h) for h in HANDS},
        root_bodies={h: "palm" for h in HANDS},
        entities={"hand_r": hr, "hand_l": hl},
    )


def _profile_parts(sp: CascadeSpec) -> _HandParts:
    hp = hand_cascade_params(sp.hand)
    # Model side per hand position: a side without a stable grip uses the other side's model (juggle3_probe.py).
    profs = [
        get_profile(sp.hand, hp.get("sides", {}).get(s, {}).get("model_side", s)) for s in SIDES
    ]
    entities, actions = {}, {}
    for i, (h, p, s) in enumerate(zip(HANDS, profs, SIDES)):
        ws = _ws(h, s, (-1 if s == "right" else 1) * sp.hand_y, sp, p)
        e = make_hand_entity_cfg(
            p.spec, ws, base_gains=sp.slide_gains, rot_gains=sp.rot_gains, profile=p
        )
        init = dict(e.init_state.joint_pos or {})
        init.update(
            {j: v for j, v in hp["poses"][s][sp.hold_pose].items() if j not in p.tendon_targets}
        )
        init.update(dict(p.fixed_joints))
        e.init_state.joint_pos = init
        entities[h] = e
        actions[h] = ResidualActionCfg(
            entity_name=h,
            hand_index=i,
            finger_names=tuple(p.finger_joints),
            tendon_names=tuple(p.tendon_targets),
            held_joint_pos=dict(p.fixed_joints),
            poses=hp["poses"][s],
        )
    command = CascadeCommandCfg(
        spec=sp,
        palm_body=tuple(p.palm_body for p in profs),
        cup_rel_palm=tuple(tuple(hp["cup_rel_palm"][s]) for s in SIDES),
    )
    return _HandParts(
        command=command,
        actions=actions,
        joint_cfgs={h: hand_joint_cfg(h, p) for h, p in zip(HANDS, profs)},
        root_bodies={h: p.root_body for h, p in zip(HANDS, profs)},
        entities=entities,
    )


def make_cascade_env_cfg(sp: CascadeSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
    sp = copy.deepcopy(sp)
    if play:
        sp.random_phase = False
    parts = _allegro_parts(sp) if sp.hand == "allegro" else _profile_parts(sp)
    jc, root = parts.joint_cfgs, parts.root_bodies
    obs = {
        "joint_pos_r": ObservationTermCfg(
            func=tmdp.hand_joint_pos_unit, params={"asset_cfg": jc["hand_r"]}
        ),
        "joint_pos_l": ObservationTermCfg(
            func=tmdp.hand_joint_pos_unit, params={"asset_cfg": jc["hand_l"]}
        ),
        "joint_vel_r": ObservationTermCfg(
            func=tmdp.hand_joint_vel, params={"asset_cfg": jc["hand_r"]}
        ),
        "joint_vel_l": ObservationTermCfg(
            func=tmdp.hand_joint_vel, params={"asset_cfg": jc["hand_l"]}
        ),
        "hands": ObservationTermCfg(func=obs_hands),
        "balls": ObservationTermCfg(func=obs_balls),
        "phase": ObservationTermCfg(func=obs_phase),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(dict(obs), enable_corruption=False),
        "critic": ObservationGroupCfg(dict(obs), enable_corruption=False),
    }
    events = {
        "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_cascade": EventTermCfg(func=reset_cascade, mode="reset"),
    }
    rewards = {
        "catch": RewardTermCfg(func=reward_catch, weight=1.0),
        "land": RewardTermCfg(func=reward_land, weight=1.0),
        "alive": RewardTermCfg(func=reward_alive, weight=0.02),
        "failure": RewardTermCfg(func=reward_failure, weight=-5.0),
        "residual": RewardTermCfg(func=reward_residual, weight=sp.reward_residual),
    }
    if sp.finger_action_rate and not play:
        rewards["finger_action_rate"] = RewardTermCfg(
            func=tmdp.finger_action_rate_l2, weight=sp.finger_action_rate, params={"hands": HANDS}
        )
    terms = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "failed": TerminationTermCfg(func=terminated_failed),
        "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    }
    sensors = [_contact(f"{h}_{b}_contact", h, root[h], b, "ball") for h in HANDS for b in BALLS]
    sensors += [_contact("hand_hand_contact", "hand_r", root["hand_r"], "hand_l", root["hand_l"])]
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.5,
            sensors=tuple(sensors),
        ),
        observations=observations,
        actions=parts.actions,
        commands={"juggler": parts.command},
        events=events,
        rewards=rewards,
        terminations=terms,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.9),
            distance=1.8,
            elevation=-8.0,
            azimuth=180.0,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=1200,
            mujoco=MujocoCfg(
                timestep=sp.timestep, iterations=50, ls_iterations=50, impratio=10, cone="elliptic"
            ),
        ),
        decimation=sp.decimation,
        episode_length_s=sp.episode_s,
        scale_rewards_by_dt=False,
        is_finite_horizon=False,
    )
    ents = dict(parts.entities)
    for b, rgba in zip(BALLS, BALL_RGBA):
        ents[b] = EntityCfg(
            spec_fn=lambda rgba=rgba: make_ball_spec(
                sp.ball_radius, sp.ball_mass, sp.ball_solref, (1.0, 0.005, 0.0005), rgba=rgba
            )
        )
    cfg.scene.entities = ents
    return cfg
