"""Check native tactile depth and slip on a deterministic two-environment scene."""

import argparse
import json

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ObjRef, TactileSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg

PAD = """
<mujoco><worldbody><body name="pad">
  <geom name="pad" type="box" size="0.02 0.02 0.002"/>
</body></worldbody></mujoco>
"""
BALL = """
<mujoco><worldbody><body name="ball"><freejoint/>
  <geom type="sphere" size="0.005" mass="0.01"/>
</body></worldbody></mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scene = Scene(
        SceneCfg(
            num_envs=2,
            env_spacing=1.0,
            entities={
                "pad": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(PAD)),
                "ball": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(BALL)),
            },
            sensors=(
                TactileSensorCfg(
                    name="tactile",
                    geom=ObjRef("geom", "pad", entity="pad"),
                    size=(0.01, 0.01),
                    resolution=(5, 5),
                ),
            ),
        ),
        args.device,
    )
    sim = Simulation(
        num_envs=2,
        cfg=SimulationCfg(njmax=20),
        model=scene.compile(),
        device=args.device,
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    if scene.sensor_context is not None:
        sim.set_sensor_context(scene.sensor_context)
    # First ball penetrates by 1 mm and slides at 0.1 m/s; second is clear.
    state = torch.tensor(
        [
            [0, 0, 0.006, 1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0],
            [0, 0, 0.030, 1, 0, 0, 0, 0.0, 0, 0, 0, 0, 0],
        ],
        device=args.device,
        dtype=torch.float32,
    )
    scene["ball"].write_root_state_to_sim(state)
    sim.forward()
    sensor = scene["pad/tactile"]
    depth, slip = sensor.data.depth, sensor.data.slip
    assert depth.shape == (2, 5, 5) and slip.shape == (2, 5, 5, 2)
    torch.testing.assert_close(
        depth[0, 2, 2],
        torch.tensor(0.001, device=args.device),
        atol=1e-6,
        rtol=0,
    )
    expected_depth = (0.005 - (sensor.taxel_pos_w()[0] - state[0, :3]).norm(dim=-1)).clamp(min=0)
    torch.testing.assert_close(depth[0], expected_depth.view(5, 5), atol=1e-6, rtol=0)
    assert torch.count_nonzero(slip[0, ..., 1]) == 0
    torch.testing.assert_close(slip[0, ..., 0], 0.1 * (depth[0] > 0).float())
    assert torch.isfinite(depth).all() and torch.isfinite(slip).all()
    assert torch.count_nonzero(depth[1]) == 0
    assert torch.count_nonzero(slip[1]) == 0
    print(
        json.dumps(
            {
                "device": args.device,
                "depth_shape": list(depth.shape),
                "slip_shape": list(slip.shape),
                "center_depth_m": depth[0, 2, 2].item(),
                "center_slip_m_s": slip[0, 2, 2].tolist(),
                "noncontact_depth_sum": depth[1].sum().item(),
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
