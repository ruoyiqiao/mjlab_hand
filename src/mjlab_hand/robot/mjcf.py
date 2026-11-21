"""Load packaged MJCF meshes without changing model physics."""

from pathlib import Path

import mujoco


def load_mjcf(path: Path, joint_renames: dict[str, str] | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(path))
    mesh_root = path.parent / spec.meshdir
    assets = {}
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        source = mesh_root / mesh.file
        key = f"{path.stem}/{Path(mesh.file).name}"
        assets[key] = source.read_bytes()
        mesh.file = key
    spec.meshdir = ""
    spec.assets = assets
    if joint_renames:
        # The task's reset and delta-action code uses these six canonical names.
        for joint in spec.joints:
            joint.name = joint_renames.get(joint.name, joint.name)
        for actuator in spec.actuators:
            actuator.target = joint_renames.get(actuator.target, actuator.target)
    return spec
