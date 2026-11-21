"""Rebuild packaged XHand MJCF from pinned DexScrew URDF and nominal parameters.

Run with the project's MuJoCo 3.11 environment. Source files remain unchanged.
"""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1] / "src/mjlab_hand/robot/xhand"


def build() -> None:
    params = json.loads((ROOT / "dexscrew_parameters.json").read_text())
    source = ROOT / "assets/dexscrew/xhand_left/urdf/xhand_left.urdf"
    urdf = ET.parse(source).getroot()
    compiler = ET.SubElement(ET.SubElement(urdf, "mujoco"), "compiler")
    compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false")
    for link in urdf.findall("link"):
        inertia = link.find("inertial/inertia")
        # Seventeen fixed CAD placeholders have all six entries = 1e-11,
        # a singular, non-physical tensor rejected by MuJoCo. Keep mass, COM,
        # and diagonal entries; zero only the placeholder cross terms.
        if inertia is not None and all(float(v) == 1e-11 for v in inertia.attrib.values()):
            inertia.attrib.update(ixy="0", ixz="0", iyz="0")
    for mesh in urdf.iter("mesh"):
        mesh.set("filename", str((source.parent / mesh.attrib["filename"]).resolve()))
    spec = mujoco.MjSpec.from_string(ET.tostring(urdf, encoding="unicode"))
    spec.compile()
    root = ET.fromstring(spec.to_xml())
    root.set("model", "dexscrew_xhand_left")
    root.find("compiler").set("meshdir", "../assets/dexscrew/xhand_left/meshes")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)
    ET.SubElement(root, "option", timestep=str(params["sim_dt"]), integrator="implicitfast")
    order = {name: i for i, name in enumerate(params["joint_order"])}
    for parent in root.iter("body"):
        # Explicit policy order: index, middle, pinky, ring, thumb.
        children = list(parent.findall("body"))
        for child in children:
            parent.remove(child)
        children.sort(
            key=lambda b: min((order[j.attrib["name"]] for j in b.iter("joint")), default=-1)
        )
        parent.extend(children)
        parent.set("gravcomp", "1")
        counts = {"visual": 0, "collision": 0}
        for geom in parent.findall("geom"):
            visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            kind = "visual" if visual else "collision"
            suffix = "" if counts[kind] == 0 else f"_{counts[kind]}"
            geom.set("name", f"{parent.attrib['name']}_{kind}{suffix}")
            counts[kind] += 1
            geom.set("group", "1" if visual else "3")
            if not visual:
                # DexScrew create_actor(..., collision_filter=-1) disables hand self-contact.
                geom.set("contype", "2")
                geom.set("conaffinity", "1")
                geom.set("friction", f"{params['nominal_sliding_friction']} 0.005 0.0001")
    actuator = ET.SubElement(root, "actuator")
    joints = {j.attrib["name"]: j for j in root.iter("joint")}
    for i, name in enumerate(params["joint_order"]):
        lo, hi = params["lower_limits"][i], params["upper_limits"][i]
        effort = params["effort_limits"][i]
        joints[name].attrib.update(
            range=f"{lo} {hi}",
            limited="true",
            damping=str(params["joint_damping"]),
            frictionloss=str(params["frictionloss"]),
            armature=str(params["armature"]),
            actuatorfrcrange=f"{-effort} {effort}",
            actuatorfrclimited="true",
        )
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=str(params["kp"]),
            kv=str(params["kd"]),
            ctrlrange=f"{lo} {hi}",
            ctrllimited="true",
            forcerange=f"{-effort} {effort}",
            forcelimited="true",
        )
    floating = copy.deepcopy(root)
    floating.set("model", "dexscrew_xhand_left_floating")
    world = floating.find("worldbody")
    palm = world.find("body")
    world.remove(palm)
    base = ET.SubElement(world, "body", name="xhand_float_base", gravcomp="1")
    ET.SubElement(base, "inertial", pos="0 0 0", mass="0.000001", diaginertia="2e-12 2e-12 2e-12")
    float_acts = floating.find("actuator")
    for i, (axis_name, axis) in enumerate(
        zip(("x", "y", "z", "roll", "pitch", "yaw"), ("1 0 0", "0 1 0", "0 0 1") * 2, strict=True)
    ):
        name = f"base_{axis_name}"
        trans = i < 3
        limits = "-2 2" if trans else "-3.141592653589793 3.141592653589793"
        ET.SubElement(
            base,
            "joint",
            name=name,
            type="slide" if trans else "hinge",
            axis=axis,
            limited="true",
            range=limits,
            armature="10" if trans else "0.1",
        )
        float_acts.insert(
            i,
            ET.Element(
                "position",
                name=name,
                joint=name,
                kp="1000" if trans else "100",
                kv="100" if trans else "10",
                ctrlrange=limits,
                forcerange="-200 200" if trans else "-20 20",
            ),
        )
    base.append(palm)
    out = ROOT / "xmls"
    out.mkdir(exist_ok=True)
    for filename, tree in (("left.xml", root), ("left_floating.xml", floating)):
        ET.indent(tree, space="  ")
        path = out / filename
        path.write_text(ET.tostring(tree, encoding="unicode") + "\n")
        model = mujoco.MjModel.from_xml_path(str(path))
        print(filename, "nq", model.nq, "nu", model.nu, "mass", model.body_mass.sum())


if __name__ == "__main__":
    build()
