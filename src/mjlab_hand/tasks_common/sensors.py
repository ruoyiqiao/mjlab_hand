"""Entity-parameterized contact sensors for the Allegro tasks.

Every sensor names its entities explicitly (no implicit ``"hand"`` default), so a
scene with ``hand_left`` and ``hand_right`` gets independent sensors.
"""

from __future__ import annotations

from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_hand.tasks_common.hand_profile import HandProfile, get_profile

ALLEGRO_TIP_GEOMS = r"(ff|mf|rf|th)_tip_collision"


def _root(profile: HandProfile | None) -> str:
    """Hand-subtree root body (covers the whole hand); Allegro's is its palm."""
    return (profile or get_profile("allegro")).root_body


def hand_object_contact(
    hand: str, obj: str = "object", profile: HandProfile | None = None
) -> ContactSensorCfg:
    """Any contact between the whole hand subtree and the object (net wrench)."""
    return ContactSensorCfg(
        name=f"{hand}_object_contact",
        primary=ContactMatch(mode="subtree", pattern=_root(profile), entity=hand),
        secondary=ContactMatch(mode="subtree", pattern="object", entity=obj),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        global_frame=True,
    )


def tip_object_contact(
    hand: str, obj: str = "object", profile: HandProfile | None = None
) -> ContactSensorCfg:
    """Per-fingertip contact with the object (tip collision geoms of the hand profile)."""
    return ContactSensorCfg(
        name=f"{hand}_tip_contact",
        primary=ContactMatch(
            mode="geom", pattern=(profile or get_profile("allegro")).tip_geoms, entity=hand
        ),
        secondary=ContactMatch(mode="subtree", pattern="object", entity=obj),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        global_frame=True,
    )


def object_table_contact(obj: str = "object", table: str = "table") -> ContactSensorCfg:
    return ContactSensorCfg(
        name="object_table_contact",
        primary=ContactMatch(mode="subtree", pattern="object", entity=obj),
        secondary=ContactMatch(mode="geom", pattern="table_surface", entity=table),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )


def hand_table_contact(
    hand: str, table: str = "table", profile: HandProfile | None = None
) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=f"{hand}_table_contact",
        primary=ContactMatch(mode="subtree", pattern=_root(profile), entity=hand),
        secondary=ContactMatch(mode="geom", pattern="table_surface", entity=table),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        global_frame=True,
    )


def hand_hand_contact(
    hand_a: str = "hand_left",
    hand_b: str = "hand_right",
    history: int = 4,
    profile_a: HandProfile | None = None,
    profile_b: HandProfile | None = None,
) -> ContactSensorCfg:
    """Inter-hand contact; per-substep penetration history catches brief contacts."""
    return ContactSensorCfg(
        name="hand_hand_contact",
        primary=ContactMatch(mode="subtree", pattern=_root(profile_a), entity=hand_a),
        secondary=ContactMatch(mode="subtree", pattern=_root(profile_b), entity=hand_b),
        fields=("found", "force", "dist"),
        reduce="mindist",
        num_slots=1,
        history_length=history,
    )


def object_bucket_contact(obj: str = "object", bucket: str = "bucket") -> ContactSensorCfg:
    return ContactSensorCfg(
        name="object_bucket_contact",
        primary=ContactMatch(mode="subtree", pattern="object", entity=obj),
        secondary=ContactMatch(mode="subtree", pattern="bucket", entity=bucket),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )


def hand_bucket_contact(
    hand: str, bucket: str = "bucket", profile: HandProfile | None = None
) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=f"{hand}_bucket_contact",
        primary=ContactMatch(mode="subtree", pattern=_root(profile), entity=hand),
        secondary=ContactMatch(mode="subtree", pattern="bucket", entity=bucket),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
