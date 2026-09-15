"""sensor_mount — build123d script driven entirely by PARAMS."""
from build123d import *

PARAMS = {
    "plate_x": (80.0, "spec:envelope.extents.x"),
    "plate_y": (50.0, "spec:envelope.extents.y"),
    "plate_t": (6.0, "derived:planes.top.offset + planes.bottom.offset"),
    "top_z": (3.0, "spec:planes.top.offset"),
    "mount_hole_d": (4.3, "spec:features.mount_holes.diameter"),
    "mount_hole_x": (35.0, "spec:features.mount_holes.positions[*][0]"),
    "mount_hole_y": (20.0, "spec:features.mount_holes.positions[*][1]"),
    "window_d": (20.0, "spec:features.window.diameter"),
    "window_x": (0.0, "spec:features.window.positions[0][0]"),
    "window_y": (0.0, "spec:features.window.positions[0][1]"),
    "boss_d": (16.0, "spec:features.boss.diameter"),
    "boss_h": (8.0, "spec:features.boss.height"),
    "boss_x": (-25.0, "spec:features.boss.positions[0][0]"),
    "boss_y": (0.0, "spec:features.boss.positions[0][1]"),
    "seat_d": (8.0, "spec:features.bearing_seat.diameter"),
    "seat_depth": (6.0, "spec:features.bearing_seat.length"),
    "boss_top_z": (11.0, "derived:top_z + boss_h"),
    "seat_bottom_z": (5.0, "derived:boss_top_z - seat_depth"),
}
P = {k: v[0] for k, v in PARAMS.items()}

# Base plate, centred on the origin, spanning z = -plate_t/2 .. +plate_t/2
plate = Box(P["plate_x"], P["plate_y"], P["plate_t"])

# Boss standing on the top face (z = top_z), extending up by boss_h
boss = Pos(P["boss_x"], P["boss_y"], P["top_z"]) * Cylinder(
    P["boss_d"] / 2, P["boss_h"], align=(Align.CENTER, Align.CENTER, Align.MIN)
)
body = plate + boss

# Through holes in the plate (mount holes in the 4 corners, central window)
mount_pts = [
    (-P["mount_hole_x"], -P["mount_hole_y"]),
    (P["mount_hole_x"], -P["mount_hole_y"]),
    (-P["mount_hole_x"], P["mount_hole_y"]),
    (P["mount_hole_x"], P["mount_hole_y"]),
]
for (hx, hy) in mount_pts:
    body -= Pos(hx, hy, 0) * Cylinder(P["mount_hole_d"] / 2, P["plate_t"])

body -= Pos(P["window_x"], P["window_y"], 0) * Cylinder(P["window_d"] / 2, P["plate_t"])

# Blind bearing seat bored into the boss from its top face
body -= Pos(P["boss_x"], P["boss_y"], P["seat_bottom_z"]) * Cylinder(
    P["seat_d"] / 2, P["seat_depth"], align=(Align.CENTER, Align.CENTER, Align.MIN)
)

result = body
