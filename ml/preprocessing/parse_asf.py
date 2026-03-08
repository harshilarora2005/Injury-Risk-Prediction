import numpy as np
from math import cos, sin, radians


def euler2mat(rx=0., ry=0., rz=0.):
    """Angles in RADIANS → 3x3 rotation matrix (Rz @ Ry @ Rx)."""
    Rx = np.array([[1, 0, 0],
                   [0, cos(rx), -sin(rx)],
                   [0, sin(rx),  cos(rx)]])
    Ry = np.array([[ cos(ry), 0, sin(ry)],
                   [0, 1, 0],
                   [-sin(ry), 0, cos(ry)]])
    Rz = np.array([[cos(rz), -sin(rz), 0],
                   [sin(rz),  cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


class Joint:
    def __init__(self, name, direction, length, axis, dof):
        self.name      = name
        self.direction = np.reshape(direction, [3, 1])
        self.length    = length
        self.dof       = dof  # e.g. ['rx', 'ry', 'rz'] or ['rx'] for knee

        axis_rad  = np.deg2rad(axis)
        self.C    = euler2mat(*axis_rad)
        self.Cinv = np.linalg.inv(self.C)

        self.parent     = None
        self.children   = []
        self.coordinate = None  # (3,1) world-space position, set by set_motion
        self.matrix     = None  # (3,3) world-space rotation, set by set_motion


def parse_asf(asf_path):
    """
    Parse a .asf skeleton file.

    Returns
    -------
    joints : dict { joint_name -> Joint }  includes 'root'
    """
    with open(asf_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    joints = {}
    i = 0

    while i < len(lines):
        line = lines[i]

        # ── bonedata ──────────────────────────────────────────────────────────
        if line == ':bonedata':
            i += 1
            while i < len(lines) and lines[i] != ':hierarchy':
                if lines[i] == 'begin':
                    name      = None
                    direction = np.zeros(3)
                    length    = 0.0
                    axis      = np.zeros(3)
                    dof       = []
                    i += 1
                    while i < len(lines) and lines[i] != 'end':
                        parts = lines[i].split()
                        if not parts:
                            i += 1
                            continue
                        key = parts[0]
                        if key == 'name':
                            name = parts[1]
                        elif key == 'direction':
                            direction = np.array(list(map(float, parts[1:4])))
                        elif key == 'length':
                            length = float(parts[1])
                        elif key == 'axis':
                            axis = np.array(list(map(float, parts[1:4])))
                        elif key == 'dof':
                            dof = parts[1:]
                        i += 1
                    if name:
                        joints[name] = Joint(name, direction, length, axis, dof)
                i += 1
            continue

        # ── hierarchy ─────────────────────────────────────────────────────────
        if line == ':hierarchy':
            i += 1
            while i < len(lines) and lines[i] != 'end':
                parts = lines[i].split()
                if len(parts) >= 2:
                    parent_name = parts[0]
                    for child_name in parts[1:]:
                        if parent_name in joints and child_name in joints:
                            joints[parent_name].children.append(joints[child_name])
                            joints[child_name].parent = joints[parent_name]
                i += 1

        i += 1

    # Root is defined in the hierarchy block but has no bonedata entry
    root = Joint(
        name='root',
        direction=np.zeros(3),
        length=0,
        axis=np.zeros(3),
        dof=['rx', 'ry', 'rz'],
    )
    joints['root'] = root

    # Wire root's children (re-scan hierarchy for 'root' as parent)
    with open(asf_path, 'r') as f:
        lines2 = [l.strip() for l in f.readlines()]
    i = 0
    while i < len(lines2):
        if lines2[i] == ':hierarchy':
            i += 1
            while i < len(lines2) and lines2[i] != 'end':
                parts = lines2[i].split()
                if len(parts) >= 2:
                    parent_name = parts[0]
                    for child_name in parts[1:]:
                        if parent_name in joints and child_name in joints:
                            child  = joints[child_name]
                            parent = joints[parent_name]
                            if child not in parent.children:
                                parent.children.append(child)
                            child.parent = parent
                i += 1
        i += 1

    return joints