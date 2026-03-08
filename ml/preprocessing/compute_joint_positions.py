import numpy as np

def rotation_matrix(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])

    Rx = np.array([
        [1,0,0],
        [0,np.cos(rx),-np.sin(rx)],
        [0,np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [ np.cos(ry),0,np.sin(ry)],
        [0,1,0],
        [-np.sin(ry),0,np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz),-np.sin(rz),0],
        [np.sin(rz), np.cos(rz),0],
        [0,0,1]
    ])

    return Rz @ Ry @ Rx


def compute_joint_positions(frames, bones):

    all_positions = []

    for frame in frames:
        positions = {}

        root_vals = frame["root"]
        root_pos = np.array(root_vals[:3])

        positions["root"] = root_pos

        for joint, bone in bones.items():

            if joint not in frame:
                continue

            parent = "root"  # simplified assumption

            direction = np.array(bone["direction"])
            length = bone["length"]

            rot_vals = frame[joint]

            if len(rot_vals) == 3:
                R = rotation_matrix(*rot_vals)
                direction = R @ direction

            pos = positions[parent] + direction * length
            positions[joint] = pos

        all_positions.append(positions)

    return all_positions