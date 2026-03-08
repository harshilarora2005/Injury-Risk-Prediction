import numpy as np
from parse_asf import euler2mat


def compute_joint_positions(motions, joints):
    """
    Compute world-space joint positions for every frame using the
    CalciferZh formulas:

        root  : matrix = C @ euler2mat(*rotation) @ Cinv
        child : matrix = parent.matrix @ C @ euler2mat(*rotation) @ Cinv
        coord : parent.coordinate + length * matrix @ direction

    DOF-aware: only the axes listed in joint.dof are populated from
    the AMC values — e.g. a knee with dof=['rx'] only fills rotation[0].

    Parameters
    ----------
    motions : list of dicts  { joint_name -> [float, ...] }  (from parse_amc)
    joints  : dict           { joint_name -> Joint }         (from parse_asf)

    Returns
    -------
    all_positions : list of dicts  { joint_name -> np.array([x, y, z]) }
    """
    all_positions = []

    for motion in motions:
        _set_motion(joints['root'], motion)
        frame_coords = {}
        for name, joint in joints.items():
            if joint.coordinate is not None:
                frame_coords[name] = np.squeeze(joint.coordinate)
        all_positions.append(frame_coords)

    return all_positions


def _set_motion(joint, motion):
    """Recursively apply one frame of motion data to the joint tree."""

    if joint.name == 'root':
        vals = motion.get('root', [0, 0, 0, 0, 0, 0])
        joint.coordinate = np.reshape(np.array(vals[:3], dtype=float), [3, 1])
        rotation         = np.deg2rad(vals[3:6])
        joint.matrix     = joint.C @ euler2mat(*rotation) @ joint.Cinv

    else:
        # Only fill the axes this joint actually has (DOF-aware)
        rotation = np.zeros(3)
        if joint.name in motion:
            vals = motion[joint.name]
            idx  = 0
            for ax in joint.dof:
                if   ax == 'rx': rotation[0] = vals[idx]
                elif ax == 'ry': rotation[1] = vals[idx]
                elif ax == 'rz': rotation[2] = vals[idx]
                idx += 1

        rotation     = np.deg2rad(rotation)
        joint.matrix = joint.parent.matrix @ joint.C @ euler2mat(*rotation) @ joint.Cinv
        joint.coordinate = (
            joint.parent.coordinate + joint.length * joint.matrix @ joint.direction
        )

    for child in joint.children:
        _set_motion(child, motion)