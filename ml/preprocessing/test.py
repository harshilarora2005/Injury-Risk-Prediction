import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from parse_asf import parse_asf
from parse_amc import parse_amc
from parse_asf_geometry import parse_asf_geometry
from compute_joint_positions import compute_joint_positions


# ------------------------------------------------
# SKELETON CONNECTIONS
# ------------------------------------------------

BONES = [
    ("root","lowerback"),
    ("lowerback","upperback"),
    ("upperback","thorax"),
    ("thorax","lowerneck"),
    ("lowerneck","upperneck"),
    ("upperneck","head"),

    ("thorax","lclavicle"),
    ("lclavicle","lhumerus"),
    ("lhumerus","lradius"),
    ("lradius","lwrist"),
    ("lwrist","lhand"),

    ("thorax","rclavicle"),
    ("rclavicle","rhumerus"),
    ("rhumerus","rradius"),
    ("rradius","rwrist"),
    ("rwrist","rhand"),

    ("root","lfemur"),
    ("lfemur","ltibia"),
    ("ltibia","lfoot"),
    ("lfoot","ltoes"),

    ("root","rfemur"),
    ("rfemur","rtibia"),
    ("rtibia","rfoot"),
    ("rfoot","rtoes"),
]


ASF_PATH = "data/raw/cmu_mocap/subject_01/skeleton.asf"
AMC_PATH = "data/raw/cmu_mocap/subject_01/motions/01_01.amc"


# ------------------------------------------------
# LOAD MOTION DATA
# ------------------------------------------------

def load_motion_data():

    joints, hierarchy = parse_asf(ASF_PATH)
    frames = parse_amc(AMC_PATH)

    print("Number of joints:", len(joints))
    print("Number of frames:", len(frames))

    return joints, hierarchy, frames


# ------------------------------------------------
# EXTRACT KNEE ANGLE SIGNAL
# ------------------------------------------------

def extract_knee_angles(frames):

    knee_angles = []
    frame_indices = []

    for i, frame in enumerate(frames):

        if "ltibia" in frame:
            knee_angles.append(frame["ltibia"][0])
            frame_indices.append(i)

    knee_angles = np.array(knee_angles)

    return frame_indices, knee_angles


# ------------------------------------------------
# PLOT KNEE ANGLE
# ------------------------------------------------

def plot_knee_angles(frame_indices, knee_angles):

    plt.figure(figsize=(10,4))
    plt.plot(frame_indices, knee_angles)

    plt.xlabel("Frame")
    plt.ylabel("Knee Angle (deg)")
    plt.title("Left Knee Angle Over Time")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# PHASE SPACE PLOT
# ------------------------------------------------

def plot_phase_space(frame_indices, knee_angles):

    velocity = np.gradient(knee_angles)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(frame_indices, knee_angles, velocity)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Velocity")

    ax.set_title("Knee Motion Phase Space")

    plt.show()


# ------------------------------------------------
# DRAW SKELETON
# ------------------------------------------------

def plot_skeleton(frame_positions, ax, scale):

    xs, ys, zs = [], [], []

    for joint, pos in frame_positions.items():

        pos = pos * scale

        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

    ax.scatter(xs, ys, zs, s=40, c="red")

    for j1, j2 in BONES:

        if j1 in frame_positions and j2 in frame_positions:

            p1 = frame_positions[j1] * scale
            p2 = frame_positions[j2] * scale

            ax.plot(
                [p1[0],p2[0]],
                [p1[1],p2[1]],
                [p1[2],p2[2]],
                c="black",
                linewidth=2
            )


# ------------------------------------------------
# ANIMATION
# ------------------------------------------------

def animate_skeleton(positions):

    scale = 5

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_box_aspect([1,1,1])

    # Compute global bounds once
    xs, ys, zs = [], [], []

    for frame in positions[:200]:
        for p in frame.values():
            p = p * scale
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])

    margin = 20

    xmin, xmax = min(xs)-margin, max(xs)+margin
    ymin, ymax = min(ys)-margin, max(ys)+margin
    zmin, zmax = min(zs)-margin, max(zs)+margin

    for frame in positions[:200]:

        ax.cla()

        plot_skeleton(frame, ax, scale)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title("Motion Capture Skeleton")

        plt.pause(0.02)

    plt.show()


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def main():

    joints, hierarchy, frames = load_motion_data()

    frame_indices, knee_angles = extract_knee_angles(frames)

    plot_knee_angles(frame_indices, knee_angles)

    plot_phase_space(frame_indices, knee_angles)

    bones = parse_asf_geometry(ASF_PATH)

    positions = compute_joint_positions(frames, bones)

    animate_skeleton(positions)


if __name__ == "__main__":
    main()