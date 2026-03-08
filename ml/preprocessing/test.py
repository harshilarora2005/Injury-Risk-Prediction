import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from parse_asf import parse_asf
from parse_amc import parse_amc
from compute_joint_positions import compute_joint_positions


# ── Paths ─────────────────────────────────────────────────────────────────────
ASF_PATH = 'data/raw/cmu_mocap/subject_01/skeleton.asf'
AMC_PATH = 'data/raw/cmu_mocap/subject_01/motions/01_01.amc'

# ── Bone pairs for drawing lines ──────────────────────────────────────────────
BONE_PAIRS = [
    ('root', 'lowerback'),
    ('lowerback', 'upperback'),
    ('upperback', 'thorax'),
    ('thorax', 'lowerneck'),
    ('lowerneck', 'upperneck'),
    ('upperneck', 'head'),

    ('thorax', 'lclavicle'),
    ('lclavicle', 'lhumerus'),
    ('lhumerus', 'lradius'),
    ('lradius', 'lwrist'),
    ('lwrist', 'lhand'),

    ('thorax', 'rclavicle'),
    ('rclavicle', 'rhumerus'),
    ('rhumerus', 'rradius'),
    ('rradius', 'rwrist'),
    ('rwrist', 'rhand'),

    ('root', 'lfemur'),
    ('lfemur', 'ltibia'),
    ('ltibia', 'lfoot'),
    ('lfoot', 'ltoes'),

    ('root', 'rfemur'),
    ('rfemur', 'rtibia'),
    ('rtibia', 'rfoot'),
    ('rfoot', 'rtoes'),
]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_motion_data():
    print('Parsing ASF ...')
    joints = parse_asf(ASF_PATH)
    print(f'  {len(joints)} joints loaded')

    print('Parsing AMC ...')
    motions = parse_amc(AMC_PATH)
    print(f'  {len(motions)} frames loaded')

    print('Computing joint positions ...')
    positions = compute_joint_positions(motions, joints)
    print('  Done.')

    return positions


# ── Knee angle signal ─────────────────────────────────────────────────────────
def extract_knee_angles(motions):
    frame_indices, knee_angles = [], []
    for i, motion in enumerate(motions):
        if 'ltibia' in motion:
            knee_angles.append(motion['ltibia'][0])
            frame_indices.append(i)
    return frame_indices, np.array(knee_angles)


def plot_knee_angles(frame_indices, knee_angles):
    plt.figure(figsize=(10, 4))
    plt.plot(frame_indices, knee_angles)
    plt.xlabel('Frame')
    plt.ylabel('Knee Angle (deg)')
    plt.title('Left Knee Angle Over Time')
    plt.tight_layout()
    plt.show()


def plot_phase_space(frame_indices, knee_angles):
    velocity = np.gradient(knee_angles)
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(frame_indices, knee_angles, velocity)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle')
    ax.set_zlabel('Velocity')
    ax.set_title('Knee Motion Phase Space')
    plt.show()


# ── Animation ─────────────────────────────────────────────────────────────────
def compute_bounds(positions, margin=15):
    xs, ys, zs = [], [], []
    for frame in positions:
        for p in frame.values():
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
    return (min(xs)-margin, max(xs)+margin,
            min(ys)-margin, max(ys)+margin,
            min(zs)-margin, max(zs)+margin)


def draw_skeleton(ax, frame_pos, bounds, frame_idx, total):
    ax.cla()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'CMU MoCap  —  frame {frame_idx + 1} / {total}')
    ax.set_box_aspect([1, 1, 1])

    pts = np.array(list(frame_pos.values()))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18, c='red', zorder=5)

    for j1, j2 in BONE_PAIRS:
        if j1 in frame_pos and j2 in frame_pos:
            p1, p2 = frame_pos[j1], frame_pos[j2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    c='black', linewidth=1.8)


def animate_skeleton(positions):
    MAX_FRAMES = 300
    positions  = positions[:MAX_FRAMES]
    total      = len(positions)
    bounds     = compute_bounds(positions)

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')

    def update(i):
        frame_idx = i % total   # wrap → continuous, no reset jump
        draw_skeleton(ax, positions[frame_idx], bounds, frame_idx, total)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=itertools.count(),   # infinite — never resets
        interval=33,                # ~30 fps
        repeat=False,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()

    # Uncomment to save:
    # ani.save('skeleton.gif', writer='pillow', fps=30)
    # ani.save('skeleton.mp4', writer='ffmpeg', fps=30)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Parsing AMC for knee angles ...')
    motions = parse_amc(AMC_PATH)
    frame_indices, knee_angles = extract_knee_angles(motions)
    plot_knee_angles(frame_indices, knee_angles)
    plot_phase_space(frame_indices, knee_angles)

    positions = load_motion_data()
    animate_skeleton(positions)


if __name__ == '__main__':
    main()