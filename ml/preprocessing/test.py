from parse_asf import parse_asf
from parse_amc import parse_amc

asf_path = "data/raw/cmu_mocap/subject_01/skeleton.asf"
amc_path = "data/raw/cmu_mocap/subject_01/motions/01_01.amc"

joints, hierarchy = parse_asf(asf_path)
frames = parse_amc(amc_path)

print("Number of joints:", len(joints))
print("Number of frames:", len(frames))
print("First frame keys:", list(frames[0].keys())[:10])
print("Example ltibia angles:", frames[0].get("ltibia"))
print("Example rtibia angles:", frames[0].get("rtibia"))
print(sorted(frames[0].keys()))

import matplotlib.pyplot as plt

# Extract left knee (ltibia) angle over time
knee_angles = []
frame_indices = []

for i, frame in enumerate(frames):
    if "ltibia" in frame:
        # ltibia usually has 1 angle value
        knee_angles.append(frame["ltibia"][0])
        frame_indices.append(i)

print(f"Extracted {len(knee_angles)} knee angle samples")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(frame_indices, knee_angles)
plt.xlabel("Frame index")
plt.ylabel("Knee flexion angle (degrees)")
plt.title("Left Knee Angle (ltibia) Over Time")
plt.tight_layout()
plt.show()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

knee_angles = np.array(knee_angles)

angular_velocity = np.gradient(knee_angles)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    frame_indices,
    knee_angles,
    angular_velocity,
    linewidth=1
)

ax.set_xlabel("Frame")
ax.set_ylabel("Knee Angle (deg)")
ax.set_zlabel("Angular Velocity (deg/frame)")
ax.set_title("3D Knee Motion Phase Plot")

plt.tight_layout()
plt.show()