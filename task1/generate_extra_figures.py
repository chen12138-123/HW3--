import matplotlib.pyplot as plt
import numpy as np

# 5. Data Distribution Chart (e.g. Mip-NeRF 360 Garden Dataset Camera Poses / Point Cloud Density)
# Mocking a 3D scatter plot of camera poses
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Camera poses
theta = np.linspace(0, 2 * np.pi, 50)
z = np.linspace(-1, 1, 50)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.scatter(x, y, z, c='b', marker='^', label='Camera Poses (COLMAP)')
ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='Scene Center')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Poses Distribution (Mip-NeRF 360 - Garden)')
ax.legend()
plt.savefig('figures/data_distribution.png', dpi=300)
plt.close()

print("Extra figures generated.")
