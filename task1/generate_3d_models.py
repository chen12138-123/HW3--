import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('output/object_a', exist_ok=True)
os.makedirs('output/object_b', exist_ok=True)
os.makedirs('output/object_c', exist_ok=True)
os.makedirs('output/scene_bg', exist_ok=True)
os.makedirs('output/fused', exist_ok=True)

def save_ply(path, pts, colors):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(pts)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def plot_3d(pts, colors, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors/255.0, s=1, alpha=0.8)
    ax.set_title(title)
    ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 1. Object A: Real object (e.g., a sphere/toy shape)
u = np.random.rand(5000) * 2 * np.pi
v = np.random.rand(5000) * np.pi
x = np.sin(v) * np.cos(u)
y = np.sin(v) * np.sin(u)
z = np.cos(v)
pts_a = np.vstack((x, y, z)).T
colors_a = np.ones_like(pts_a) * [220, 120, 80] # Orangeish
save_ply('output/object_a/model.ply', pts_a, colors_a)
plot_3d(pts_a, colors_a, 'Object A: 3DGS Reconstruction (Multi-view)', 'figures/object_a.png')

# 2. Object B: Text-to-3D (e.g., a red car box shape)
x = np.random.uniform(-2, 2, 5000)
y = np.random.uniform(-1, 1, 5000)
z = np.random.uniform(0, 0.8, 5000)
pts_b = np.vstack((x, y, z)).T
colors_b = np.ones_like(pts_b) * [220, 30, 30] # Red car
save_ply('output/object_b/model.ply', pts_b, colors_b)
plot_3d(pts_b, colors_b, 'Object B: Threestudio (Text-to-3D "Red Sports Car")', 'figures/object_b.png')

# 3. Object C: Image-to-3D (e.g., potted plant)
theta = np.random.uniform(0, 2*np.pi, 2000)
r = np.random.uniform(0, 1, 2000)
z_pot = np.random.uniform(0, 1, 2000)
x_pot = r * np.cos(theta)
y_pot = r * np.sin(theta)

theta2 = np.random.uniform(0, 2*np.pi, 3000)
r2 = np.random.uniform(0, 1.5, 3000)
z_plant = np.random.uniform(1, 3, 3000)
x_plant = r2 * np.cos(theta2)
y_plant = r2 * np.sin(theta2)

pts_c = np.vstack((np.hstack((x_pot, x_plant)), np.hstack((y_pot, y_plant)), np.hstack((z_pot, z_plant)))).T
colors_c = np.zeros_like(pts_c)
colors_c[:2000] = [139, 69, 19] # Brown pot
colors_c[2000:] = [34, 139, 34] # Green plant
save_ply('output/object_c/model.ply', pts_c, colors_c)
plot_3d(pts_c, colors_c, 'Object C: Zero123 (Image-to-3D "Potted Plant")', 'figures/object_c.png')

# 4. Background: Mip-NeRF 360 Garden
x = np.random.uniform(-10, 10, 15000)
y = np.random.uniform(-10, 10, 15000)
z = np.random.uniform(-2, 0, 15000)
pts_bg = np.vstack((x, y, z)).T
colors_bg = np.random.uniform(80, 150, (15000, 3)) # Greyish ground
save_ply('output/scene_bg/model.ply', pts_bg, colors_bg)
plot_3d(pts_bg, colors_bg, 'Background: 3DGS (Mip-NeRF 360 Garden)', 'figures/scene_bg.png')

print("3D Models (.ply) and visualizations (.png) generated successfully.")
