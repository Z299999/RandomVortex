import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)

h1 = 1
Re = 0.0001 / nu
layer_thickness = np.sqrt(Re)
h2_0 = layer_thickness * 1  # finer layer thickness
h2 = h1 / (h1 // h2_0)

print("boundary layer thickness:", layer_thickness)
print("mesh grid:", h1, h2)

region_x = [-6, 6]
region_y = [-6, 0]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Vorticity and Grid
# ---------------------------
def vorticity(x, y):
    return -np.cos(y)

def generate_nonuniform_grid_D(region_x=region_x, region_y=region_y, 
                               layer_thickness=layer_thickness, h1=h1, h2=h2):
    """
    Generates a nonuniform grid in D with a coarse grid in most of the domain
    and a finer grid near the boundary (x2=0).
    """
    x1, x2 = region_x
    y1, y2 = region_y
    y3 = y2 - layer_thickness
    num_x_coarse = int((x2 - x1) / h1) + 1
    num_y_coarse = int((y3 - y1) / h1)
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y1, y3, num_y_coarse, endpoint=False)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse)
    grid_coarse = np.column_stack((xx_coarse.ravel(), yy_coarse.ravel()))
    
    num_x_fine = int((x2 - x1) / h2) + 1
    num_y_fine = int((y2 - y3) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y3, y2, num_y_fine)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
    grid_fine = np.column_stack((xx_fine.ravel(), yy_fine.ravel()))
    
    grid = np.concatenate((grid_coarse, grid_fine), axis=0)
    print(f"Number of points in coarse grid: {len(grid_coarse)}")
    print(f"Number of points in fine grid: {len(grid_fine)}")
    print(f"Total number of points in the final grid: {len(grid)}")
    return grid, grid_coarse, grid_fine

query_grid, grid_coarse, grid_fine = generate_nonuniform_grid_D()

# ---------------------------
# Vortex Initialization in D (x2 < 0)
# ---------------------------
vortex_positions = []
w0 = []

for pt in query_grid:
    x, y = pt
    w = vorticity(x, y)
    if np.abs(w) > 0.5:
        vortex_positions.append(pt)
        w0.append(w)

vortex_positions = np.array(vortex_positions)
w0 = np.array(w0)
print("* Number of nonzero vortices: ", w0.size)
num_vortices = w0.size

# ---------------------------
# Mollified Biot-Savart Kernel and Helpers
# ---------------------------
def K_delta(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r2     = (x1 - y1)**2 + (x2 - y2)**2
    r2_bar = (x1 - y1)**2 + (-x2 - y2)**2
    if r2 < 1e-10 or r2_bar < 1e-10:
        return np.zeros(2)
    k1 = 0.5 / np.pi * ((y2 - x2)/r2 - (y2 + x2)/r2_bar)
    k2 = 0.5 / np.pi * ((y1 - x1)/r2_bar - (y1 - x1)/r2)
    factor = 1 - np.exp(- (r2 / delta)**2)
    return np.array([k1, k2]) * factor

def indicator_D(point):
    return 1 if point[1] < 0 else 0

def reflect(point):
    return np.array([point[0], -point[1]])

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates the trajectories of vortices using the Euler-Maruyama scheme.
    Returns an array of shape (num_steps+1, num_vortices, 2).
    """
    traj = np.zeros((num_steps + 1, num_vortices, 2))
    traj[0] = vortex_positions
    for step in range(num_steps):
        current = traj[step]  # shape: (num_vortices, 2)
        new = np.zeros_like(current)
        for i in range(num_vortices):
            drift = np.zeros(2)
            for j in range(num_vortices):
                pos_j = current[j]
                contrib1 = indicator_D(pos_j) * K_delta(pos_j, current[i], delta)
                contrib2 = indicator_D(reflect(pos_j)) * K_delta(reflect(pos_j), current[i], delta)
                drift += (contrib1 - contrib2) * w0[j]
            dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
            new[i] = current[i] + dt * drift + dW
        traj[step + 1] = new
    return traj



# ---------------------------
# Boat Simulation on a Nonuniform Grid in D
# ---------------------------
def generate_boat_grid():
    # For now, we use the same grid as for velocity queries.
    return generate_nonuniform_grid_D()

def simulate_boats(vortex_traj):
    """
    Simulate boat trajectories on the nonuniform grid in D.
    Boats move according to the local velocity computed from the vortex simulation.
    Returns:
      boat_positions: shape (num_steps+1, num_boats, 2)
      boat_streams: shape (num_steps+1, num_boats, 2) -- line segment vectors.
    """
    boat_grid, _, _ = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps + 1, num_boats, 2))
    boat_streams = np.zeros((num_steps + 1, num_boats, 2))
    boat_positions[0] = boat_grid
    arrow_scale = 0.05  # scaling factor for visualization
    for step in range(num_steps + 1):
        current_vortex = vortex_traj[step]  # shape: (num_vortices, 2)
        for b in range(num_boats):
            pos = boat_positions[step, b]
            vel = np.zeros(2)
            for i in range(num_vortices):
                pos_i = current_vortex[i]
                contrib1 = indicator_D(pos_i) * K_delta(pos_i, pos, delta)
                contrib2 = indicator_D(reflect(pos_i)) * K_delta(reflect(pos_i), pos, delta)
                vel += (contrib1 - contrib2) * w0[i]
            boat_streams[step, b] = arrow_scale * vel
            if step < num_steps:
                boat_positions[step + 1, b] = pos + dt * vel
    return boat_positions, boat_streams

# ---------------------------
# Velocity Field Computation
# ---------------------------
def compute_velocity_field(vortex_positions_t, query_points):
    """
    Computes the velocity field at the given query points using the method of images.
    vortex_positions_t: shape (num_vortices, 2) at time t.
    query_points: array of shape (P,2) in D.
    Returns U, V (each of length P).
    """
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        pos = query_points[p]
        vel = np.zeros(2)
        for i in range(num_vortices):
            pos_i = vortex_positions_t[i]
            contrib1 = indicator_D(pos_i) * K_delta(pos_i, pos, delta)
            contrib2 = indicator_D(reflect(pos_i)) * K_delta(reflect(pos_i), pos, delta)
            vel += (contrib1 - contrib2) * w0[i]
        U[p], V[p] = vel[0], vel[1]
    return U, V
    

# ---------------------------
# Main Simulation
# ---------------------------
print("Computing vortices trajectories......")
trajectories = simulate_vortex_trajectories()  # shape: (num_steps+1, num_vortices, 2)
print("Simulating vortex boats......")
boat_positions, boat_streams = simulate_boats(trajectories)

# Create nonuniform grid for velocity field query in D
query_grid, grid_coarse, grid_fine = generate_nonuniform_grid_D()

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Initialize velocity field quiver
U, V = compute_velocity_field(trajectories[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

# Initialize boat segments: endpoints from center ± 0.5 * stream vector.
boats_init = boat_positions[0]
streams_init = boat_streams[0]
left_init = boats_init - 0.5 * streams_init
right_init = boats_init + 0.5 * streams_init
initial_segments = np.stack([left_init, right_init], axis=1)

# LineCollection for boat segments (blue lines)
boat_lines = LineCollection(initial_segments, colors='blue', linewidths=2)
ax.add_collection(boat_lines)

# Scatter plot for endpoints (to mimic tapered, sharper ends)
all_endpoints_init = np.concatenate([left_init, right_init], axis=0)
scat = ax.scatter(all_endpoints_init[:, 0], all_endpoints_init[:, 1],
                  s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    current_vortex = trajectories[frame]  # shape: (num_vortices, 2)
    U, V = compute_velocity_field(current_vortex, query_grid)
    vel_quiver.set_UVC(U, V)
    
    current_boats = boat_positions[frame]
    current_streams = boat_streams[frame]
    left_endpoints = current_boats - 0.5 * current_streams
    right_endpoints = current_boats + 0.5 * current_streams
    segments = np.stack([left_endpoints, right_endpoints], axis=1)
    boat_lines.set_segments(segments)
    
    all_endpoints = np.concatenate([left_endpoints, right_endpoints], axis=0)
    scat.set_offsets(all_endpoints)
    
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_lines, scat

anim = FuncAnimation(fig, update, frames=num_steps + 1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex7.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
