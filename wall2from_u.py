import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001          # Viscosity
T = 10.0            # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T/dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)
N = 10              # Number of sample paths per vortex
num_vortices = 10   # Total number of vortices

h1 = 0.5
Re = 0.0001 / nu
layer_thickness = np.sqrt(Re)
h2_0 = layer_thickness * 0.4  # finer layer thickness
h2 = h1 / (h1 // h2_0)

print("boundary layer thickness:", layer_thickness)
print("mesh grid:", h1, h2)

region_x = [-6, 6]
region_y = [-6, 0]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Vorticity Initialization in D (x2 < 0)
# ---------------------------
def vorticity(x, y):
    return -np.cos(y)

def generate_nonuniform_grid_D(region_x=region_x, region_y=region_y, 
                               layer_thickness=layer_thickness, h1=h1, h2=h2):
    """
    Generates a nonuniform grid in D: a coarse grid in most of D
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

# Compute and store initial vorticity (only nonzero ones)
vortex_positions = []
w0 = []
for pt in query_grid:
    x, y = pt
    w = vorticity(x, y)
    if w != 0:
        vortex_positions.append(pt)
        w0.append(w)
vortex_positions = np.array(vortex_positions)
w0 = np.array(w0)
print("* Number of nonzero vortices:", w0.size)

# ---------------------------
# Known Initial Velocity Field Function
# ---------------------------
def initial_velocity(point):
    """
    Returns the known velocity at a given point.
    Here we use the initial velocity field:
      u(x,y) = -sin(y),  v(x,y) = 0.
    """
    x, y = point
    return np.array([-np.sin(y), 0])

# ---------------------------
# Mollified Biot-Savart Kernel (kept for future use if needed)
# ---------------------------
def K_delta(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r2 = (x1 - y1)**2 + (x2 - y2)**2
    r2_bar = (x1 - y1)**2 + (-x2 - y2)**2
    if r2 < 1e-10 or r2_bar < 1e-10:
        return np.zeros(2)
    k1 = 0.5 / np.pi * ((y2 - x2)/r2 - (y2 + x2)/r2_bar)
    k2 = 0.5 / np.pi * ((y1 - x1)/r2_bar - (y1 - x1)/r2)
    factor = 1 - np.exp(- (r2/delta)**2)
    return np.array([k1, k2]) * factor

def indicator_D(point):
    return 1 if point[1] < 0 else 0

def reflect(point):
    return np.array([point[0], -point[1]])

# ---------------------------
# Vortex Trajectory Simulation using Known Velocity Field
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates the trajectories of vortices (each with N sample paths)
    using the Euler–Maruyama scheme with a known velocity field.
    The SDE is:
      dX_t = u(X_t,t) dt + sqrt(2*nu) dB_t,   X_0 = ξ,
    with u given by initial_velocity.
    
    We also store the velocity used at each step in the array “velocities”.
    
    Returns:
      traj: array of shape (num_steps+1, num_vortices, N, 2) (vortex trajectories)
      velocities: array of shape (num_steps+1, num_vortices, N, 2) (velocity field at vortex positions)
    """
    traj = np.zeros((num_steps+1, num_vortices, N, 2))
    velocities = np.zeros((num_steps+1, num_vortices, N, 2))
    # Initialize positions for each vortex (using the first num_vortices from vortex_positions)
    for i in range(num_vortices):
        for rho in range(N):
            traj[0, i, rho, :] = vortex_positions[i]
            velocities[0, i, rho, :] = initial_velocity(vortex_positions[i])
    
    for step in range(num_steps):
        current = traj[step]  # shape: (num_vortices, N, 2)
        new = np.zeros_like(current)
        new_vel = np.zeros_like(current)
        for i in range(num_vortices):
            for rho in range(N):
                pos = current[i, rho]
                # Use the known velocity field (here time independent) for drift
                vel = initial_velocity(pos)
                new_vel[i, rho] = vel
                # Update using Euler–Maruyama scheme
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new[i, rho] = pos + dt * vel + dW
        traj[step+1] = new
        velocities[step+1] = new_vel  # store the velocity computed at this time step
    return traj, velocities

print("Simulating vortex trajectories...")
trajectories, vortex_velocities = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation on a Nonuniform Grid in D using Known Velocity Field
# ---------------------------
def simulate_boats():
    """
    Simulates boat trajectories on the nonuniform grid in D.
    Boats follow the known velocity field (i.e. they are updated via
      dX_t = u(X_t,t) dt
    with u given by initial_velocity).
    
    Returns:
      boat_positions: array of shape (num_steps+1, num_boats, 2)
      boat_streams: array of shape (num_steps+1, num_boats, 2) representing scaled velocity segments.
    """
    boat_grid, _, _ = generate_nonuniform_grid_D()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_streams = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    arrow_scale = 0.05  # scaling factor for visualization
    for step in range(num_steps+1):
        for b in range(num_boats):
            pos = boat_positions[step, b]
            vel = initial_velocity(pos)
            boat_streams[step, b] = arrow_scale * vel
            if step < num_steps:
                boat_positions[step+1, b] = pos + dt * vel
    return boat_positions, boat_streams

print("Simulating boat trajectories...")
boat_positions, boat_streams = simulate_boats()

# ---------------------------
# Precompute velocity field at query grid for animation
# ---------------------------
def compute_velocity_field_on_grid(points):
    """
    Computes the velocity field at the given query grid points using initial_velocity.
    (For this simulation the velocity field is time-independent.)
    """
    U = np.zeros(points.shape[0])
    V = np.zeros(points.shape[0])
    for i, pt in enumerate(points):
        vel = initial_velocity(pt)
        U[i] = vel[0]
        V[i] = vel[1]
    return U, V

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Initialize quiver plot with the velocity field on the query grid
U, V = compute_velocity_field_on_grid(query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V, color='black', alpha=0.9,
                       pivot='mid', scale=None, angles='xy', scale_units='xy')

# Boat visualization: using LineCollection to draw thin blue segments
boats_init = boat_positions[0]
streams_init = boat_streams[0]
left_init = boats_init - 0.5 * streams_init
right_init = boats_init + 0.5 * streams_init
initial_segments = np.stack([left_init, right_init], axis=1)
boat_lines = LineCollection(initial_segments, colors='blue', linewidths=2)
ax.add_collection(boat_lines)
all_endpoints_init = np.concatenate([left_init, right_init], axis=0)
scat = ax.scatter(all_endpoints_init[:, 0], all_endpoints_init[:, 1],
                  s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    # For this simulation, the velocity field remains the same (time-independent)
    U, V = compute_velocity_field_on_grid(query_grid)
    vel_quiver.set_UVC(U, V)
    
    # Update boat segments based on boat_positions and boat_streams
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

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

# Save the animation to a subfolder "animation" as "vortex7.mp4"
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex7.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
