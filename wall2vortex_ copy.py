import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001          # Viscosity
T = 40.0            # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T/dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)
N = 10              # Number of sample paths per vortex
num_vortices = 10    # Total number of vortices

h1 = 0.3
Re = 0.0001/nu
layer_thickness = np.sqrt(Re)
h2_0 = layer_thickness * 0.3  # finer layer thickness
h2 = h1 / (h1//h2_0)

print("boundary layer thickness:", layer_thickness)
print("mesh grid:", h1, h2)

region_x = [-6, 6]
region_y = [-6, 0]
window_x = [x1 + x2 for x1, x2 in zip(region_x, [-0, 0])]
window_y = [y1 + y2 for y1, y2 in zip(region_y, [-0, 0])]

np.random.seed(42)

# ---------------------------
# Nonuniform Grid for Query Points (Finer near boundary x2=0)
# ---------------------------
def generate_nonuniform_grid_D(region_x=region_x, region_y=region_y, 
                               layer_thickness=layer_thickness, h1=h1, h2=h2):
    """
    Generates a nonuniform grid in D, using a coarse grid in most of D
    and a finer grid near the boundary (x2=0).
    Returns:
      grid: full set of query points (P x 2)
      grid_coarse: points in the coarse region
      grid_fine: points in the fine region
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

# ---------------------------
# Vortex Initialization in D (x2 < 0)
# ---------------------------
# Generate vortex positions with x1 in [-1,1] and x2 in [-1,0]
vortex_positions = np.zeros((num_vortices, 2))
vortex_positions[:, 0] = np.random.uniform(region_x[0], region_x[1], num_vortices)
vortex_positions[:, 1] = np.random.uniform(region_y[0], region_y[1], num_vortices)

# Vortex strengths (vorticity)
w0 = np.random.uniform(-2, 2, num_vortices)
# w0 = np.array([-1, 1])

# ---------------------------
# Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r2     = (x1 - y1)**2 + (x2 - y2)**2
    r2_bar = (x1 - y1)**2 + (-x2 - y2)**2
    if r2 < 1e-10 or r2_bar < 1e-10:
        return np.zeros(2)
    k1 = 0.5 / np.pi * ((y2 - x2)/r2     - (y2 + x2)/r2_bar)
    k2 = 0.5 / np.pi * ((y1 - x1)/r2_bar - (y1 - x1)/r2    )
    factor = 1 - np.exp(- (r2/delta)**2)
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
    Simulates the trajectories of vortices (each with N sample paths)
    using the Euler-Maruyama scheme.
    Returns an array of shape (num_steps+1, num_vortices, N, 2).
    """
    traj = np.zeros((num_steps+1, num_vortices, N, 2))
    # Initialization
    for i in range(num_vortices):
        for rho in range(N):
            traj[0, i, rho, :] = vortex_positions[i]
    for step in range(num_steps):
        current = traj[step]  # shape: (num_vortices, N, 2)
        new = np.zeros_like(current)
        for i in range(num_vortices):
            for rho in range(N):
                drift = np.zeros(2)
                # Sum over contributions from all vortices (and their images)
                for j in range(num_vortices):
                    tmp = np.zeros(2)
                    for sigma in range(N):
                        pos_j = current[j, sigma]
                        contrib1 = indicator_D(pos_j         ) * K_delta(pos_j,          current[i, rho], delta)
                        contrib2 = indicator_D(reflect(pos_j)) * K_delta(reflect(pos_j), current[i, rho], delta)
                        tmp += (contrib1 - contrib2)
                    drift += (tmp / N) * w0[j]
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new[i, rho] = current[i, rho] + dt * drift + dW
        traj[step+1] = new
    return traj



# ---------------------------
# Boat Simulation on a Nonuniform Grid in D
# ---------------------------
def generate_boat_grid():
    # For now, we use the same grid as the velocity query grid.
    return generate_nonuniform_grid_D()

def simulate_boats(vortex_traj):
    """
    Simulate boat trajectories on the nonuniform grid in D.
    Boats move according to the local velocity computed from the vortex simulation.
    Instead of red arrows, we now create thin blue line segments whose total length is
    proportional to the local velocity. For a boat at position x with scaled velocity v,
    we will later draw a line segment from x - 0.5*v to x + 0.5*v (centered at x).
    
    Returns:
      boat_positions: shape (num_steps+1, num_boats, 2)
      boat_streams: shape (num_steps+1, num_boats, 2) -- total segment vector.
    """
    boat_grid, _, _ = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_streams = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    arrow_scale = 0.05  # scaling factor for visualization (adjust to change segment lengths)
    for step in range(num_steps+1):
        current_vortex = vortex_traj[step]  # shape: (num_vortices, N, 2)
        for b in range(num_boats):
            pos = boat_positions[step, b]
            vel = np.zeros(2)
            for i in range(num_vortices):
                tmp = np.zeros(2)
                for rho in range(N):
                    pos_i = current_vortex[i, rho]
                    contrib1 = indicator_D(pos_i         ) * K_delta(pos_i,          pos, delta)
                    contrib2 = indicator_D(reflect(pos_i)) * K_delta(reflect(pos_i), pos, delta)
                    tmp += (contrib1 - contrib2)
                vel += (tmp / N) * w0[i]
            boat_streams[step, b] = arrow_scale * vel
            if step < num_steps:
                boat_positions[step+1, b] = pos + dt * vel
    return boat_positions, boat_streams

# ---------------------------
# Velocity Field Computation
# ---------------------------
def compute_velocity_field(vortex_positions_t, query_points):
    """
    Computes the velocity field at the given query points using the method of images.
    vortex_positions_t: shape (num_vortices, N, 2) at time t.
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
            tmp = np.zeros(2)
            for rho in range(N):
                pos_i = vortex_positions_t[i, rho]
                contrib1 = indicator_D(pos_i         ) * K_delta(pos_i,          pos, delta)
                contrib2 = indicator_D(reflect(pos_i)) * K_delta(reflect(pos_i), pos, delta)
                tmp += (contrib1 - contrib2)
            vel += (tmp / N) * w0[i]
        U[p], V[p] = vel[0], vel[1]
    return U, V

# ---------------------------
# Main Simulation
# ---------------------------
print("Computing vortices trajectories......")
trajectories = simulate_vortex_trajectories()  # shape: (num_steps+1, num_vortices, N, 2)
print("Simulating vortex boats......")
boat_positions, boat_streams = simulate_boats(trajectories)

# Create nonuniform grid for velocity field query (in D)
query_grid, grid_coarse, grid_fine = generate_nonuniform_grid_D()

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])  # Show a bit above the boundary
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Initialize velocity field quiver (50% transparent arrows)
U, V = compute_velocity_field(trajectories[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V, color='black', alpha=0.9,
                       pivot='mid', scale=None, angles='xy', scale_units='xy')

# ---------------------------
# Boat visualization using LineCollection and scatter for tapered endpoints
# ---------------------------
# Compute initial boat segments:
# For each boat, compute endpoints: [x - 0.5*v, x + 0.5*v]
boats_init = boat_positions[0]
streams_init = boat_streams[0]
left_init = boats_init - 0.5 * streams_init
right_init = boats_init + 0.5 * streams_init
initial_segments = np.stack([left_init, right_init], axis=1)

# Create a LineCollection for the segments (blue lines)
boat_lines = LineCollection(initial_segments, colors='blue', linewidths=2)
ax.add_collection(boat_lines)

# Create a scatter plot for the endpoints (to mimic tapered, sharper ends)
all_endpoints_init = np.concatenate([left_init, right_init], axis=0)
scat = ax.scatter(all_endpoints_init[:, 0], all_endpoints_init[:, 1],
                  s=10, color='blue', zorder=3)

# ---------------------------
# Animation update function
# ---------------------------
def update(frame):
    t_current = frame * dt
    current_vortex = trajectories[frame]  # shape: (num_vortices, N, 2)
    U, V = compute_velocity_field(current_vortex, query_grid)
    # Update the velocity field quiver
    vel_quiver.set_UVC(U, V)
    
    # Update boat segments: compute endpoints for each boat.
    current_boats = boat_positions[frame]
    current_streams = boat_streams[frame]
    left_endpoints = current_boats - 0.5 * current_streams
    right_endpoints = current_boats + 0.5 * current_streams
    segments = np.stack([left_endpoints, right_endpoints], axis=1)
    boat_lines.set_segments(segments)
    
    # Update the scatter plot for endpoints (combine left and right endpoints)
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