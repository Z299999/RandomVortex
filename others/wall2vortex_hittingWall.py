import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)
N = 3               # Number of samples per particle for the noise average

h1 = 0.2
Re = 0.0001 / nu
layer_thickness = np.sqrt(Re)
h2_0 = layer_thickness * 0.3  # finer layer thickness
h2 = h1 / (h1 // h2_0)

print("boundary layer thickness:", layer_thickness)
print("mesh grid:", h1, h2)

region_x = [-6, 6]
region_y = [-6, 0]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# (Optional) Background Velocity & Vorticity Functions
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return -np.cos(y)

# ---------------------------
# Nonuniform Grid Generation in D
# ---------------------------
def generate_nonuniform_grid_D(region_x=region_x, region_y=region_y, 
                               layer_thickness=layer_thickness, h1=h1, h2=h2):
    """
    Generates a nonuniform grid in D with a coarse grid over most of the domain 
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

# Create a grid for velocity field queries
query_grid, grid_coarse, grid_fine = generate_nonuniform_grid_D()

# ---------------------------
# Vortex Initialization: Determined Vortices
# ---------------------------
# Two vortices with prescribed positions and strengths
vortex_positions = np.array([[-2, -3], [2, -3]])
w0 = np.array([-2, 2])  # strengths corresponding to each vortex
num_vortices = len(w0)
print("* Number of vortices: ", num_vortices)

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
    Simulates vortex trajectories using the Euler-Maruyama scheme.
    
    Returns:
      traj: array of shape (num_steps+1, num_vortices, 2)
      uFuncs: list of functions, one per time step, each of which computes
              the local velocity (drift) at any position x.
      displacements: array of shape (num_steps, num_vortices, 2) storing the displacement 
                     (drift*dt + noise) for each vortex at each time step.
    """
    traj = np.zeros((num_steps + 1, num_vortices, 2))
    traj[0] = vortex_positions
    uFuncs = []
    displacements = np.zeros((num_steps, num_vortices, 2))
    
    for step in range(num_steps):
        current = traj[step]
        new = np.zeros_like(current)
        
        # Freeze the velocity field at the current time step into a function.
        def u_func(x, current=current):
            u_val = np.zeros(2)
            for j in range(num_vortices):
                pos_j = current[j]
                contrib1 = indicator_D(pos_j) * K_delta(pos_j, x, delta)
                contrib2 = indicator_D(reflect(pos_j)) * K_delta(reflect(pos_j), x, delta)
                u_val += (contrib1 - contrib2) * w0[j]
            return u_val
        
        uFuncs.append(u_func)
        
        for i in range(num_vortices):
            u_now = u_func(current[i])
            # Compute noise as the average of N independent samples
            dW_samples = np.sqrt(2 * nu * dt) * np.random.randn(N, 2)
            dW = dW_samples.mean(axis=0)
            disp = dt * u_now + dW
            displacements[step, i] = disp
            new[i] = current[i] + disp
        traj[step + 1] = new
    return traj, uFuncs, displacements

# ---------------------------
# Simplified Velocity Field Computation using uFuncs
# ---------------------------
def compute_velocity_field(u_func, query_points):
    """
    Computes the velocity field at given query points using a precomputed u_func.
    """
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        vel = u_func(query_points[p])
        U[p] = vel[0]
        V[p] = vel[1]
    return U, V

# ---------------------------
# Boat Simulation using Displacement Tensor (via uFuncs)
# ---------------------------
def generate_boat_grid():
    # We use the same grid as for the velocity field query.
    return generate_nonuniform_grid_D()

def simulate_boats(uFuncs):
    """
    Simulate boat trajectories on the nonuniform grid in D.
    Boat positions are updated using the precomputed uFuncs (i.e. the drift without noise).
    
    Returns:
      boat_positions: array of shape (num_steps+1, num_boats, 2)
      boat_displacements: array of shape (num_steps, num_boats, 2)
    """
    boat_grid, _, _ = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps + 1, num_boats, 2))
    boat_positions[0] = boat_grid
    boat_displacements = np.zeros((num_steps, num_boats, 2))
    
    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            vel = u_func(boat_positions[step, b])
            boat_displacements[step, b] = dt * vel  # boats follow the drift (no noise)
            boat_positions[step + 1, b] = boat_positions[step, b] + dt * vel
    return boat_positions, boat_displacements

# ---------------------------
# Main Simulation
# ---------------------------
print("Computing vortex trajectories......")
trajectories, uFuncs, displacement = simulate_vortex_trajectories()
print("Simulating vortex boats......")
boat_positions, boat_displacements = simulate_boats(uFuncs)

# Regenerate the query grid (for velocity field display)
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

# Initialize the velocity field quiver using uFuncs[0]
U, V = compute_velocity_field(uFuncs[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

# For boat simulation, we now draw boat positions as a scatter plot.
boat_scatter = ax.scatter(boat_positions[0, :, 0], boat_positions[0, :, 1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    
    # Update boat positions via scatter plot.
    boat_scatter.set_offsets(boat_positions[frame])
    
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps + 1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex7.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
