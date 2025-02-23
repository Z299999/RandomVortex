import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from K_thinPlate import K_delta  # do not modify this external kernel

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)
N = 5               # Number of sample paths per vortex

# Grid parameters for boat simulation (nonuniform grid)
h1 = 1
Re = 0.0001 / nu
layer_thickness = np.sqrt(Re)
h2_0 = layer_thickness * 0.6  # finer layer thickness
h2 = h1 / (h1 // h2_0)

print("boundary layer thickness:", layer_thickness)
print("mesh grid:", h1, h2)

region_x = [-6, 6]
region_y = [-3, 3]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Two Vortices Initialization
# ---------------------------
# Vortex 1: strength -2, initial position [-0.5, 0.5]
# Vortex 2: strength  2, initial position [-0.5, -0.5]
vortex_positions = np.array([[-0.5, 0.5],
                             [-0.5, -0.5]])
w0 = np.array([-2, 2])
num_vortices = 2

# ---------------------------
# Generate Nonuniform Grid for Boat Simulation
# ---------------------------
def generate_nonuniform_grid_D(region_x=region_x, region_y=region_y, 
                               layer_thickness=layer_thickness, h1=h1, h2=h2):
    x_min, x_max = region_x
    y_min, y_max = region_y

    # Coarse grid covering the entire simulation domain with spacing h1
    x_coarse = np.arange(x_min, x_max + h1, h1)
    y_coarse = np.arange(y_min, y_max + h1, h1)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse)
    grid_coarse = np.column_stack((xx_coarse.ravel(), yy_coarse.ravel()))
    
    # Fine grid: impose dense mesh near the thin plate boundary
    # Thin plate is defined as {(x1,x2): x1 > 0, x2 = 0}, so we refine only for x > 0.
    # Since the fluid lies below the plate (as per indicator_D), we refine for y in [-layer_thickness, 0].
    x_fine = np.arange(0, x_max + h2, h2)
    y_fine = np.arange(-layer_thickness, 0 + h2, h2)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
    grid_fine = np.column_stack((xx_fine.ravel(), yy_fine.ravel()))
    
    # Combine the coarse and fine grids and remove duplicate points
    grid = np.vstack((grid_coarse, grid_fine))
    grid = np.unique(grid, axis=0)
    
    print(f"Number of points in coarse grid: {len(grid_coarse)}")
    print(f"Number of points in fine grid: {len(grid_fine)}")
    print(f"Total number of points in the final grid: {len(grid)}")
    return grid, grid_coarse, grid_fine

# ---------------------------
# Mollified Biot-Savart Kernel and Helpers
# ---------------------------
def indicator_D(point):
    return 1 if point[1] < 0 else 0

def reflect(point):
    return np.array([point[0], -point[1]])

# ---------------------------
# Vortex Trajectory Simulation using Euler–Maruyama with N sample paths
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates vortex trajectories using the Euler–Maruyama scheme.
    Uses N sample paths for each vortex.
    """
    traj = np.zeros((num_steps + 1, num_vortices, N, 2))
    # Set initial conditions for each sample path of each vortex
    for i in range(num_vortices):
        for rho in range(N):
            traj[0, i, rho, :] = vortex_positions[i]
    uFuncs = []
    displacements = np.zeros((num_steps, num_vortices, N, 2))
    
    for step in range(num_steps):
        current = traj[step]  # shape: (num_vortices, N, 2)
        new = np.zeros_like(current)
        
        # Freeze the current velocity field into a function u_func
        def u_func(x, current=current):
            u_val = np.zeros(2)
            for j in range(num_vortices):
                tmp = np.zeros(2)
                for sigma in range(N):
                    pos_j = current[j, sigma]
                    contrib1 = K_delta(pos_j, x, delta)
                    contrib2 = K_delta(reflect(pos_j), x, delta)
                    tmp += (contrib1 - contrib2)
                u_val += (tmp / N) * w0[j]
            return u_val
        
        uFuncs.append(u_func)
        
        for i in range(num_vortices):
            for rho in range(N):
                u_now = u_func(current[i, rho])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_now + dW
                displacements[step, i, rho] = disp
                new[i, rho] = current[i, rho] + disp
        traj[step + 1] = new
    return traj, uFuncs, displacements

# ---------------------------
# Compute Velocity Field using uFuncs (for visualization)
# ---------------------------
def compute_velocity_field(u_func, query_points):
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        vel = u_func(query_points[p])
        U[p] = vel[0]
        V[p] = vel[1]
    return U, V

# ---------------------------
# Boat Simulation (using the drift computed from uFuncs)
# ---------------------------
def generate_boat_grid():
    return generate_nonuniform_grid_D()

def simulate_boats(uFuncs):
    boat_grid, _, _ = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps + 1, num_boats, 2))
    boat_positions[0] = boat_grid
    boat_displacements = np.zeros((num_steps, num_boats, 2))
    
    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            vel = u_func(boat_positions[step, b])
            boat_displacements[step, b] = dt * vel
            boat_positions[step + 1, b] = boat_positions[step, b] + dt * vel
    return boat_positions, boat_displacements

# ---------------------------
# Main Simulation
# ---------------------------
print("Computing vortex trajectories......")
trajectories, uFuncs, displacement = simulate_vortex_trajectories()
print("Simulating boat trajectories......")
boat_positions, boat_displacements = simulate_boats(uFuncs)

# Regenerate the query grid for visualization
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

# Plot boat positions as a scatter plot
boat_scatter = ax.scatter(boat_positions[0, :, 0], boat_positions[0, :, 1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    
    # Update boat positions
    boat_scatter.set_offsets(boat_positions[frame])
    
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps + 1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "thinPlate.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
