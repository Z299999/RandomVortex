import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time  # Add import for time module

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01          # Viscosity
T = 20             # Final time (seconds)
dt = 0.1           # Time step
num_steps = int(T / dt)
delta = 0.1        # Mollification parameter

# Domain parameters:
H = 6              # Adjustable domain height
region_x = [-6, 6]
region_y = [0, H]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

# Mesh parameters:
h0 = 1             # Coarse mesh grid spacing (approximate)
h1 = 0.5           # Fine mesh grid spacing along x
h2 = 0.3           # Fine mesh grid spacing along y
layer_thickness = 0.8  # Boundary layer thickness (for both lower and upper walls)
hh = layer_thickness   # alias

np.random.seed(42)

# ---------------------------
# Vorticity, Velocity and Related Functions
# ---------------------------
def velocity(x, y, H=H):
    # Velocity vanishes at y=0 and y=H.
    return np.array([-0.5 * np.sin(np.pi * y / H), 0])

def vorticity(x, y, H=H):
    # vorticity computed as the derivative of u: 0.5*(pi/H)*cos(pi*y/H)
    return 0.5 * (np.pi / H) * np.cos(np.pi * y / H)

# ---------------------------
# Grid Generation: Lower Fine, Upper Fine (by reflection), and Coarse Interior
# ---------------------------
def generate_nonuniform_grid_D():
    """
    Generates a symmetric nonuniform grid in the band [0,H] with three parts:
      1. Lower fine grid: y in [0, hh) using step size h2 (endpoint excluded).
      2. Upper fine grid: obtained by reflecting the lower fine grid via y -> H - y.
      3. Coarse grid: y in [hh, H - hh] with endpoint=True.
    
    In the x-direction, the fine grids use spacing h1 and the coarse grid uses spacing h0.
    
    Returns:
      A dictionary with keys 'coarse', 'fine_lower', and 'fine_upper'. Each value is a tuple (grid, A),
      where grid is an array of shape (n_x, n_y, 2) and A is the corresponding area array.
    """
    x1, x2 = region_x
    y1, y2 = region_y  # here, y2 is H

    # ---- Lower Fine Grid ----
    # Generate y-values from 0 (inclusive) to hh (exclusive) with step size h2.
    y_vals_lower = np.arange(0, hh, h2)
    num_x_fine = int((x2 - x1) / h1) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    xx_fine_lower, yy_fine_lower = np.meshgrid(x_fine, y_vals_lower, indexing='ij')
    grid_fine_lower = np.stack((xx_fine_lower, yy_fine_lower), axis=-1)
    A_fine_lower = h1 * h2 * np.ones((num_x_fine, len(y_vals_lower)))
    
    # ---- Upper Fine Grid ----
    # Reflect lower fine grid: y -> H - y.
    y_vals_upper = H - y_vals_lower
    y_vals_upper = np.sort(y_vals_upper)  # Ensure ascending order.
    xx_fine_upper, yy_fine_upper = np.meshgrid(x_fine, y_vals_upper, indexing='ij')
    grid_fine_upper = np.stack((xx_fine_upper, yy_fine_upper), axis=-1)
    A_fine_upper = h1 * h2 * np.ones((num_x_fine, len(y_vals_upper)))
    
    # ---- Coarse Grid ----
    # Generate y in [hh, H - hh] with endpoints included.
    num_y_coarse = int(round((H - 2 * hh) / h0)) + 1
    num_x_coarse = int((x2 - x1) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(hh, H - hh, num_y_coarse, endpoint=True)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))
    
    print(f"Lower Fine grid shape: {grid_fine_lower.shape}, number of points: {grid_fine_lower.size//2}")
    print(f"Upper Fine grid shape: {grid_fine_upper.shape}, number of points: {grid_fine_upper.size//2}")
    print(f"Coarse grid shape: {grid_coarse.shape}, number of points: {grid_coarse.size//2}")
    
    return {
        'coarse': (grid_coarse, A_coarse),
        'fine_lower': (grid_fine_lower, A_fine_lower),
        'fine_upper': (grid_fine_upper, A_fine_upper)
    }

# Generate the grids
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_fine_lower, A_fine_lower = grids['fine_lower']
grid_fine_upper, A_fine_upper = grids['fine_upper']

# ---------------------------
# Debug Plot: Mesh Grids
# ---------------------------
def plot_mesh_grids():
    plt.figure(figsize=(8, 6))
    # Plot coarse grid points in blue squares.
    plt.scatter(grid_coarse[:,:,0], grid_coarse[:,:,1], c='blue', marker='s', label='Coarse')
    # Plot lower fine grid points in red circles.
    plt.scatter(grid_fine_lower[:,:,0], grid_fine_lower[:,:,1], c='red', marker='o', label='Fine Lower')
    # Plot upper fine grid points in red circles.
    plt.scatter(grid_fine_upper[:,:,0], grid_fine_upper[:,:,1], c='red', marker='o', label='Fine Upper')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Symmetric Mesh Grids in [0,H]")
    plt.legend()
    plt.xlim(window_x)
    plt.ylim(window_y)
    plt.grid(True)
    plt.show()

plot_mesh_grids()

# ---------------------------
# Vortex Initialization
# ---------------------------
def initialize_vortices(grid):
    n_x, n_y, _ = grid.shape
    w = np.zeros((n_x, n_y))
    u = np.zeros((n_x, n_y, 2))
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y, H)
            u[i, j] = velocity(x, y, H)
    return w, u

w0_coarse, u0_coarse = initialize_vortices(grid_coarse)
w0_fine_lower, u0_fine_lower = initialize_vortices(grid_fine_lower)
w0_fine_upper, u0_fine_upper = initialize_vortices(grid_fine_upper)

num_vortices = (grid_coarse.size + grid_fine_lower.size + grid_fine_upper.size) // 2
print("* Number of vortices (total):", num_vortices)

# ---------------------------
# Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, y, delta=delta):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    k1 = 0.0
    k2 = 0.0
    j_values = [-1, 0, 1, 2]
    for j in j_values:
        diff1 = np.array([x1 - y1, x2 - (y2 + 12*j)])
        r2 = diff1[0]**2 + diff1[1]**2
        if r2 < 1e-10:
            term1 = 0.0
        else:
            moll1 = 1 - np.exp(-(r2/delta)**2)
            term1 = (x2 - (y2 + 12*j)) / r2 * moll1
        diff2 = np.array([x1 - y1, x2 - (-y2 + 12*j)])
        r2_bar = diff2[0]**2 + diff2[1]**2
        if r2_bar < 1e-10:
            term1_bar = 0.0
        else:
            moll2 = 1 - np.exp(-(r2_bar/delta)**2)
            term1_bar = (x2 - (-y2 + 12*j)) / r2_bar * moll2
        k1 += (term1 - term1_bar)
        
        if r2 < 1e-10:
            term2 = 0.0
        else:
            term2 = (x1 - y1) / r2 * moll1
        if r2_bar < 1e-10:
            term2_bar = 0.0
        else:
            term2_bar = (x1 - y1) / r2_bar * moll2
        k2 += (term2 - term2_bar)
    k1 = - k1 / (2 * np.pi)
    k2 = k2 / (2 * np.pi)
    return np.array([k1, k2])

def indicator_D(point):
    return 1 if (point[1] > 0 and point[1] < H) else 0

# ---------------------------
# Function Factory for u_func
# ---------------------------
def make_u_func(current_coarse, current_fine_lower, current_fine_upper,
                w0_coarse, w0_fine_lower, w0_fine_upper,
                A_coarse, A_fine_lower, A_fine_upper):
    current_coarse_copy = current_coarse.copy()
    current_fine_lower_copy = current_fine_lower.copy()
    current_fine_upper_copy = current_fine_upper.copy()
    
    def u_func(x):
        u_val = np.zeros(2)
        n1, n2 = current_coarse_copy.shape[0], current_coarse_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                contrib = K_delta(pos, x, delta)
                u_val += indicator_D(pos) * contrib * w0_coarse[i, j] * A_coarse[i, j]
        n1_f, n2_f = current_fine_lower_copy.shape[0], current_fine_lower_copy.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                pos = current_fine_lower_copy[i, j]
                contrib = K_delta(pos, x, delta)
                u_val += indicator_D(pos) * contrib * w0_fine_lower[i, j] * A_fine_lower[i, j]
        n1_fu, n2_fu = current_fine_upper_copy.shape[0], current_fine_upper_copy.shape[1]
        for i in range(n1_fu):
            for j in range(n2_fu):
                pos = current_fine_upper_copy[i, j]
                contrib = K_delta(pos, x, delta)
                u_val += indicator_D(pos) * contrib * w0_fine_upper[i, j] * A_fine_upper[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama)
# ---------------------------
def simulate_vortex_trajectories():
    traj_coarse = np.zeros((num_steps + 1, *grid_coarse.shape))
    traj_fine_lower = np.zeros((num_steps + 1, *grid_fine_lower.shape))
    traj_fine_upper = np.zeros((num_steps + 1, *grid_fine_upper.shape))
    traj_coarse[0] = grid_coarse
    traj_fine_lower[0] = grid_fine_lower
    traj_fine_upper[0] = grid_fine_upper
    
    uFuncs = []
    disp_coarse = np.zeros((num_steps, *grid_coarse.shape))
    disp_fine_lower = np.zeros((num_steps, *grid_fine_lower.shape))
    disp_fine_upper = np.zeros((num_steps, *grid_fine_upper.shape))
    
    for step in range(num_steps):
        current_coarse = traj_coarse[step]
        current_fine_lower = traj_fine_lower[step]
        current_fine_upper = traj_fine_upper[step]
        
        u_func = make_u_func(current_coarse, current_fine_lower, current_fine_upper,
                             w0_coarse, w0_fine_lower, w0_fine_upper,
                             A_coarse, A_fine_lower, A_fine_upper)
        uFuncs.append(u_func)
        
        n1, n2 = current_coarse.shape[0], current_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_coarse[step, i, j] = disp
                traj_coarse[step + 1, i, j] = current_coarse[i, j] + disp
        
        n1_f, n2_f = current_fine_lower.shape[0], current_fine_lower.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                u_val = u_func(current_fine_lower[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_fine_lower[step, i, j] = disp
                traj_fine_lower[step + 1, i, j] = current_fine_lower[i, j] + disp
        
        n1_fu, n2_fu = current_fine_upper.shape[0], current_fine_upper.shape[1]
        for i in range(n1_fu):
            for j in range(n2_fu):
                u_val = u_func(current_fine_upper[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_fine_upper[step, i, j] = disp
                traj_fine_upper[step + 1, i, j] = current_fine_upper[i, j] + disp
                
    return (traj_coarse, traj_fine_lower, traj_fine_upper), uFuncs, (disp_coarse, disp_fine_lower, disp_fine_upper)

print("Computing vortex trajectories......")
start_time = time.time()  # Start timer for trajectory computation
(trajectories_coarse, trajectories_fine_lower, trajectories_fine_upper), uFuncs, _ = simulate_vortex_trajectories()
end_time = time.time()  # End timer for trajectory computation
print(f"Time for trajectory computation: {end_time - start_time:.2f} seconds")

# ---------------------------
# Boat Simulation (using the velocity functions from vortex simulation)
# ---------------------------
def generate_boat_grid():
    grids = generate_nonuniform_grid_D()
    grid_coarse, _ = grids['coarse']
    grid_fine_lower, _ = grids['fine_lower']
    grid_fine_upper, _ = grids['fine_upper']
    boat_grid = np.concatenate((
        grid_coarse.reshape(-1, 2),
        grid_fine_lower.reshape(-1, 2),
        grid_fine_upper.reshape(-1, 2)
    ), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    boat_grid = generate_boat_grid()
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

print("Simulating vortex boats......")
start_time = time.time()  # Start timer for boat simulation
boat_positions, boat_displacements = simulate_boats(uFuncs)
end_time = time.time()  # End timer for boat simulation
print(f"Time for boat simulation: {end_time - start_time:.2f} seconds")

# ---------------------------
# Velocity Field Computation for Visualization
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
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

query_grid = generate_boat_grid()
U, V = compute_velocity_field(uFuncs[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

boat_scatter = ax.scatter(boat_positions[0, :, 0], boat_positions[0, :, 1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    u_func = uFuncs[frame] if frame < len(uFuncs) else uFuncs[-1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    boat_scatter.set_offsets(boat_positions[frame])
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps + 1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "belt_domain.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
