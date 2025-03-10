import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time  # For timing

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01          # Viscosity
T = 20             # Final time (seconds)
dt = 0.1           # Time step
num_steps = int(T / dt)
delta = 0.1        # Mollification parameter

# Domain parameters:
H = 4              # Adjustable domain height
region_x = [-6, 6]
region_y = [0, H]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

# Mesh parameters:
h0 = 1             # Coarse mesh grid spacing
h1 = 0.5           # Fine mesh grid spacing along x
h2 = 0.4           # Fine mesh grid spacing along y
layer_thickness = 0.8  # Boundary layer thickness for both walls
hh = layer_thickness   # alias

np.random.seed(42)

# ---------------------------
# Vorticity and Velocity Functions
# ---------------------------
def velocity(x, y, H=H):
    # Velocity vanishes at y=0 and y=H.
    return np.array([-0.5 * np.sin(2 * np.pi * y / H), 0])

def vorticity(x, y, H=H):
    # Vorticity computed as (pi/H)*cos(pi*y/H)
    return (np.pi / H) * np.cos(np.pi * y / H)

# ---------------------------
# Grid Generation
# ---------------------------
def generate_nonuniform_grid_D():
    """
    Generates a symmetric nonuniform grid in [0,H] with three parts:
      1. Lower fine grid: y in [0, hh) with step size h2 (endpoint excluded).
      2. Upper fine grid: reflected from lower via y -> H - y.
      3. Coarse grid: y in [hh, H - hh] (endpoints included).
    In x-direction, fine grids use spacing h1 and coarse grid uses spacing h0.
    Returns a dictionary with keys 'coarse', 'fine_lower', and 'fine_upper'.
    """
    x1, x2 = region_x
    # Lower Fine Grid:
    y_vals_lower = np.arange(0, hh, h2)
    num_x_fine = int((x2 - x1) / h1) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    xx_fine_lower, yy_fine_lower = np.meshgrid(x_fine, y_vals_lower, indexing='ij')
    grid_fine_lower = np.stack((xx_fine_lower, yy_fine_lower), axis=-1)
    A_fine_lower = h1 * h2 * np.ones((num_x_fine, len(y_vals_lower)))
    
    # Upper Fine Grid:
    y_vals_upper = H - y_vals_lower
    y_vals_upper = np.sort(y_vals_upper)
    xx_fine_upper, yy_fine_upper = np.meshgrid(x_fine, y_vals_upper, indexing='ij')
    grid_fine_upper = np.stack((xx_fine_upper, yy_fine_upper), axis=-1)
    A_fine_upper = h1 * h2 * np.ones((num_x_fine, len(y_vals_upper)))
    
    # Coarse Grid:
    num_y_coarse = int(round((H - 2 * hh) / h0)) + 1
    num_x_coarse = int((x2 - x1) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(hh, H - hh, num_y_coarse, endpoint=True)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))
    
    print(f"Lower Fine grid: {grid_fine_lower.shape}")
    print(f"Upper Fine grid: {grid_fine_upper.shape}")
    print(f"Coarse grid: {grid_coarse.shape}")
    
    return {'coarse': (grid_coarse, A_coarse),
            'fine_lower': (grid_fine_lower, A_fine_lower),
            'fine_upper': (grid_fine_upper, A_fine_upper)}

# Generate grids
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_fine_lower, A_fine_lower = grids['fine_lower']
grid_fine_upper, A_fine_upper = grids['fine_upper']

# ---------------------------
# Mollified Biot-Savart Kernel and Indicator
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
    # Local copies of the constant vorticity arrays
    w0_coarse_copy = np.copy(w0_coarse)
    w0_fine_lower_copy = np.copy(w0_fine_lower)
    w0_fine_upper_copy = np.copy(w0_fine_upper)
    
    current_coarse_copy = current_coarse.copy()
    current_fine_lower_copy = current_fine_lower.copy()
    current_fine_upper_copy = current_fine_upper.copy()
    
    def u_func(x):
        u_val = np.zeros(2)
        # Contribution from the coarse grid
        n1, n2 = current_coarse_copy.shape[0], current_coarse_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                u_val += indicator_D(pos) * K_delta(pos, x, delta) * w0_coarse_copy[i, j] * A_coarse[i, j]
        # Contribution from the fine lower grid
        n1_f, n2_f = current_fine_lower_copy.shape[0], current_fine_lower_copy.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                pos = current_fine_lower_copy[i, j]
                u_val += indicator_D(pos) * K_delta(pos, x, delta) * w0_fine_lower_copy[i, j] * A_fine_lower[i, j]
        # Contribution from the fine upper grid
        n1_fu, n2_fu = current_fine_upper_copy.shape[0], current_fine_upper_copy.shape[1]
        for i in range(n1_fu):
            for j in range(n2_fu):
                pos = current_fine_upper_copy[i, j]
                u_val += indicator_D(pos) * K_delta(pos, x, delta) * w0_fine_upper_copy[i, j] * A_fine_upper[i, j]
        return u_val
    return u_func

# ---------------------------
# Precompute Constant Vorticity for Each Grid
# ---------------------------
def initialize_vorticity(grid):
    n_x, n_y, _ = grid.shape
    w = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y, H)
    return w

w0_coarse = initialize_vorticity(grid_coarse)
w0_fine_lower = initialize_vorticity(grid_fine_lower)
w0_fine_upper = initialize_vorticity(grid_fine_upper)

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

    uFuncs = []  # List to store velocity functions at each time step

    for step in range(num_steps):
        current_coarse = traj_coarse[step]
        current_fine_lower = traj_fine_lower[step]
        current_fine_upper = traj_fine_upper[step]

        u_func = make_u_func(current_coarse, current_fine_lower, current_fine_upper,
                             w0_coarse, w0_fine_lower, w0_fine_upper,
                             A_coarse, A_fine_lower, A_fine_upper)
        uFuncs.append(u_func)

        # Update positions using Euler-Maruyama
        n1, n2 = current_coarse.shape[0], current_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                traj_coarse[step + 1, i, j] = current_coarse[i, j] + disp

        n1_f, n2_f = current_fine_lower.shape[0], current_fine_lower.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                u_val = u_func(current_fine_lower[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                traj_fine_lower[step + 1, i, j] = current_fine_lower[i, j] + disp

        n1_fu, n2_fu = current_fine_upper.shape[0], current_fine_upper.shape[1]
        for i in range(n1_fu):
            for j in range(n2_fu):
                u_val = u_func(current_fine_upper[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                traj_fine_upper[step + 1, i, j] = current_fine_upper[i, j] + disp

    return uFuncs

print("Computing vortex trajectories...")
start_time = time.time()
uFuncs = simulate_vortex_trajectories()  # List of velocity functions at each time step
end_time = time.time()
print(f"Time for trajectory computation: {end_time - start_time:.2f} seconds")

# ---------------------------
# Visualization Utilities for Streamlines
# ---------------------------
# Custom colormap from white (slow) to red (fast)
my_cmap = LinearSegmentedColormap.from_list("my_cmap", ["#ffffff", "#ff0000"])

# Background grid for velocity magnitude computation
Nx_bg, Ny_bg = 100, 100
x_bg = np.linspace(window_x[0], window_x[1], Nx_bg)
y_bg = np.linspace(window_y[0], window_y[1], Ny_bg)
X_bg, Y_bg = np.meshgrid(x_bg, y_bg)

def compute_velocity_magnitude(u_func, X, Y):
    query_points = np.column_stack((X.ravel(), Y.ravel()))
    U = np.zeros(query_points.shape[0])
    V = np.zeros(query_points.shape[0])
    for i, pt in enumerate(query_points):
        vel = u_func(pt)
        U[i] = vel[0]
        V[i] = vel[1]
    mag = np.sqrt(U**2 + V**2).reshape(X.shape)
    return mag

# Fix color scale based on initial time step
initial_u_func = uFuncs[0]
mag0 = compute_velocity_magnitude(initial_u_func, X_bg, Y_bg)
vmin = 0
vmax = np.max(mag0)
global_max = vmax

def compute_uv_on_grid(u_func, X, Y):
    vec_u = np.vectorize(lambda x, y: u_func(np.array([x, y]))[0])
    vec_v = np.vectorize(lambda x, y: u_func(np.array([x, y]))[1])
    U = vec_u(X, Y)
    V = vec_v(X, Y)
    return U, V

def plot_streamlines(u_func, t_current):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(window_x[0], window_x[1])
    ax.set_ylim(window_y[0], window_y[1])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Streamlines at t={t_current:.2f}")
    
    # Compute background velocity magnitude on the global grid
    vel_bg = np.array([u_func(np.array([x, y])) for x, y in zip(X_bg.ravel(), Y_bg.ravel())])
    mag_bg = np.linalg.norm(vel_bg, axis=1).reshape(X_bg.shape)
    ax.imshow(mag_bg, extent=(window_x[0], window_x[1], window_y[0], window_y[1]),
              origin='lower', cmap=my_cmap, alpha=0.5, vmin=0, vmax=global_max)
    
    # Lower region: dense seeding near lower boundary (y from 0 to layer_thickness)
    x_lower = np.linspace(region_x[0], region_x[1], 50)
    y_lower = np.linspace(region_y[0], layer_thickness, 50)
    U_lower, V_lower = compute_uv_on_grid(u_func, *np.meshgrid(x_lower, y_lower, indexing='ij'))
    ax.streamplot(x_lower, y_lower, U_lower, V_lower, color='k', linewidth=1)
    
    # Upper region: dense seeding near upper boundary (y from H - layer_thickness to H)
    x_upper = np.linspace(region_x[0], region_x[1], 50)
    y_upper = np.linspace(H - layer_thickness, H, 50)
    U_upper, V_upper = compute_uv_on_grid(u_func, *np.meshgrid(x_upper, y_upper, indexing='ij'))
    ax.streamplot(x_upper, y_upper, U_upper, V_upper, color='k', linewidth=1)
    
    # Middle region: intermediate seeding (y from layer_thickness to H - layer_thickness)
    x_mid = np.linspace(region_x[0], region_x[1], 30)
    y_mid = np.linspace(layer_thickness, H - layer_thickness, 30)
    U_mid, V_mid = compute_uv_on_grid(u_func, *np.meshgrid(x_mid, y_mid, indexing='ij'))
    ax.streamplot(x_mid, y_mid, U_mid, V_mid, color='k', linewidth=1)
    
    return fig, ax

# ---------------------------
# Save Streamline Images at Specified Times
# ---------------------------
# Times (seconds) at which to save streamline images
save_times = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
save_frames = [int(t/dt) for t in save_times]

# Save images into the "belt" subfolder inside the "figure" folder.
output_folder = os.path.join("figure", "belt")
os.makedirs(output_folder, exist_ok=True)

for frame in save_frames:
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    fig, ax = plot_streamlines(u_func, t_current)
    filename = os.path.join(output_folder, f"streamlines_t{t_current:05.2f}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved streamline image at t={t_current:.2f} to {filename}")
