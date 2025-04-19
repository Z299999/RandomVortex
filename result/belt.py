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
delta = 0.1        # Mollification parameter for kernel (used in K_delta)

# Domain parameters:
H = 6              # Domain height (belt with walls at y=0 and y=H)
region_x = [-6, 6]
region_y = [0, H]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

# Mesh parameters:
h0 = 0.8             # Coarse grid spacing for both x and y
h1 = 0.6           # Fine grid spacing along x
h2 = 0.3           # Fine grid spacing along y
layer_thickness = 0.8  # Thickness of the fine grid layer near boundaries
fh = layer_thickness   # alias

np.random.seed(42)

# ---------------------------
# Vorticity and Velocity Functions
# ---------------------------
def velocity(x, y, H=H):
    # A simple velocity field with one sine period in y.
    return np.array([-0.5 * np.sin(2 * np.pi * y / H), 0])

def vorticity(x, y, H=H):
    # Vorticity: (π/H)*cos(2πy/H)
    return (np.pi / H) * np.cos(2 * np.pi * y / H)

# ---------------------------
# Grid Generation (Modified to include boundaries)
# ---------------------------
def generate_nonuniform_grid_D():
    """
    Generate a nonuniform grid covering the entire domain [0, H] in y.
    It is split into three parts:
      1. Lower fine grid: y in [0, fh) (includes y = 0)
      2. Coarse grid: y in [fh, H - fh] (includes both endpoints)
      3. Upper fine grid: y in (H - fh, H] (includes y = H)
    In the x-direction:
      - Fine grid uses spacing h1
      - Coarse grid uses spacing h0
    """
    x1, x2 = region_x

    # Lower fine grid: use np.arange to include 0, but not fh.
    y_vals_lower = np.arange(0, fh, h2)  # e.g., if fh = 0.8 and h2 = 0.4 => [0.0, 0.4]
    
    # Upper fine grid: cover (H - fh, H]. Create points from H-fh to H.
    temp = np.arange(H - fh, H + h2/2, h2)
    if len(temp) > 1:
        y_vals_upper = temp[1:]  # skip the first point (which overlaps with coarse grid)
    else:
        y_vals_upper = temp

    # Coarse grid: evenly spaced in [fh, H - fh]
    num_y_coarse = int(round((H - 2*fh) / h0)) + 1
    y_coarse = np.linspace(fh, H - fh, num_y_coarse)
    
    # x-direction fine grid (for fine regions)
    num_x_fine = int((x2 - x1) / h1) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    
    # Lower fine grid
    xx_fine_lower, yy_fine_lower = np.meshgrid(x_fine, y_vals_lower, indexing='ij')
    grid_fine_lower = np.stack((xx_fine_lower, yy_fine_lower), axis=-1)
    A_fine_lower = h1 * h2 * np.ones(grid_fine_lower.shape[:2])
    
    # Coarse grid (x uses h0, y from coarse)
    num_x_coarse = int((x2 - x1) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones(grid_coarse.shape[:2])
    
    # Upper fine grid
    xx_fine_upper, yy_fine_upper = np.meshgrid(x_fine, y_vals_upper, indexing='ij')
    grid_fine_upper = np.stack((xx_fine_upper, yy_fine_upper), axis=-1)
    A_fine_upper = h1 * h2 * np.ones(grid_fine_upper.shape[:2])
    
    print(f"Lower fine grid shape: {grid_fine_lower.shape}")
    print(f"Upper fine grid shape: {grid_fine_upper.shape}")
    print(f"Coarse grid shape: {grid_coarse.shape}")
    
    return {'coarse': (grid_coarse, A_coarse),
            'fine_lower': (grid_fine_lower, A_fine_lower),
            'fine_upper': (grid_fine_upper, A_fine_upper)}

grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_fine_lower, A_fine_lower = grids['fine_lower']
grid_fine_upper, A_fine_upper = grids['fine_upper']

# ---------------------------
# Mollified Biot-Savart Kernel and Domain Indicator
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
    k1 = -k1/(2*np.pi)
    k2 = k2/(2*np.pi)
    return np.array([k1, k2])

def indicator_D(point):
    return 1 if (point[1] >= 0 and point[1] <= H) else 0

# ---------------------------
# Function Factory for u_func (with Boundary Correction)
# ---------------------------
def make_u_func(current_coarse, current_fine_lower, current_fine_upper,
                w0_coarse, w0_fine_lower, w0_fine_upper,
                A_coarse, A_fine_lower, A_fine_upper,
                theta0, theta1, x_bound):
    """
    Constructs the velocity function u(x) by splitting the vorticity into a regular
    part and a boundary correction. The boundary correction term uses theta0 and theta1,
    which are the current boundary vorticities for the bottom and top walls.
    x_bound is the array of x coordinates at which the boundary theta values are defined.
    """
    # Create local copies of the fields.
    w0_coarse_copy = np.copy(w0_coarse)
    w0_fine_lower_copy = np.copy(w0_fine_lower)
    w0_fine_upper_copy = np.copy(w0_fine_upper)
    
    current_coarse_copy = current_coarse.copy()
    current_fine_lower_copy = current_fine_lower.copy()
    current_fine_upper_copy = current_fine_upper.copy()
    
    # Define a cutoff function phi(r): 1 for r < 1/3, linearly decreasing for 1/3 ≤ r < 2/3, and 0 for r ≥ 2/3.
    def phi(r):
        if r < 1/3:
            return 1.0
        elif r < 2/3:
            s = (r - 1/3) / (2/3 - 1/3)
            return 1 - s
        else:
            return 0.0
    eps = 0.5  # Boundary mollification parameter
    
    def grid_contribution(x, grid, w0_grid, A_grid):
        """Compute contribution from one grid (coarse or fine) at evaluation point x."""
        contrib_reg = np.zeros(2)
        contrib_bound = np.zeros(2)
        nx, ny = grid.shape[0], grid.shape[1]
        for i in range(nx):
            for j in range(ny):
                pos = grid[i, j]
                # Compute cutoff values for bottom and top boundaries.
                phi_bot = phi(pos[1] / eps)          # Near y = 0
                phi_top = phi((H - pos[1]) / eps)      # Near y = H
                # Interpolate the boundary vorticity using the updated theta values.
                w_boundary = np.interp(pos[0], x_bound, theta0) * phi_bot + \
                             np.interp(pos[0], x_bound, theta1) * phi_top
                # Regularized vorticity.
                w_reg = w0_grid[i, j] - w_boundary
                if indicator_D(pos):
                    contrib_reg += K_delta(pos, x, delta) * w_reg * A_grid[i, j]
                    contrib_bound += K_delta(pos, x, delta) * w_boundary * A_grid[i, j]
        return contrib_reg + contrib_bound

    def u_func(x):
        u_val = np.zeros(2)
        u_val += grid_contribution(x, current_coarse_copy, w0_coarse_copy, A_coarse)
        u_val += grid_contribution(x, current_fine_lower_copy, w0_fine_lower_copy, A_fine_lower)
        u_val += grid_contribution(x, current_fine_upper_copy, w0_fine_upper_copy, A_fine_upper)
        return u_val
    return u_func

# ---------------------------
# Initialize Vorticity on Each Grid (Static Initial Condition)
# ---------------------------
def initialize_vorticity(grid):
    n_x, n_y, _ = grid.shape
    w = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            x_pt, y_pt = grid[i, j]
            w[i, j] = vorticity(x_pt, y_pt, H)
    return w

w0_coarse = initialize_vorticity(grid_coarse)
w0_fine_lower = initialize_vorticity(grid_fine_lower)
w0_fine_upper = initialize_vorticity(grid_fine_upper)

# ---------------------------
# Initialize Boundary Vorticity (theta) and Parameters
# ---------------------------
# Use the coarse grid x-coordinates as sampling points for the boundary.
x_bound = grid_coarse[:, 0, 0]  # x-coordinates from the coarse grid (each column)
N_bound = len(x_bound)
# Allocate arrays for theta (for all time steps) for bottom (y = 0) and top (y = H)
theta0_all = np.zeros((num_steps+1, N_bound))  # Bottom boundary vorticity
theta1_all = np.zeros((num_steps+1, N_bound))  # Top boundary vorticity
# Initial boundary vorticity from the vorticity function.
theta0_all[0, :] = (np.pi / H) * np.cos(0)    # cos(0)=1
theta1_all[0, :] = (np.pi / H) * np.cos(2*np.pi)  # cos(2pi)=1

# Monte Carlo parameters for boundary update:
M_boundary = 200   # Number of Monte Carlo samples
sigma = np.sqrt(4 * nu * dt)  # Standard deviation for sampling x-values on the boundary

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama with Monte Carlo Boundary Theta Updates)
# ---------------------------
def simulate_vortex_trajectories():
    # Initialize trajectory arrays for each grid.
    traj_coarse = np.zeros((num_steps + 1, *grid_coarse.shape))
    traj_fine_lower = np.zeros((num_steps + 1, *grid_fine_lower.shape))
    traj_fine_upper = np.zeros((num_steps + 1, *grid_fine_upper.shape))
    traj_coarse[0] = grid_coarse
    traj_fine_lower[0] = grid_fine_lower
    traj_fine_upper[0] = grid_fine_upper

    uFuncs = []  # List to store the velocity functions at each time step
    global theta0_all, theta1_all  # Use the global arrays for boundary theta

    for step in range(num_steps):
        current_coarse = traj_coarse[step]
        current_fine_lower = traj_fine_lower[step]
        current_fine_upper = traj_fine_upper[step]
        # Retrieve current boundary theta values.
        theta0_current = theta0_all[step, :]
        theta1_current = theta1_all[step, :]

        # Construct velocity function (with boundary correction) at the current time step.
        u_func = make_u_func(current_coarse, current_fine_lower, current_fine_upper,
                             w0_coarse, w0_fine_lower, w0_fine_upper,
                             A_coarse, A_fine_lower, A_fine_upper,
                             theta0_current, theta1_current, x_bound)
        uFuncs.append(u_func)

        # Update vortex positions using Euler-Maruyama.
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

        # ---------------------------
        # Monte Carlo Update for Boundary Theta (Only smoothing, ψ and φ neglected)
        # ---------------------------
        theta0_new = np.zeros(N_bound)
        theta1_new = np.zeros(N_bound)
        for idx, x_val in enumerate(x_bound):
            # For bottom boundary: sample x-values from N(x_val, sigma)
            x_samples = np.random.normal(loc=x_val, scale=sigma, size=M_boundary)
            # Interpolate the current boundary theta values and average them.
            theta0_samples = np.interp(x_samples, x_bound, theta0_current)
            theta0_new[idx] = np.mean(theta0_samples)
            
            # For top boundary:
            x_samples_top = np.random.normal(loc=x_val, scale=sigma, size=M_boundary)
            theta1_samples = np.interp(x_samples_top, x_bound, theta1_current)
            theta1_new[idx] = np.mean(theta1_samples)
        
        theta0_all[step+1, :] = theta0_new
        theta1_all[step+1, :] = theta1_new

    return uFuncs

print("Computing vortex trajectories and updating boundary theta using Monte Carlo smoothing...")
start_time = time.time()
uFuncs = simulate_vortex_trajectories()  # Obtain the velocity function at each time step
end_time = time.time()
print(f"Time for vortex trajectory and boundary update simulation: {end_time - start_time:.2f} seconds")

# ---------------------------
# Visualization Utilities for Streamlines
# ---------------------------
# Custom colormap from white (slow) to red (fast)
my_cmap = LinearSegmentedColormap.from_list("my_cmap", ["#ffffff", "#ff0000"])

# Uniform background grid for velocity magnitude calculation
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

# Determine the color scale based on the initial time step.
initial_u_func = uFuncs[0]
mag0 = compute_velocity_magnitude(initial_u_func, X_bg, Y_bg)
vmin = 0
vmax = np.max(mag0)
global_max = vmax

def compute_uv_on_grid(u_func, X, Y):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vel = u_func(np.array([X[i, j], Y[i, j]]))
            U[i, j] = vel[0]
            V[i, j] = vel[1]
    return U, V

def plot_streamlines(u_func, t_current):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Tighten the margins so that the plot is almost flush with the edges.
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(window_x[0], window_x[1])
    ax.set_ylim(window_y[0], window_y[1])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Streamlines at t = {t_current:.2f}", fontsize=16)
    
    U_bg, V_bg = compute_uv_on_grid(u_func, X_bg, Y_bg)
    mag_bg = np.sqrt(U_bg**2 + V_bg**2)
    ax.imshow(mag_bg, extent=(window_x[0], window_x[1], window_y[0], window_y[1]),
              origin='lower', cmap=my_cmap, alpha=0.5, vmin=0, vmax=global_max)
    
    # Generate seeding points: uniformly in x and using cosine spacing in y to cluster near boundaries.
    num_x_seeds = 20
    num_y_seeds = 40
    x_seeds = np.linspace(window_x[0], window_x[1], num_x_seeds)
    theta_seed = np.linspace(0, np.pi, num_y_seeds)
    y_seeds = 0.5 * (window_y[1] - window_y[0]) * (1 - np.cos(theta_seed)) + window_y[0]
    X_seed, Y_seed = np.meshgrid(x_seeds, y_seeds, indexing='ij')
    start_points = np.column_stack((X_seed.ravel(), Y_seed.ravel()))
    
    ax.streamplot(x_bg, y_bg, U_bg, V_bg, color='k', linewidth=1, start_points=start_points)
    return fig, ax

# ---------------------------
# Save Streamline Images at Specified Times (Exporting as SVG)
# ---------------------------
save_times = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
save_frames = [int(t/dt) for t in save_times]

output_folder = os.path.join("figure", "belt")
os.makedirs(output_folder, exist_ok=True)

for frame in save_frames:
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    fig, ax = plot_streamlines(u_func, t_current)
    filename = os.path.join(output_folder, f"streamlines_t{t_current:05.2f}.svg")
    # Save as SVG with tight margins.
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved streamlines at t = {t_current:.2f} to {filename}")
