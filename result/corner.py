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

# Mesh parameters:
h0 = 0.8    # Coarse grid spacing (both x and y)
h1 = 0.4  # Fine grid spacing for Region B (x-direction) and Region C (y-direction)
h2 = 0.2  # Fine grid spacing for Region A (both x and y) and for Region B (y) and Region C (x)
layer_thickness = 0.8  # dividing length between fine and coarse regions

region_x = [0, 8]
region_y = [0, 8]
window_x = [region_x[0], region_x[1] ]
window_y = [region_y[0], region_y[1] ]

np.random.seed(42)

# ---------------------------
# Vorticity, Velocity and Grid
# ---------------------------
def velocity(x, y):
    factor = np.sqrt(2) / 2
    common_term = -factor * np.sin(factor * (y - x))
    return np.array([common_term, common_term])

def vorticity(x, y):
    factor = np.sqrt(2) / 2
    return np.cos(factor * (y - x))

def generate_nonuniform_grid_D():
    """
    Generates four meshes for the domain:
      - Coarse grid: [x1+hh, x2] x [y1+hh, y2] with spacing h0.
      - Region A (Dbxy): [x1, x1+hh) x [y1, y1+hh) with spacing h2.
      - Region B (Dbx): [x1+hh, x2] x [y1, y1+hh) with spacing (h1 for x and h2 for y).
      - Region C (Dby): [x1, x1+hh) x [y1+hh, y2] with spacing (h2 for x and h1 for y).
    Returns grids and associated cell areas.
    """
    x1, x2 = region_x
    y1, y2 = region_y
    hh = layer_thickness

    # Coarse grid:
    num_x_coarse = int((x2 - (x1 + hh)) / h0) + 1
    num_y_coarse = int((y2 - (y1 + hh)) / h0) + 1
    x_coarse = np.linspace(x1 + hh, x2, num_x_coarse)
    y_coarse = np.linspace(y1 + hh, y2, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))

    # Region A (Dbxy):
    num_x_Dbxy = int(hh / h2)
    num_y_Dbxy = int(hh / h2)
    x_Dbxy = np.linspace(x1, x1 + hh, num_x_Dbxy, endpoint=False)
    y_Dbxy = np.linspace(y1, y1 + hh, num_y_Dbxy, endpoint=False)
    xx_Dbxy, yy_Dbxy = np.meshgrid(x_Dbxy, y_Dbxy, indexing='ij')
    grid_Dbxy = np.stack((xx_Dbxy, yy_Dbxy), axis=-1)
    A_Dbxy = h2 * h2 * np.ones((num_x_Dbxy, num_y_Dbxy))

    # Region B (Dbx):
    num_x_Dbx = int((x2 - (x1 + hh)) / h1) + 1
    num_y_Dbx = int(hh / h2)
    x_Dbx = np.linspace(x1 + hh, x2, num_x_Dbx)
    y_Dbx = np.linspace(y1, y1 + hh, num_y_Dbx, endpoint=False)
    xx_Dbx, yy_Dbx = np.meshgrid(x_Dbx, y_Dbx, indexing='ij')
    grid_Dbx = np.stack((xx_Dbx, yy_Dbx), axis=-1)
    A_Dbx = h1 * h2 * np.ones((num_x_Dbx, num_y_Dbx))

    # Region C (Dby):
    num_x_Dby = int(hh / h2)
    num_y_Dby = int((y2 - (y1 + hh)) / h1) + 1
    x_Dby = np.linspace(x1, x1 + hh, num_x_Dby, endpoint=False)
    y_Dby = np.linspace(y1 + hh, y2, num_y_Dby)
    xx_Dby, yy_Dby = np.meshgrid(x_Dby, y_Dby, indexing='ij')
    grid_Dby = np.stack((xx_Dby, yy_Dby), axis=-1)
    A_Dby = h2 * h1 * np.ones((num_x_Dby, num_y_Dby))

    print(f"Coarse grid: {grid_coarse.shape[0]} x {grid_coarse.shape[1]} = {grid_coarse.shape[0]*grid_coarse.shape[1]} points")
    print(f"Region A (Dbxy): {grid_Dbxy.shape[0]} x {grid_Dbxy.shape[1]} = {grid_Dbxy.shape[0]*grid_Dbxy.shape[1]} points")
    print(f"Region B (Dbx): {grid_Dbx.shape[0]} x {grid_Dbx.shape[1]} = {grid_Dbx.shape[0]*grid_Dbx.shape[1]} points")
    print(f"Region C (Dby): {grid_Dby.shape[0]} x {grid_Dby.shape[1]} = {grid_Dby.shape[0]*grid_Dby.shape[1]} points")
    
    return {
        'coarse': (grid_coarse, A_coarse),
        'Dbxy': (grid_Dbxy, A_Dbxy),
        'Dbx': (grid_Dbx, A_Dbx),
        'Dby': (grid_Dby, A_Dby)
    }

# Generate the grids
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_Dbxy, A_Dbxy = grids['Dbxy']
grid_Dbx, A_Dbx = grids['Dbx']
grid_Dby, A_Dby = grids['Dby']

# Combine the three fine regions for simulation:
grid_fine = np.concatenate((grid_Dbxy.reshape(-1, 2),
                            grid_Dbx.reshape(-1, 2),
                            grid_Dby.reshape(-1, 2)), axis=0)
A_fine = np.concatenate((A_Dbxy.reshape(-1),
                         A_Dbx.reshape(-1),
                         A_Dby.reshape(-1)), axis=0)

def plot_grid(grid_coarse, grid_Dbxy, grid_Dbx, grid_Dby):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(region_x[0], region_x[1])
    ax.set_ylim(region_y[0], region_y[1])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Nonuniform Grid Visualization")
    coarse_points = grid_coarse.reshape(-1, 2)
    ax.scatter(coarse_points[:, 0], coarse_points[:, 1], color='red', label='Coarse Grid', s=20)
    ax.scatter(grid_Dbxy[:, :, 0].reshape(-1), grid_Dbxy[:, :, 1].reshape(-1), color='blue', label='Region A (Dbxy)', s=20)
    ax.scatter(grid_Dbx[:, :, 0].reshape(-1), grid_Dbx[:, :, 1].reshape(-1), color='green', label='Region B (Dbx)', s=20)
    ax.scatter(grid_Dby[:, :, 0].reshape(-1), grid_Dby[:, :, 1].reshape(-1), color='purple', label='Region C (Dby)', s=20)
    ax.legend()
    plt.show()

# Optional: visualize the grid
# plot_grid(grid_coarse, grid_Dbxy, grid_Dbx, grid_Dby)

# ---------------------------
# Vortex Initialization (for 3D grids)
# ---------------------------
def initialize_vortices_grid(grid):
    n_i, n_j, _ = grid.shape
    w = np.zeros((n_i, n_j))
    u = np.zeros((n_i, n_j, 2))
    for i in range(n_i):
        for j in range(n_j):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y)
            u[i, j] = velocity(x, y)
    return w, u

w0_coarse, u0_coarse = initialize_vortices_grid(grid_coarse)
w0_Dbxy, u0_Dbxy = initialize_vortices_grid(grid_Dbxy)
w0_Dbx, u0_Dbx = initialize_vortices_grid(grid_Dbx)
w0_Dby, u0_Dby = initialize_vortices_grid(grid_Dby)

# ---------------------------
# Mollified Biot-Savart Kernel and Helpers
# ---------------------------
def K_Q(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r2 = (x1 - y1)**2 + (x2 - y2)**2
    r2_bar1 = (x1 + y1)**2 + (x2 - y2)**2
    r2_bar2 = (x1 - y1)**2 + (x2 + y2)**2
    r2_bar3 = (x1 + y1)**2 + (x2 + y2)**2
    if r2 < 1e-10 or r2_bar1 < 1e-10 or r2_bar2 < 1e-10 or r2_bar3 < 1e-10:
        return np.zeros(2)
    k1 = -0.5/np.pi * ((x2 - y2)/r2 - (x2 - y2)/r2_bar1 - (x2 + y2)/r2_bar2 + (x2 + y2)/r2_bar3)
    k2 = 0.5/np.pi * ((x1 - y1)/r2 - (x1 + y1)/r2_bar1 - (x1 - y1)/r2_bar2 + (x1 + y1)/r2_bar3)
    factor = 1 - np.exp(- (r2 / delta)**2)
    return np.array([k1, k2]) * factor

def indicator_D(point):
    return 1 if (point[0] > 0 and point[1] > 0) else 0

def Rx(point):
    return np.array([point[0], -point[1]])

def Ry(point):
    return np.array([-point[0], point[1]])

# ---------------------------
# Function Factory for u_func
# ---------------------------
def make_u_func(current_coarse, current_fine_list, w0_coarse, w0_fine_list, A_coarse, A_fine_list):
    current_coarse_copy = current_coarse.copy()
    def u_func(x):
        u_val = np.zeros(2)
        n1, n2, _ = current_coarse_copy.shape
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                term1 = indicator_D(pos) * K_Q(pos, x, delta)
                term2 = indicator_D(Rx(pos)) * K_Q(Rx(pos), x, delta)
                term3 = indicator_D(Ry(pos)) * K_Q(Ry(pos), x, delta)
                term4 = indicator_D(Rx(Ry(pos))) * K_Q(Rx(Ry(pos)), x, delta)
                u_val += (term1 - term2 - term3 + term4) * w0_coarse[i, j] * A_coarse[i, j]
        for grid_fine, w_f, A_f in zip(current_fine_list, w0_fine_list, A_fine_list):
            n_i, n_j, _ = grid_fine.shape
            for i in range(n_i):
                for j in range(n_j):
                    pos = grid_fine[i, j]
                    term1 = indicator_D(pos) * K_Q(pos, x, delta)
                    term2 = indicator_D(Rx(pos)) * K_Q(Rx(pos), x, delta)
                    term3 = indicator_D(Ry(pos)) * K_Q(Ry(pos), x, delta)
                    term4 = indicator_D(Rx(Ry(pos))) * K_Q(Rx(Ry(pos)), x, delta)
                    u_val += (term1 - term2 - term3 + term4) * w_f[i, j] * A_f[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama)
# ---------------------------
def simulate_vortex_trajectories():
    n1, n2, _ = grid_coarse.shape
    n_Dbxy1, n_Dbxy2, _ = grid_Dbxy.shape
    n_Dbx1, n_Dbx2, _ = grid_Dbx.shape
    n_Dby1, n_Dby2, _ = grid_Dby.shape
    
    traj_coarse = np.zeros((num_steps + 1, n1, n2, 2))
    traj_Dbxy = np.zeros((num_steps + 1, n_Dbxy1, n_Dbxy2, 2))
    traj_Dbx = np.zeros((num_steps + 1, n_Dbx1, n_Dbx2, 2))
    traj_Dby = np.zeros((num_steps + 1, n_Dby1, n_Dby2, 2))
    
    traj_coarse[0] = grid_coarse
    traj_Dbxy[0] = grid_Dbxy
    traj_Dbx[0] = grid_Dbx
    traj_Dby[0] = grid_Dby
    
    uFuncs = []
    # (Displacement arrays are not used later, but we compute them for consistency.)
    disp_coarse = np.zeros((num_steps, n1, n2, 2))
    disp_Dbxy = np.zeros((num_steps, n_Dbxy1, n_Dbxy2, 2))
    disp_Dbx = np.zeros((num_steps, n_Dbx1, n_Dbx2, 2))
    disp_Dby = np.zeros((num_steps, n_Dby1, n_Dby2, 2))
    
    for step in range(num_steps):
        current_coarse = traj_coarse[step]
        current_Dbxy = traj_Dbxy[step]
        current_Dbx = traj_Dbx[step]
        current_Dby = traj_Dby[step]
        u_func = make_u_func(current_coarse, [current_Dbxy, current_Dbx, current_Dby],
                             w0_coarse, [w0_Dbxy, w0_Dbx, w0_Dby],
                             A_coarse, [A_Dbxy, A_Dbx, A_Dby])
        uFuncs.append(u_func)
        # Update each grid using Euler-Maruyama:
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_coarse[step, i, j] = disp
                traj_coarse[step + 1, i, j] = current_coarse[i, j] + disp
        for i in range(n_Dbxy1):
            for j in range(n_Dbxy2):
                u_val = u_func(current_Dbxy[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_Dbxy[step, i, j] = disp
                traj_Dbxy[step + 1, i, j] = current_Dbxy[i, j] + disp
        for i in range(n_Dbx1):
            for j in range(n_Dbx2):
                u_val = u_func(current_Dbx[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_Dbx[step, i, j] = disp
                traj_Dbx[step + 1, i, j] = current_Dbx[i, j] + disp
        for i in range(n_Dby1):
            for j in range(n_Dby2):
                u_val = u_func(current_Dby[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_Dby[step, i, j] = disp
                traj_Dby[step + 1, i, j] = current_Dby[i, j] + disp
    return (traj_coarse, traj_Dbxy, traj_Dbx, traj_Dby), uFuncs, (disp_coarse, disp_Dbxy, disp_Dbx, disp_Dby)

print("Computing vortex trajectories......")
start_time = time.time()
(trajectories_coarse, trajectories_Dbxy, trajectories_Dbx, trajectories_Dby), uFuncs, _ = simulate_vortex_trajectories()
end_time = time.time()
print(f"Vortex trajectories computed in {end_time - start_time:.2f} seconds")

# ---------------------------
# Setup for Streamline Plotting
# ---------------------------
# Create a uniform grid for background velocity magnitude computation
num_bg = 100
x_bg = np.linspace(window_x[0], window_x[1], num_bg)
y_bg = np.linspace(window_y[0], window_y[1], num_bg)
X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
points_bg = np.stack([X_bg.flatten(), Y_bg.flatten()], axis=-1)

# Create a colormap from white (slow) to red (fast)
vel_cmap = LinearSegmentedColormap.from_list("vel_cmap", ["white", "red"])

# Compute global maximum velocity magnitude from the first u_func for fixed color scaling
u_func0 = uFuncs[0]
vel_bg = np.array([u_func0(p) for p in points_bg])
mag_bg = np.linalg.norm(vel_bg, axis=1).reshape(X_bg.shape)
global_max = mag_bg.max()

# Define a vectorized function to compute U,V on a grid (for streamline plotting)
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
    # Background: show velocity magnitude using the same colormap and fixed scale
    vel_bg = np.array([u_func(p) for p in points_bg])
    mag_bg = np.linalg.norm(vel_bg, axis=1).reshape(X_bg.shape)
    ax.imshow(mag_bg, extent=(window_x[0], window_x[1], window_y[0], window_y[1]),
              origin='lower', cmap=vel_cmap, alpha=0.5, vmin=0, vmax=global_max)
    
    # Region A: dense seeding
    xA = np.linspace(region_x[0], layer_thickness, 50)
    yA = np.linspace(region_y[0], layer_thickness, 50)
    XA, YA = np.meshgrid(xA, yA)
    U_A, V_A = compute_uv_on_grid(u_func, XA, YA)
    ax.streamplot(xA, yA, U_A, V_A, color='k', linewidth=1)

    # Region B: intermediate density
    xB = np.linspace(layer_thickness, region_x[1], 30)
    yB = np.linspace(region_y[0], layer_thickness, 30)
    XB, YB = np.meshgrid(xB, yB)
    U_B, V_B = compute_uv_on_grid(u_func, XB, YB)
    ax.streamplot(xB, yB, U_B, V_B, color='k', linewidth=1)
    
    # Region C: intermediate density
    xC = np.linspace(region_x[0], layer_thickness, 30)
    yC = np.linspace(layer_thickness, region_y[1], 30)
    XC, YC = np.meshgrid(xC, yC)
    U_C, V_C = compute_uv_on_grid(u_func, XC, YC)
    ax.streamplot(xC, yC, U_C, V_C, color='k', linewidth=1)
    
    # Coarse region: sparse seeding
    xCoarse = np.linspace(layer_thickness, region_x[1], 15)
    yCoarse = np.linspace(layer_thickness, region_y[1], 15)
    XCoarse, YCoarse = np.meshgrid(xCoarse, yCoarse)
    U_Coarse, V_Coarse = compute_uv_on_grid(u_func, XCoarse, YCoarse)
    ax.streamplot(xCoarse, yCoarse, U_Coarse, V_Coarse, color='k', linewidth=1)
    
    return fig, ax

# ---------------------------
# Plot and Save Streamline Images at Specified Times
# ---------------------------
# Times (in seconds) at which to save streamline images
save_times = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
save_frames = [int(t/dt) for t in save_times]

# Create folder for streamline images
output_folder = os.path.join("figure", "corner")
os.makedirs(output_folder, exist_ok=True)

for frame in save_frames:
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    fig, ax = plot_streamlines(u_func, t_current)
    filename = os.path.join(output_folder, f"streamlines_t{t_current:05.2f}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved streamline image at t={t_current:.2f} to {filename}")
