import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01           # viscosity
T = 20              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter for kernel K_δ
eps_bl = 0.1        # Boundary layer parameter (ε_bl)

# Flags
VIS = False   # use viscosity term
EXT = False   # external force turned off

# Mesh parameters:
h0 = 0.8    # Coarse grid spacing (Do)
h1 = 0.4    # Fine grid spacing in x (Db)
h2 = 0.2    # Fine grid spacing in y (Db)
layer_thickness = 0.4 

# h0 = 0.8    # Coarse grid spacing (Do)
# h1 = 0.4    # Fine grid spacing in x (Db)
# h2 = 0.2    # Fine grid spacing in y (Db)
# layer_thickness = 0.4 

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)
time_array = np.arange(0, T + dt, dt)

# ---------------------------
# Basic Vorticity and Velocity
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

# ---------------------------
# Grid Generation (Nonuniform in y)
# ---------------------------
def generate_nonuniform_grid_D():
    x1, x2 = region_x
    y1_, y2_ = region_y
    y3 = layer_thickness

    # Coarse grid (Do)
    num_x_coarse = int((x2 - x1) / h0) + 1
    num_y_coarse = int((y2_ - y3) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2_, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))

    # Fine grid (Db)
    num_x_fine = int((x2 - x1) / h1) + 1
    num_y_fine = int((y3 - y1_) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y1_, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = np.stack((xx_fine, yy_fine), axis=-1)
    A_fine = h1 * h2 * np.ones((num_x_fine, num_y_fine))

    print(f"Coarse grid shape: {grid_coarse.shape}, points: {grid_coarse.size//2}")
    print(f"Fine grid shape: {grid_fine.shape}, points: {grid_fine.size//2}")
    return {'coarse': (grid_coarse, A_coarse), 'fine': (grid_fine, A_fine)}

grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse_arr = grids['coarse']
grid_fine, A_fine_arr = grids['fine']

# Optionally, if you previously plotted the mesh grid points for debugging, you can comment this out.
# plt.figure(figsize=(6,6))
# plt.scatter(grid_coarse[:,:,0].ravel(), grid_coarse[:,:,1].ravel(), s=50, marker='o', color='green', label='Coarse Mesh')
# plt.scatter(grid_fine[:,:,0].ravel(), grid_fine[:,:,1].ravel(), s=30, marker='x', color='purple', label='Fine Mesh')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Mesh Grid Points')
# plt.legend()
# plt.grid(True)
# plt.show()

# ---------------------------
# Initialize Vortices on the Grids
# ---------------------------
def initialize_vortices(grid):
    n_x, n_y, _ = grid.shape
    w = np.zeros((n_x, n_y))
    u = np.zeros((n_x, n_y, 2))
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y)
            u[i, j] = velocity(x, y)
    return w, u

w0_coarse, u0_coarse = initialize_vortices(grid_coarse)
w0_fine, u0_fine = initialize_vortices(grid_fine)
print("* Number of vortices in coarse grid:", grid_coarse.shape[0]*grid_coarse.shape[1])
print("* Number of vortices in fine grid:", grid_fine.shape[0]*grid_fine.shape[1])
num_vortices = grid_coarse.shape[0]*grid_coarse.shape[1] + grid_fine.shape[0]*grid_fine.shape[1]

# ---------------------------
# Mollified Biot-Savart Kernel and Helpers
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
    factor = 1 - np.exp(- (r2 / delta)**2)
    return np.array([k1, k2]) * factor

def indicator_D(point):
    return 0 if point[1] <= 0 else 1

def reflect(point):
    return np.array([point[0], -point[1]])

# ---------------------------
# Viscosity-related Functions
# ---------------------------
def phi_dd_func(r):
    if r >= 1/3 and r <= 2/3:
        return 324 * (r - 0.5)
    else:
        return 0.0

Theta_history_fine = []
Theta_history_fine.append(np.ones(grid_fine.shape[:2]))

def compute_phi(u_func, x1, h_diff=1e-3):
    f_h = u_func(np.array([x1, h_diff]))[0]
    f_2h = u_func(np.array([x1, 2 * h_diff]))[0]
    f_3h = u_func(np.array([x1, 3 * h_diff]))[0]
    third_deriv = (-f_3h + 3 * f_2h - 3 * f_h) / (h_diff**3)
    return nu * third_deriv

# ---------------------------
# Gamma update for the Fine Grid (Db)
# ---------------------------
gamma_fine = -np.inf * np.ones(grid_fine.shape[:2])
def update_gamma_fine(current_fine, gamma_fine, t_current):
    n1, n2 = current_fine.shape[0], current_fine.shape[1]
    for i in range(n1):
        for j in range(n2):
            if current_fine[i, j][1] > 0:
                pass  
            else:
                gamma_fine[i, j] = t_current
    return gamma_fine

# ---------------------------
# Modified u_func Factories
# ---------------------------
def make_u_func_X(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                  w0_coarse, w0_fine, A_coarse, A_fine, delta, 
                  traj_X_fine_history, gamma_fine, time_array, k_index, eps_bl):
    def compute_u_X(x):
        u_val = np.zeros(2)
        n1, n2 = current_X_coarse.shape[0], current_X_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_X_coarse[i, j]
                pos_ref = reflect(current_Y_coarse[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        n1f, n2f = current_X_fine.shape[0], current_X_fine.shape[1]
        for i in range(n1f):
            for j in range(n2f):
                pos = current_X_fine[i, j]
                pos_ref = reflect(current_Y_fine[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
        if VIS:
            for i in range(n1f):
                for j in range(n2f):
                    inner_sum = 0.0
                    for l in range(k_index + 1):
                        if time_array[l] > gamma_fine[i, j]:
                            pos_l = traj_X_fine_history[l][i, j]
                            inner_sum += Theta_history_fine[l][i, j] * phi_dd_func(pos_l[1] / eps_bl)
                    u_val += (nu / (eps_bl**2)) * h1 * h2 * dt * K_delta(current_X_fine[i, j], x, delta) * inner_sum
        return u_val

    def u_func_X(x):
        if x[1] > 0:
            return compute_u_X(x)
        elif np.isclose(x[1], 0.0):
            return np.zeros(2)
        else:
            x_ref = reflect(x)
            u_val_upper = compute_u_X(x_ref)
            return reflect(u_val_upper)
    return u_func_X

def make_u_func_Y(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                  w0_coarse, w0_fine, A_coarse, A_fine, delta,
                  traj_Y_fine_history, gamma_fine, time_array, k_index, eps_bl):
    def compute_u_Y(x):
        u_val = np.zeros(2)
        n1, n2 = current_Y_coarse.shape[0], current_Y_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_Y_coarse[i, j]
                pos_ref = reflect(current_X_coarse[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        n1f, n2f = current_Y_fine.shape[0], current_Y_fine.shape[1]
        for i in range(n1f):
            for j in range(n2f):
                pos = current_Y_fine[i, j]
                pos_ref = reflect(current_X_fine[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
        if VIS:
            for i in range(n1f):
                for j in range(n2f):
                    inner_sum = 0.0
                    for l in range(k_index + 1):
                        if time_array[l] > gamma_fine[i, j]:
                            pos_l = traj_Y_fine_history[l][i, j]
                            inner_sum += Theta_history_fine[l][i, j] * phi_dd_func(pos_l[1] / eps_bl)
                    u_val += (nu / (eps_bl**2)) * h1 * h2 * dt * K_delta(current_Y_fine[i, j], x, delta) * inner_sum
        return u_val

    def u_func_Y(x):
        if x[1] > 0:
            return compute_u_Y(x)
        elif np.isclose(x[1], 0.0):
            return np.zeros(2)
        else:
            x_ref = reflect(x)
            u_val_upper = compute_u_Y(x_ref)
            return reflect(u_val_upper)
    return u_func_Y

# ---------------------------
# Vortex Trajectory Simulation and Theta Update
# ---------------------------
def simulate_vortex_trajectories_XY():
    traj_X_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_X_fine   = np.zeros((num_steps + 1, grid_fine.shape[0], grid_fine.shape[1], 2))
    traj_Y_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_Y_fine   = np.zeros((num_steps + 1, grid_fine.shape[0], grid_fine.shape[1], 2))
    
    traj_X_coarse[0] = grid_coarse
    traj_X_fine[0]   = grid_fine
    traj_Y_coarse[0] = grid_coarse.copy()
    traj_Y_fine[0]   = grid_fine.copy()
    
    uFuncs_X = []
    uFuncs_Y = []
    
    n1, n2 = grid_coarse.shape[0], grid_coarse.shape[1]
    n1f, n2f = grid_fine.shape[0], grid_fine.shape[1]
    
    start_time = time.time()
    
    for step in range(num_steps):
        t_current = step * dt
        current_X_coarse = traj_X_coarse[step].copy()
        current_X_fine   = traj_X_fine[step].copy()
        current_Y_coarse = traj_Y_coarse[step].copy()
        current_Y_fine   = traj_Y_fine[step].copy()
        
        gamma_fine[:] = update_gamma_fine(current_X_fine, gamma_fine, t_current)
        
        u_func_X = make_u_func_X(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                                 w0_coarse, w0_fine, A_coarse_arr, A_fine_arr, delta,
                                 traj_X_fine, gamma_fine, time_array, step, eps_bl)
        u_func_Y = make_u_func_Y(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                                 w0_coarse, w0_fine, A_coarse_arr, A_fine_arr, delta,
                                 traj_Y_fine, gamma_fine, time_array, step, eps_bl)
        uFuncs_X.append(u_func_X)
        uFuncs_Y.append(u_func_Y)
        
        for i in range(n1):
            for j in range(n2):
                u_val = u_func_X(current_X_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_X_coarse[step+1, i, j] = current_X_coarse[i, j] + dt * u_val + dW
        for i in range(n1f):
            for j in range(n2f):
                u_val = u_func_X(current_X_fine[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_X_fine[step+1, i, j] = current_X_fine[i, j] + dt * u_val + dW
        
        for i in range(n1):
            for j in range(n2):
                u_val = u_func_Y(current_Y_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Y_coarse[step+1, i, j] = current_Y_coarse[i, j] + dt * u_val + dW
        for i in range(n1f):
            for j in range(n2f):
                u_val = u_func_Y(current_Y_fine[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Y_fine[step+1, i, j] = current_Y_fine[i, j] + dt * u_val + dW
                
        t_new = (step + 1) * dt
        theta_new = np.copy(Theta_history_fine[-1])
        MC_samples = 10
        h_diff = 1e-3
        for i in range(n1f):
            for j in range(n2f):
                x1 = traj_X_fine[step+1, i, j, 0]
                sigma = np.sqrt(4 * nu * t_new) if t_new > 0 else 0.0
                if sigma > 0:
                    samples = np.random.normal(loc=x1, scale=sigma, size=MC_samples)
                    phi_vals = [compute_phi(u_func_X, s, h_diff) for s in samples]
                    MC_phi = np.mean(phi_vals)
                else:
                    MC_phi = 0.0
                theta_new[i, j] = Theta_history_fine[-1][i, j] - dt * MC_phi
        Theta_history_fine.append(theta_new)
                
    total_vortex_time = time.time() - start_time
    print(f"Vortex trajectory simulation time: {total_vortex_time:.2f} seconds")
    
    return (traj_X_coarse, traj_X_fine), uFuncs_X, (traj_Y_coarse, traj_Y_fine), uFuncs_Y

print("Simulating vortex trajectories for X and Y ...")
(traj_X_coarse, traj_X_fine), uFuncs_X, (traj_Y_coarse, traj_Y_fine), uFuncs_Y = simulate_vortex_trajectories_XY()

# ---------------------------
# Precompute Background Velocity Magnitude Fields and Velocity Grids
# ---------------------------
num_bg_x = 50
num_bg_y = 50
x_bg = np.linspace(region_x[0], region_x[1], num_bg_x)
y_bg = np.linspace(region_y[0], region_y[1], num_bg_y)
X_bg, Y_bg = np.meshgrid(x_bg, y_bg)

bg_fields = []    # List for velocity magnitude fields
vel_fields = []   # List for (U_grid, V_grid) tuples
global_min = np.inf
global_max = -np.inf
for frame in range(num_steps+1):
    u_func = uFuncs_X[frame] if frame < len(uFuncs_X) else uFuncs_X[-1]
    U_grid = np.zeros((num_bg_y, num_bg_x))
    V_grid = np.zeros((num_bg_y, num_bg_x))
    vel_mag = np.zeros((num_bg_y, num_bg_x))
    for i in range(num_bg_y):
        for j in range(num_bg_x):
            pt = np.array([x_bg[j], y_bg[i]])
            vel = u_func(pt)
            U_grid[i, j] = vel[0]
            V_grid[i, j] = vel[1]
            vel_mag[i, j] = np.sqrt(vel[0]**2 + vel[1]**2)
    bg_fields.append(vel_mag)
    vel_fields.append((U_grid, V_grid))
    global_min = min(global_min, vel_mag.min())
    global_max = max(global_max, vel_mag.max())
print(f"Background velocity magnitude range: min={global_min:.4f}, max={global_max:.4f}")

# ---------------------------
# Create a Custom Colormap (from white to red)
# ---------------------------
cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

# Folder to save static streamline images (updated output path)
output_folder = os.path.join("figure", "one_wall")
os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# Streamline Images at Specified Times
# ---------------------------
save_times = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
save_frames = [int(t/dt) for t in save_times]

# Define a vectorized function to compute U, V on a grid
def compute_uv_on_grid(u_func, X, Y):
    vec_u = np.vectorize(lambda x, y: u_func(np.array([x, y]))[0])
    vec_v = np.vectorize(lambda x, y: u_func(np.array([x, y]))[1])
    U = vec_u(X, Y)
    V = vec_v(X, Y)
    return U, V

# Define a function to plot streamlines at a given time t_current
def plot_streamlines(u_func, t_current):
    """
    Draw streamlines of the velocity field at time `t_current`
    and save the figure.  Only the bottom spine (y = 0) is kept,
    because there is a single physical wall on that boundary.
    """
    # ------------------------------------------------------------------
    # Figure and axes setup
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(window_x[0], window_x[1])
    ax.set_ylim(window_y[0], window_y[1])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"Streamlines at t = {t_current:.2f}")

    # Keep only the bottom spine (the wall); hide the others
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)          # Thickness of wall line
    ax.spines["bottom"].set_position(("data", 0.0)) # Align exactly at y = 0

    # ------------------------------------------------------------------
    # Background colour map: velocity magnitude |u|
    # ------------------------------------------------------------------
    points_bg = np.column_stack((X_bg.ravel(), Y_bg.ravel()))
    vel_bg    = np.array([u_func(p) for p in points_bg])
    mag_bg    = np.linalg.norm(vel_bg, axis=1).reshape(X_bg.shape)

    ax.imshow(
        mag_bg,
        extent=(window_x[0], window_x[1], window_y[0], window_y[1]),
        origin="lower",
        cmap=cmap,
        alpha=0.5,
        vmin=0.0,
        vmax=global_max,
    )

    # ------------------------------------------------------------------
    # Streamlines with four different seeding densities
    # ------------------------------------------------------------------
    # Region A: high density near the wall (lower‑left)
    xA = np.linspace(region_x[0], layer_thickness, 50)
    yA = np.linspace(region_y[0], layer_thickness, 50)
    XA, YA = np.meshgrid(xA, yA)
    U_A, V_A = compute_uv_on_grid(u_func, XA, YA)
    ax.streamplot(xA, yA, U_A, V_A, color="k", linewidth=1)

    # Region B: medium density near the wall (lower‑right)
    xB = np.linspace(layer_thickness, region_x[1], 30)
    yB = np.linspace(region_y[0], layer_thickness, 30)
    XB, YB = np.meshgrid(xB, yB)
    U_B, V_B = compute_uv_on_grid(u_func, XB, YB)
    ax.streamplot(xB, yB, U_B, V_B, color="k", linewidth=1)

    # Region C: medium density in the interior (upper‑left)
    xC = np.linspace(region_x[0], layer_thickness, 30)
    yC = np.linspace(layer_thickness, region_y[1], 30)
    XC, YC = np.meshgrid(xC, yC)
    U_C, V_C = compute_uv_on_grid(u_func, XC, YC)
    ax.streamplot(xC, yC, U_C, V_C, color="k", linewidth=1)

    # Coarse region: sparse seeds in the interior (upper‑right)
    xCoarse = np.linspace(layer_thickness, region_x[1], 15)
    yCoarse = np.linspace(layer_thickness, region_y[1], 15)
    XCoarse, YCoarse = np.meshgrid(xCoarse, yCoarse)
    U_Coarse, V_Coarse = compute_uv_on_grid(u_func, XCoarse, YCoarse)
    ax.streamplot(xCoarse, yCoarse, U_Coarse, V_Coarse, color="k", linewidth=1)

    # ------------------------------------------------------------------
    # Final layout adjustments
    # ------------------------------------------------------------------
    fig.tight_layout(pad=0)
    return fig, ax


# Loop over each specified frame and save streamline image in SVG format
for frame in save_frames:
    t_current = frame * dt
    u_func = uFuncs_X[frame] if frame < len(uFuncs_X) else uFuncs_X[-1]
    fig, ax = plot_streamlines(u_func, t_current)
    filename = os.path.join(output_folder, f"streamlines_o_t{t_current:05.2f}.svg")
    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved streamline image at t={t_current:.2f} to {filename}")

