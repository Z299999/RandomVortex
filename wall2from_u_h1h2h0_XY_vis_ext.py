import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01           # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step (also used in viscosity term as h)
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (for kernel)
eps_bl = 0.1       # New constant for boundary layer (ε_bl)

VIS = False # viscosity
EXT = False # external force

# Mesh parameters:
h0 = 1.0    # Coarse mesh grid spacing (Do)
h1 = 0.5    # Fine mesh grid spacing in the x direction (Db)
h2 = 0.1    # Fine mesh grid spacing in the y direction (Db)

layer_thickness = 0.3 

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# Time array for use in viscosity term
time_array = np.arange(0, T + dt, dt)

# ---------------------------
# Vorticity, Velocity and Grid
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

def generate_nonuniform_grid_D():
    """
    Generates a nonuniform grid in D with a coarse grid covering y in [y3, y2]
    and a fine grid covering y in [y1, y3), where y3 is the layer_thickness.
    Returns a dictionary with keys 'coarse' (Do) and 'fine' (Db), where each value is a tuple (grid, A).
    """
    x1, x2 = region_x
    y1_, y2_ = region_y
    y3 = layer_thickness  # dividing line in y

    # Coarse grid (Do): y in [y3, y2] with spacing h0
    num_x_coarse = int((x2 - x1) / h0) + 1
    num_y_coarse = int((y2_ - y3) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2_, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))

    # Fine grid (Db): y in [y1, y3) with spacing h1 (x) and h2 (y)
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

# Generate grids
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse_arr = grids['coarse']  # Do
grid_fine, A_fine_arr = grids['fine']          # Db

# Initialize vortices
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
# Viscosity-related functions: theta and phi''
# ---------------------------
def theta_func(x1, t):
    return 1.0  # placeholder; equals vorticity at y=0, i.e., cos(0)=1

def phi_dd_func(r):
    if r >= 1/3 and r <= 2/3:
        return 324*(r - 0.5)
    else:
        return 0.0

# ---------------------------
# External Force Function G
# ---------------------------
def G_func(x, t):
    # External force's curl; for gravity F=(0,-g), curl F=0,
    # but we include the term in the algorithm.
    return 0.0  # placeholder; modify if a nonzero external force is desired

# ---------------------------
# Gamma update for fine grid (Db)
# ---------------------------
gamma_fine = -np.inf * np.ones(grid_fine.shape[:2])
def update_gamma_fine(current_fine, gamma_fine, t_current):
    n1, n2 = current_fine.shape[0], current_fine.shape[1]
    for i in range(n1):
        for j in range(n2):
            if current_fine[i,j][1] > 0:
                pass  # gamma remains unchanged
            else:
                gamma_fine[i,j] = t_current
    return gamma_fine

# ---------------------------
# Function Factories for u_func with External Force and Viscosity Terms (for X and Y)
# ---------------------------
def make_u_func_X(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                  w0_coarse, w0_fine, A_coarse, A_fine, delta, 
                  traj_X_fine_history, gamma_fine, time_array, k_index, eps_bl,
                  theta_func, phi_dd_func):
    """
    Build u_func for X that includes three contributions:
      1. Standard vortex-induced velocity (Biot--Savart).
      2. External force contribution.
      3. Viscosity correction for the fine grid.
    """
    def compute_u_X(x):
        u_val = np.zeros(2)
        # Coarse contribution (Do) - Standard
        n1, n2 = current_X_coarse.shape[0], current_X_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_X_coarse[i, j]
                pos_ref = reflect(current_Y_coarse[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        # Fine contribution (Db) - Standard
        n1f, n2f = current_X_fine.shape[0], current_X_fine.shape[1]
        for i in range(n1f):
            for j in range(n2f):
                pos = current_X_fine[i, j]
                pos_ref = reflect(current_Y_fine[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
        # External force contribution (both grids)
        if EXT:
            # Coarse grid external force (Do): no gamma check (always in D)
            for i in range(n1):
                for j in range(n2):
                    pos = current_X_coarse[i, j]
                    ext_sum = 0.0
                    for l in range(k_index + 1):
                        ext_sum += G_func(pos, time_array[l])
                    u_val += A_coarse[i, j] * dt * K_delta(pos, x, delta) * ext_sum
            # Fine grid external force (Db): with gamma check
            for i in range(n1f):
                for j in range(n2f):
                    pos = current_X_fine[i, j]
                    ext_sum = 0.0
                    for l in range(k_index + 1):
                        if time_array[l] > gamma_fine[i, j]:
                            ext_sum += G_func(pos, time_array[l])
                    u_val += A_fine[i, j] * dt * K_delta(pos, x, delta) * ext_sum

        if VIS:
            # Fine contribution (Db) - Viscosity term
            for i in range(n1f):
                for j in range(n2f):
                    inner_sum = 0.0
                    for l in range(k_index + 1):
                        if time_array[l] > gamma_fine[i, j]:
                            pos_l = traj_X_fine_history[l][i, j]
                            inner_sum += theta_func(pos_l[0], time_array[l]) * phi_dd_func(pos_l[1] / eps_bl)
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
                  traj_Y_fine_history, gamma_fine, time_array, k_index, eps_bl,
                  theta_func, phi_dd_func):
    """
    Build u_func for Y that includes:
      1. Standard vortex-induced velocity.
      2. External force contribution.
      3. Viscosity correction for the fine grid.
    """
    def compute_u_Y(x):
        u_val = np.zeros(2)
        # Coarse contribution (Do) - Standard
        n1, n2 = current_Y_coarse.shape[0], current_Y_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_Y_coarse[i, j]
                pos_ref = reflect(current_X_coarse[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        # Fine contribution (Db) - Standard
        n1f, n2f = current_Y_fine.shape[0], current_Y_fine.shape[1]
        for i in range(n1f):
            for j in range(n2f):
                pos = current_Y_fine[i, j]
                pos_ref = reflect(current_X_fine[i, j])
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(pos_ref) * K_delta(pos_ref, x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
        # External force contribution (both grids)
        if EXT:
            # Coarse grid external force (Do)
            for i in range(n1):
                for j in range(n2):
                    pos = current_Y_coarse[i, j]
                    ext_sum = 0.0
                    for l in range(k_index + 1):
                        ext_sum += G_func(pos, time_array[l])
                    u_val += A_coarse[i, j] * dt * K_delta(pos, x, delta) * ext_sum
            # Fine grid external force (Db)
            for i in range(n1f):
                for j in range(n2f):
                    pos = current_Y_fine[i, j]
                    ext_sum = 0.0
                    for l in range(k_index + 1):
                        if time_array[l] > gamma_fine[i, j]:
                            ext_sum += G_func(pos, time_array[l])
                    u_val += A_fine[i, j] * dt * K_delta(pos, x, delta) * ext_sum
        # Fine contribution (Db) - Viscosity term
        if VIS:
            for i in range(n1f):
                for j in range(n2f):
                    inner_sum = 0.0
                    for l in range(k_index + 1):
                        if time_array[l] > gamma_fine[i, j]:
                            pos_l = traj_Y_fine_history[l][i, j]
                            inner_sum += theta_func(pos_l[0], time_array[l]) * phi_dd_func(pos_l[1] / eps_bl)
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
# Vortex Trajectory Simulation for X and Y
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
                                 traj_X_fine, gamma_fine, time_array, step, eps_bl, theta_func, phi_dd_func)
        u_func_Y = make_u_func_Y(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                                 w0_coarse, w0_fine, A_coarse_arr, A_fine_arr, delta,
                                 traj_Y_fine, gamma_fine, time_array, step, eps_bl, theta_func, phi_dd_func)
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
                
    total_vortex_time = time.time() - start_time
    print(f"Vortex trajectory simulation time: {total_vortex_time:.2f} seconds")
    
    start_boat_time = time.time()
    
    def generate_boat_grid():
        # Only use coarse (Do) and fine (Db) from X trajectories
        boat_grid = np.concatenate((grid_coarse.reshape(-1,2), grid_fine.reshape(-1,2)), axis=0)
        return boat_grid

    def simulate_boats(uFuncs):
        # Simulate only X boats (passive tracers)
        boat_grid = generate_boat_grid()
        num_boats = boat_grid.shape[0]
        boat_positions = np.zeros((num_steps+1, num_boats, 2))
        boat_positions[0] = boat_grid
        for step in range(num_steps):
            u_func = uFuncs[step]
            new_positions = np.zeros_like(boat_grid)
            for b in range(num_boats):
                vel = u_func(boat_positions[step, b])
                new_positions[b] = boat_positions[step, b] + dt * vel
            boat_positions[step+1] = new_positions
        return boat_positions

    print("Simulating boat trajectories for X (using uFuncs_X)...")
    boat_positions_X = simulate_boats(uFuncs_X)
    
    total_boat_time = time.time() - start_boat_time
    print(f"Boat simulation time: {total_boat_time:.2f} seconds")
    
    return (traj_X_coarse, traj_X_fine), uFuncs_X, (traj_Y_coarse, traj_Y_fine), uFuncs_Y, boat_positions_X

print("Simulating vortex trajectories for X and Y ...")
(traj_X_coarse, traj_X_fine), uFuncs_X, (traj_Y_coarse, traj_Y_fine), uFuncs_Y, boat_positions_X = simulate_vortex_trajectories_XY()

# ---------------------------
# Velocity Field Query for Visualization
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

def generate_query_grid():
    return np.concatenate((grid_coarse.reshape(-1,2), grid_fine.reshape(-1,2)), axis=0)

query_grid = generate_query_grid()

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation for X
# ---------------------------
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Adjust quiver parameters to get larger arrows: scale=5 and width=0.01.
U, V = compute_velocity_field(uFuncs_X[0], query_grid)
vel_quiver = ax.quiver(query_grid[:,0], query_grid[:,1], U, V,
                       color='black', alpha=0.9, pivot='mid', scale=5, width=0.01, angles='xy', scale_units='xy')
boat_scatter = ax.scatter(boat_positions_X[0][:,0], boat_positions_X[0][:,1],
                          s=10, color='blue', zorder=3)

start_anim_time = time.time()

def update(frame):
    t_current = frame * dt
    u_func = uFuncs_X[frame] if frame < len(uFuncs_X) else uFuncs_X[-1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    boat_scatter.set_offsets(boat_positions_X[frame])
    ax.set_title(f"Vortex and Boat Animation X (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex_XY_vis_ext.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
total_anim_time = time.time() - start_anim_time
print(f"Animation computation time: {total_anim_time:.2f} seconds")
print(f"Animation saved at: {save_path}")

plt.close(fig)
