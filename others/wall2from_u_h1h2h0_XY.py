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

# Mesh parameters:
h0 = 1.0    # Coarse mesh grid spacing (both x and y)
h1 = 0.5    # Fine mesh grid spacing in the x direction
h2 = 0.1    # Fine mesh grid spacing in the y direction

Re = 0.0001 / nu
layer_thickness = 1 * np.sqrt(Re)  # dividing line between fine and coarse grids

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Vorticity, Velocity and Grid (Last Week’s Code)
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

def generate_nonuniform_grid_D():
    """
    Generates a nonuniform grid in D with a coarse grid covering y in [y3, y2] 
    and a fine grid covering y in [y1, y3), where y3 is the layer_thickness.
    Returns a dictionary with keys 'coarse' and 'fine' where each value is a tuple (grid, A).
    """
    x1, x2 = region_x
    y1_, y2_ = region_y
    y3 = layer_thickness  # dividing line in y
    
    # Coarse grid: y in [y3, y2] with spacing h0 (uniform in both x and y)
    num_x_coarse = int((x2 - x1) / h0) + 1
    num_y_coarse = int((y2_ - y3) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2_, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))
    
    # Fine grid: y in [y1, y3) with spacing h1 (x) and h2 (y)
    num_x_fine = int((x2 - x1) / h1) + 1
    num_y_fine = int((y3 - y1_) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y1_, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = np.stack((xx_fine, yy_fine), axis=-1)
    A_fine = h1 * h2 * np.ones((num_x_fine, num_y_fine))
    
    print(f"Coarse grid shape: {grid_coarse.shape}, points: {grid_coarse.shape[0]*grid_coarse.shape[1]}")
    print(f"Fine grid shape: {grid_fine.shape}, points: {grid_fine.shape[0]*grid_fine.shape[1]}")
    
    return {'coarse': (grid_coarse, A_coarse), 'fine': (grid_fine, A_fine)}

# Generate grids (for vortex initialization)
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse_arr = grids['coarse']
grid_fine, A_fine_arr = grids['fine']

# Initialize vortices (double-indexed)
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
# Function Factories for u_func (for X and for Y)
# ---------------------------
# For clarity, we assume the vorticity and area arrays remain the same for both X and Y.
# We will use grid_coarse and grid_fine as the base (they are identical for X and Y initially).

def make_u_func_X(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                  w0_coarse, w0_fine, A_coarse, A_fine, delta):
    """
    For updating X: we use 
      contrib1 = indicator_D(pos_x)*K_delta(pos_x, x, delta)
      contrib2 = indicator_D(reflect(pos_y))*K_delta(reflect(pos_y), x, delta)
    with pos_x from X and pos_y from Y.
    """
    def compute_u_X(x):
        u_val = np.zeros(2)
        # Coarse contribution:
        n1, n2 = current_X_coarse.shape[0], current_X_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos_x = current_X_coarse[i, j]
                pos_y = current_Y_coarse[i, j]
                contrib1 = indicator_D(        pos_x ) * K_delta(        pos_x,  x, delta)
                contrib2 = indicator_D(reflect(pos_y)) * K_delta(reflect(pos_y), x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        # Fine contribution:
        n1f, n2f = current_X_fine.shape[0], current_X_fine.shape[1]
        for i in range(n1f):
            for j in range(n2f):
                pos_x = current_X_fine[i, j]
                pos_y = current_Y_fine[i, j]
                contrib1 = indicator_D(        pos_x ) * K_delta(        pos_x,  x, delta)
                contrib2 = indicator_D(reflect(pos_y)) * K_delta(reflect(pos_y), x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
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
                  w0_coarse, w0_fine, A_coarse, A_fine, delta):
    """
    For updating Y: we use 
      contrib1 = indicator_D(pos_y)*K_delta(pos_y, x, delta)
      contrib2 = indicator_D(reflect(pos_x))*K_delta(reflect(pos_x), x, delta)
    with pos_y from Y and pos_x from X.
    """
    def compute_u_Y(x):
        u_val = np.zeros(2)
        # Coarse contribution:
        n1, n2 = current_Y_coarse.shape[0], current_Y_coarse.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos_y = current_Y_coarse[i, j]
                pos_x = current_X_coarse[i, j]
                contrib1 = indicator_D(        pos_y ) * K_delta(        pos_y , x, delta)
                contrib2 = indicator_D(reflect(pos_x)) * K_delta(reflect(pos_x), x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        # Fine contribution:
        n1f, n2f = current_Y_fine.shape[0], current_Y_fine.shape[1]
        for i in range(n1f):
            for j in range(n2f):
                pos_y = current_Y_fine[i, j]
                pos_x = current_X_fine[i, j]
                contrib1 = indicator_D(        pos_y ) * K_delta(        pos_y , x, delta)
                contrib2 = indicator_D(reflect(pos_x)) * K_delta(reflect(pos_x), x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
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
# Vortex Trajectory Simulation for X and Y (using double-index for positions)
# ---------------------------
def simulate_vortex_trajectories_XY():
    """
    Simulates vortex trajectories using the Euler-Maruyama scheme.
    We simulate two independent sets: X and Y.
    They share the same initial conditions (grid_coarse and grid_fine) but are updated with independent Brownian motions.
    When computing the velocity field:
      - For updating X, use:
            contrib1 = indicator_D(pos_x)*K_delta(pos_x, x, delta)
            contrib2 = indicator_D(reflect(pos_y))*K_delta(reflect(pos_y), x, delta)
      - For updating Y, use:
            contrib1 = indicator_D(pos_y)*K_delta(pos_y, x, delta)
            contrib2 = indicator_D(reflect(pos_x))*K_delta(reflect(pos_x), x, delta)
    Returns:
      (traj_X_coarse, traj_X_fine), uFuncs_X and
      (traj_Y_coarse, traj_Y_fine), uFuncs_Y.
    """
    # Initialize trajectory arrays for coarse and fine grids for X and Y
    traj_X_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_X_fine   = np.zeros((num_steps + 1, grid_fine.shape[0]  , grid_fine.shape[1]  , 2))
    traj_Y_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_Y_fine   = np.zeros((num_steps + 1, grid_fine.shape[0]  , grid_fine.shape[1]  , 2))
    
    traj_X_coarse[0] = grid_coarse
    traj_X_fine[0]   = grid_fine
    traj_Y_coarse[0] = grid_coarse.copy()  # same initial condition
    traj_Y_fine[0]   = grid_fine.copy()
    
    uFuncs_X = []
    uFuncs_Y = []
    
    n1,  n2  = grid_coarse.shape[0], grid_coarse.shape[1]
    n1f, n2f = grid_fine.shape[0], grid_fine.shape[1]
    
    # For each time step, compute independent Brownian increments for X and Y.
    for step in range(num_steps):
        # Current positions:
        current_X_coarse = traj_X_coarse[step].copy()
        current_X_fine   = traj_X_fine[step].copy()
        current_Y_coarse = traj_Y_coarse[step].copy()
        current_Y_fine   = traj_Y_fine[step].copy()
        
        # Create u_func_X and u_func_Y using the cross-coupling:
        u_func_X = make_u_func_X(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                                 w0_coarse, w0_fine, A_coarse_arr, A_fine_arr, delta)
        u_func_Y = make_u_func_Y(current_X_coarse, current_X_fine, current_Y_coarse, current_Y_fine,
                                 w0_coarse, w0_fine, A_coarse_arr, A_fine_arr, delta)
        uFuncs_X.append(u_func_X)
        uFuncs_Y.append(u_func_Y)
        
        # Update trajectories for X:
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
        
        # Update trajectories for Y:
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
                
    return (traj_X_coarse, traj_X_fine), uFuncs_X, (traj_Y_coarse, traj_Y_fine), uFuncs_Y

print("Simulating vortex trajectories for X and Y ...")
(traj_X_coarse, traj_X_fine), uFuncs_X, (traj_Y_coarse, traj_Y_fine), uFuncs_Y = simulate_vortex_trajectories_XY()

# ---------------------------
# Boat Simulation (using velocity functions from vortex simulation)
# ---------------------------
def generate_boat_grid():
    """
    Generate a boat grid by concatenating the coarse and fine grids (flattened).
    """
    boat_grid = np.concatenate((grid_coarse.reshape(-1,2), grid_fine.reshape(-1,2)), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    """
    Simulate boat trajectories on the grid using drift only (no noise).
    Returns:
      boat_positions: array of shape (num_steps+1, num_boats, 2)
    """
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
print("Simulating boat trajectories for Y (using uFuncs_Y)...")
boat_positions_Y = simulate_boats(uFuncs_Y)

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
    return generate_boat_grid()

query_grid = generate_query_grid()

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation for X and Y
# ---------------------------
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# For demonstration, we display the velocity field from uFuncs_X and boat positions from boat_positions_X.
U, V = compute_velocity_field(uFuncs_X[0], query_grid)
vel_quiver = ax.quiver(query_grid[:,0], query_grid[:,1], U, V,
                       color='black', alpha=0.9, pivot='mid', scale=None, angles='xy', scale_units='xy')
boat_scatter = ax.scatter(boat_positions_X[0][:,0], boat_positions_X[0][:,1],
                          s=10, color='blue', zorder=3)

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
save_path = os.path.join("animation", "vortex_XY.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
