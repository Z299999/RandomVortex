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

# Here h1 is mesh length in x (coarse) and h2 is mesh length in y (finer)
h1 = 1              # x-direction mesh size for coarse grid
Re = 0.0001 / nu
layer_thickness = 1 * np.sqrt(Re)
h2_0 = layer_thickness * 0.5  # desired finer layer thickness in y
h2 = h2_0           # we now use h2_0 as the y-direction (finer) mesh size

print("boundary layer thickness:", layer_thickness)
print("mesh grid: h1 =", h1, ", h2 =", h2)

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Vorticity, Velocity and Grid
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return -2 * np.cos(y)

def generate_nonuniform_grid_D():
    """
    Generates a nonuniform grid in D with a coarse grid over most of the domain 
    and a finer grid near the boundary (y=0). The grids are returned as 2D arrays
    so that the vortex particle positions can be indexed by (i1, i2).
    """
    x1, x2 = region_x
    y1, y2 = region_y
    y3 = layer_thickness

    # Coarse grid: y in [y3, y2] with spacing h1 (assumed uniform in x and y for coarse region)
    num_x_coarse = int((x2 - x1) / h1) + 1
    num_y_coarse = int((y2 - y3) / h1)
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)  # shape (num_x_coarse, num_y_coarse, 2)
    A_coarse = h1 * h1 * np.ones((num_x_coarse, num_y_coarse))
    
    # Fine grid: y in [y1, y3] with spacing h2 in y-direction (and h1 in x-direction remains)
    num_x_fine = int((x2 - x1) / h1) + 1  # x-spacing remains h1
    num_y_fine = int((y3 - y1) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y1, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = np.stack((xx_fine, yy_fine), axis=-1)  # shape (num_x_fine, num_y_fine, 2)
    A_fine = h1 * h2 * np.ones((num_x_fine, num_y_fine))  # areas computed with h1 in x and h2 in y
    
    print(f"Coarse grid shape: {grid_coarse.shape}, number of points: {grid_coarse.shape[0]*grid_coarse.shape[1]}")
    print(f"Fine grid shape: {grid_fine.shape}, number of points: {grid_fine.shape[0]*grid_fine.shape[1]}")
    
    return {'coarse': (grid_coarse, A_coarse), 'fine': (grid_fine, A_fine)}

# Generate the grids (for vortex initialization)
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_fine, A_fine = grids['fine']

# Initialize vortex particles on each grid (using double indices)
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
    r2     = (x1 - y1)**2 + (x2 - y2)**2
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
# Modified Function Factory for u_func with Viscosity Term
# ---------------------------
def make_u_func(current_coarse, current_fine, w0_coarse, w0_fine, A_coarse, A_fine,
                traj_fine_history=None, uFuncs_history=None, current_step=None):
    """
    Creates a u_func that computes the local velocity at any position x.
    In addition to the inviscid contribution (Biot-Savart sums over coarse and fine grids),
    the function subtracts the viscous term computed via a time–integrated correction.
    
    Parameters:
      current_coarse, current_fine: current vortex positions (2D arrays)
      w0_coarse, w0_fine, A_coarse, A_fine: initial weights and areas.
      traj_fine_history: (optional) trajectory history of the fine grid, a numpy array
                         of shape (m, n_x_fine, n_y_fine, 2) for m past time steps.
      uFuncs_history: (optional) list of u_func functions computed for previous time steps.
      current_step: (optional) integer index for the current time step.
      
    Returns:
      u_func: a function that takes a point x (in R^2) and returns the computed velocity.
    """
    current_coarse_copy = current_coarse.copy()
    current_fine_copy = current_fine.copy()
    def u_func(x):
        u_val = np.zeros(2)
        # Inviscid contribution from the coarse grid
        n1, n2 = current_coarse_copy.shape[0], current_coarse_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        # Inviscid contribution from the fine grid
        n1_f, n2_f = current_fine_copy.shape[0], current_fine_copy.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                pos = current_fine_copy[i, j]
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
                
        # Add viscous term if history is provided (i.e. after the first step)
        if (traj_fine_history is not None) and (uFuncs_history is not None) and (current_step is not None):
            visc_term = np.zeros(2)
            # Here the number of horizontal points is given by the first dimension of the fine grid
            n1_fine_hist = traj_fine_history.shape[1]
            # Loop over horizontal indices and past time steps l = 0,..., current_step.
            # Note: In the paper the factor is (h1/h2) * dt, but near the boundary both mesh sizes are taken as h2.
            # Thus the factor becomes 1*dt.
            for i in range(n1_fine_hist):
                for l in range(current_step + 1):
                    # In our fine grid, row 0 corresponds to y=0 and row 1 corresponds to y=h2.
                    X_lower = traj_fine_history[l, i, 0, :]   # position at y = 0
                    X_upper = traj_fine_history[l, i, 1, :]   # position at y = h2
                    term = (indicator_D(X_upper) * K_delta(X_upper, x, delta) -
                            indicator_D(X_lower) * K_delta(X_lower, x, delta))
                    # Approximate theta_{i,l} = - u^1 evaluated at (i*h1, h2) divided by h2.
                    # We use the previously computed uFuncs_history.
                    theta_val = - (uFuncs_history[l](X_upper)[0]) / h2
                    visc_term += dt * term * theta_val
            # Subtract the viscous correction with factor 2*nu
            u_val = u_val - 2 * nu * visc_term
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (using double-index for positions)
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates vortex trajectories using the Euler-Maruyama scheme.
    The velocity functions (uFuncs) now incorporate a viscous correction
    based on the history of fine grid trajectories.
    
    Returns:
      traj: a tuple (traj_coarse, traj_fine) where each is an array of shape 
            (num_steps+1, n_x, n_y, 2) representing the trajectories.
      uFuncs: list of functions (one per time step) that compute the local velocity.
      displacements: a tuple (disp_coarse, disp_fine) storing the displacement for each vortex.
    """
    # Initialize trajectory arrays for coarse and fine grids
    traj_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_fine = np.zeros((num_steps + 1, grid_fine.shape[0], grid_fine.shape[1], 2))
    traj_coarse[0] = grid_coarse
    traj_fine[0] = grid_fine
    
    uFuncs = []
    disp_coarse = np.zeros((num_steps, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    disp_fine = np.zeros((num_steps, grid_fine.shape[0], grid_fine.shape[1], 2))
    
    # ----- Pre-compute initial step (step = 0) without viscous term -----
    u_func_0 = make_u_func(traj_coarse[0], traj_fine[0],
                           w0_coarse, w0_fine, A_coarse, A_fine)
    uFuncs.append(u_func_0)
    n1, n2 = traj_coarse[0].shape[0], traj_coarse[0].shape[1]
    for i in range(n1):
        for j in range(n2):
            u_val = u_func_0(traj_coarse[0][i, j])
            dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
            disp = dt * u_val + dW
            disp_coarse[0, i, j] = disp
            traj_coarse[1, i, j] = traj_coarse[0, i, j] + disp
    n1_f, n2_f = traj_fine[0].shape[0], traj_fine[0].shape[1]
    for i in range(n1_f):
        for j in range(n2_f):
            u_val = u_func_0(traj_fine[0][i, j])
            dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
            disp = dt * u_val + dW
            disp_fine[0, i, j] = disp
            traj_fine[1, i, j] = traj_fine[0, i, j] + disp

    # ----- Step 1: Build u_func_1 using history from step 0-1 -----
    u_func_1 = make_u_func(traj_coarse[1], traj_fine[1],
                           w0_coarse, w0_fine, A_coarse, A_fine,
                           traj_fine_history=traj_fine[:2],
                           uFuncs_history=uFuncs,
                           current_step=1)
    uFuncs.append(u_func_1)
    for i in range(n1):
        for j in range(n2):
            u_val = u_func_1(traj_coarse[1][i, j])
            dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
            disp = dt * u_val + dW
            disp_coarse[1, i, j] = disp
            traj_coarse[2, i, j] = traj_coarse[1, i, j] + disp
    for i in range(n1_f):
        for j in range(n2_f):
            u_val = u_func_1(traj_fine[1][i, j])
            dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
            disp = dt * u_val + dW
            disp_fine[1, i, j] = disp
            traj_fine[2, i, j] = traj_fine[1, i, j] + disp

    # ----- Continue simulation from step = 2 to num_steps-1 -----
    for step in range(2, num_steps):
        current_coarse = traj_coarse[step]
        current_fine = traj_fine[step]
        # Pass the fine trajectory history up to current time and list of previous uFuncs.
        u_func = make_u_func(current_coarse, current_fine,
                             w0_coarse, w0_fine, A_coarse, A_fine,
                             traj_fine_history=traj_fine[:step+1],
                             uFuncs_history=uFuncs,
                             current_step=step)
        uFuncs.append(u_func)
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_coarse[step, i, j] = disp
                traj_coarse[step + 1, i, j] = current_coarse[i, j] + disp
        for i in range(n1_f):
            for j in range(n2_f):
                u_val = u_func(current_fine[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_fine[step, i, j] = disp
                traj_fine[step + 1, i, j] = current_fine[i, j] + disp
                
    return (traj_coarse, traj_fine), uFuncs, (disp_coarse, disp_fine)

print("Computing vortex trajectories......")
(trajectories_coarse, trajectories_fine), uFuncs, (disp_coarse, disp_fine) = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation (using the velocity functions from vortex simulation)
# ---------------------------
def generate_boat_grid():
    """
    Generates a boat grid by combining the coarse and fine grids (flattening the double-index arrays)
    into a single flat array of query points.
    """
    grids = generate_nonuniform_grid_D()
    grid_coarse, _ = grids['coarse']
    grid_fine, _ = grids['fine']
    boat_grid = np.concatenate((grid_coarse.reshape(-1, 2), grid_fine.reshape(-1, 2)), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    """
    Simulate boat trajectories on the grid.
    Boat positions are updated using the precomputed uFuncs (drift only, no noise).
    
    Returns:
      boat_positions: array of shape (num_steps+1, num_boats, 2)
      boat_displacements: array of shape (num_steps, num_boats, 2)
    """
    boat_grid = generate_boat_grid()
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

print("Simulating vortex boats......")
boat_positions, boat_displacements = simulate_boats(uFuncs)

# For velocity field display we use the same boat grid as query points.
query_grid = generate_boat_grid()

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
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Initialize the velocity field quiver using the first u_func
U, V = compute_velocity_field(uFuncs[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

# Plot boat positions as a scatter plot.
boat_scatter = ax.scatter(boat_positions[0, :, 0], boat_positions[0, :, 1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    
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
