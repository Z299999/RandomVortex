import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.15          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step (this is our h)
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter

# Mesh parameters:
# h0: coarse mesh spacing (both x and y)
# h1: fine mesh spacing in x
# h2: fine mesh spacing in y
h0 = 1.0    # Coarse grid spacing (both x and y)
h1 = 0.5    # Fine grid spacing in x
h2 = 0.1    # Fine grid spacing in y

Re = 0.0001 / nu
layer_thickness = 1 * np.sqrt(Re)  # used as the dividing y-level between fine and coarse grids

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

# For the cutoff correction we choose eps equal to h2
eps = h2

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
    Generates a nonuniform grid in D with a coarse grid covering y in [y3, y2]
    (using spacing h0) and a fine grid covering y in [y1, y3] (using h1 in x and h2 in y).
    Here y3 is taken as layer_thickness.
    """
    x1, x2 = region_x
    y1, y2 = region_y
    y3 = layer_thickness  # dividing line between fine and coarse grids in y
    
    # Coarse grid: y in [y3, y2] with spacing h0
    num_x_coarse = int((x2 - x1) / h0) + 1
    num_y_coarse = int((y2 - y3) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))
    
    # Fine grid: y in [y1, y3] with spacing h1 (x) and h2 (y)
    num_x_fine = int((x2 - x1) / h1) + 1
    num_y_fine = int((y3 - y1) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y1, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = np.stack((xx_fine, yy_fine), axis=-1)
    A_fine = h1 * h2 * np.ones((num_x_fine, num_y_fine))
    
    print(f"Coarse grid shape: {grid_coarse.shape}, number of points: {grid_coarse.shape[0]*grid_coarse.shape[1]}")
    print(f"Fine grid shape: {grid_fine.shape}, number of points: {grid_fine.shape[0]*grid_fine.shape[1]}")
    
    return {'coarse': (grid_coarse, A_coarse), 'fine': (grid_fine, A_fine)}

# Generate grids for vortex initialization
grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_fine, A_fine = grids['fine']

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
    # D is the upper half-plane (y > 0)
    return 1 if point[1] > 0 else 0

def reflect(point):
    return np.array([point[0], -point[1]])

# ---------------------------
# Additional Functions for Scheme 2
# ---------------------------
def G(x, t):
    """External force function. (Example: constant force)"""
    return 0

def theta_func(x, t):
    """Boundary vorticity function. (Example function)"""
    return -np.sin(x[0])

def phi_double_prime(r):
    """Cut-off function second derivative.
       Returns 324*(r - 0.5) if r in [1/3, 2/3], 0 otherwise."""
    if r >= 1/3 and r <= 2/3:
        return 324 * (r - 0.5)
    else:
        return 0

def compute_gamma(traj):
    """
    Given a vortex trajectory history (array of shape (L, 2)),
    compute gamma = max { l : traj[l,1] <= 0 }.
    If never below or equal to zero, return -1.
    """
    gamma = -1
    for l, pos in enumerate(traj):
        if pos[1] <= 0:
            gamma = l
    return gamma

# ---------------------------
# New Function Factory for u_func (Scheme 2)
# ---------------------------
def make_u_func_scheme2(current_coarse, current_fine, w0_coarse, w0_fine, A_coarse, A_fine,
                        traj_fine_history=None, current_step=None, eps=eps):
    """
    Creates a velocity function u_func(x) that computes u(x,t_{k+1}) according to Scheme 2.
    
    It consists of three parts:
      1. The inviscid Biot-Savart contribution (over coarse and fine grids).
      2. A force term: for each fine-grid vortex (with j > 0), sum over past time steps l 
         (with t_l > gamma) of G(X_{t_l}, t_l), multiplied by dt and the kernel.
      3. A viscous term: for each fine-grid vortex (with j > 0), sum over past time steps l 
         (with t_l > gamma) of θ(X_{t_l}, t_l) φ''(X_{t_l}/ε), multiplied by the area factor,
         dt and the kernel, and scaled by ν/ε².
    """
    current_coarse_copy = current_coarse.copy()
    current_fine_copy = current_fine.copy()
    
    def u_func(x):
        u_val = np.zeros(2)
        # (1) Inviscid contribution (as in the previous scheme)
        # Coarse grid:
        n1, n2 = current_coarse_copy.shape[0], current_coarse_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                term = indicator_D(pos) * K_delta(pos, x, delta) - indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val += term * w0_coarse[i, j] * A_coarse[i, j]
        # Fine grid (inviscid part):
        n1_f, n2_f = current_fine_copy.shape[0], current_fine_copy.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                pos = current_fine_copy[i, j]
                term = indicator_D(pos) * K_delta(pos, x, delta) - indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val += term * w0_fine[i, j] * A_fine[i, j]
                
        # (2) Additional terms (force & viscous) computed only on fine-grid vortices with j>0.
        if traj_fine_history is not None and current_step is not None:
            for i in range(n1_f):
                for j in range(1, n2_f):  # j=0 is at the wall; use j>0 only.
                    current_pos = current_fine_copy[i, j]
                    # Extract the history for vortex (i,j) up to the current time step:
                    hist = traj_fine_history[:, i, j, :]  # shape (current_step+1, 2)
                    gamma = compute_gamma(hist)
                    force_sum = 0.0
                    visc_sum = 0.0
                    for l in range(current_step+1):
                        if l > gamma:
                            t_l = l * dt
                            pos_l = hist[l]
                            force_sum += G(pos_l, t_l)
                            visc_sum += theta_func(pos_l, t_l) * phi_double_prime(pos_l[1] / eps)
                    K_val = K_delta(current_pos, x, delta)
                    u_val += A_fine[i, j] * dt * K_val * force_sum
                    u_val += (nu/(eps**2)) * (h1 * h2 * dt) * K_val * visc_sum
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Scheme 2)
# ---------------------------
def simulate_vortex_trajectories_scheme2():
    """
    Simulates vortex trajectories using Euler-Maruyama.
    For the fine grid, the full trajectory history is stored for use in the force and viscous terms.
    
    Returns:
      (traj_coarse, traj_fine): trajectories (shape (num_steps+1, n_x, n_y, 2)).
      uFuncs: list of velocity functions (one per time step) computed using Scheme 2.
      (disp_coarse, disp_fine): displacements at each step.
    """
    traj_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_fine   = np.zeros((num_steps + 1, grid_fine.shape[0], grid_fine.shape[1], 2))
    traj_coarse[0] = grid_coarse
    traj_fine[0]   = grid_fine
    
    uFuncs = []
    disp_coarse = np.zeros((num_steps, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    disp_fine   = np.zeros((num_steps, grid_fine.shape[0], grid_fine.shape[1], 2))
    
    # For step 0, we use only the inviscid contribution (no history yet)
    u_func_0 = make_u_func_scheme2(grid_coarse, grid_fine, w0_coarse, w0_fine, A_coarse, A_fine)
    uFuncs.append(u_func_0)
    n1, n2 = traj_coarse[0].shape[0], traj_coarse[0].shape[1]
    for i in range(n1):
        for j in range(n2):
            u_val = u_func_0(traj_coarse[0][i, j])
            dW = np.sqrt(2*nu*dt)*np.random.randn(2)
            disp = dt*u_val + dW
            disp_coarse[0, i, j] = disp
            traj_coarse[1, i, j] = traj_coarse[0, i, j] + disp
    n1_f, n2_f = traj_fine[0].shape[0], traj_fine[0].shape[1]
    for i in range(n1_f):
        for j in range(n2_f):
            u_val = u_func_0(traj_fine[0][i, j])
            dW = np.sqrt(2*nu*dt)*np.random.randn(2)
            disp = dt*u_val + dW
            disp_fine[0, i, j] = disp
            traj_fine[1, i, j] = traj_fine[0, i, j] + disp
            
    # For subsequent steps, use Scheme 2 (with history)
    for step in range(1, num_steps):
        current_coarse = traj_coarse[step]
        current_fine   = traj_fine[step]
        # Pass fine grid trajectory history up to step+1 (shape: (step+1, n1_f, n2_f, 2))
        u_func = make_u_func_scheme2(current_coarse, current_fine,
                                     w0_coarse, w0_fine, A_coarse, A_fine,
                                     traj_fine_history=traj_fine[:step+1],
                                     current_step=step, eps=eps)
        uFuncs.append(u_func)
        # Update coarse grid trajectories:
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i,j])
                dW = np.sqrt(2*nu*dt)*np.random.randn(2)
                disp = dt*u_val + dW
                disp_coarse[step, i, j] = disp
                traj_coarse[step+1, i, j] = current_coarse[i,j] + disp
        # Update fine grid trajectories:
        for i in range(n1_f):
            for j in range(n2_f):
                u_val = u_func(current_fine[i,j])
                dW = np.sqrt(2*nu*dt)*np.random.randn(2)
                disp = dt*u_val + dW
                disp_fine[step, i, j] = disp
                traj_fine[step+1, i, j] = current_fine[i,j] + disp
    return (traj_coarse, traj_fine), uFuncs, (disp_coarse, disp_fine)

print("Computing vortex trajectories using Scheme 2 ......")
(trajectories_coarse, trajectories_fine), uFuncs, (disp_coarse, disp_fine) = simulate_vortex_trajectories_scheme2()

# ---------------------------
# Boat Simulation (using the velocity functions)
# ---------------------------
def generate_boat_grid():
    """
    Generates a boat grid by combining the coarse and fine grids into a single flat array of query points.
    """
    grids = generate_nonuniform_grid_D()
    grid_coarse, _ = grids['coarse']
    grid_fine, _ = grids['fine']
    boat_grid = np.concatenate((grid_coarse.reshape(-1,2), grid_fine.reshape(-1,2)), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    """
    Simulate boat trajectories using the precomputed uFuncs (drift only, no noise).
    """
    boat_grid = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    boat_displacements = np.zeros((num_steps, num_boats, 2))
    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            vel = u_func(boat_positions[step, b])
            boat_displacements[step, b] = dt * vel
            boat_positions[step+1, b] = boat_positions[step, b] + dt * vel
    return boat_positions, boat_displacements

print("Simulating vortex boats ......")
boat_positions, boat_displacements = simulate_boats(uFuncs)

# ---------------------------
# Velocity Field Display and Animation
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

query_grid = generate_boat_grid()

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

U, V = compute_velocity_field(uFuncs[0], query_grid)
vel_quiver = ax.quiver(query_grid[:,0], query_grid[:,1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

boat_scatter = ax.scatter(boat_positions[0,:,0], boat_positions[0,:,1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    boat_scatter.set_offsets(boat_positions[frame])
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex_scheme2.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
