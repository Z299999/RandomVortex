import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time  # For timing

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01          # Viscosity
T = 20             # Final time (seconds)
dt = 0.1           # Time step
num_steps = int(T / dt)
delta = 0.1        # Mollification parameter (choose δ < 0.15 for the boundary layer)

# Mesh parameters:
h0 = 1    # Coarse grid spacing (both x and y)
h1 = 0.5  # Fine grid spacing for Region B (x-direction) and Region C (y-direction)
h2 = 0.2  # Fine grid spacing for Region A (both) and for Region B (y-direction) and Region C (x-direction)
# Note: we require h2 < h1 < h0.
layer_thickness = 0.8  # dividing length between fine and coarse regions

region_x = [0, 8]
region_y = [0, 8]
window_x = [region_x[0], region_x[1]+2]
window_y = [region_y[0], region_y[1]+2]

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
    Generates four meshes for the domain as follows:
      - Coarse grid: [x1+hh, x2] x [y1+hh, y2] with mesh size h0 x h0.
      - Region A (Dbxy): [x1, x1+hh) x [y1, y1+hh) with mesh size h2 x h2.
      - Region B (Dbx):   [x1+hh, x2] x [y1, y1+hh) with mesh size h1 x h2.
      - Region C (Dby):   [x1, x1+hh) x [y1+hh, y2] with mesh size h2 x h1.
      
    In this modified version, all grids are returned as 3D arrays with shape (n_i, n_j, 2).
    The corresponding cell-area arrays have shape (n_i, n_j).
    """
    x1, x2 = region_x
    y1, y2 = region_y
    hh = layer_thickness  # dividing length

    # --- Coarse grid: [x1+hh, x2] x [y1+hh, y2] with spacing h0 ---
    num_x_coarse = int((x2 - (x1 + hh)) / h0) + 1
    num_y_coarse = int((y2 - (y1 + hh)) / h0) + 1
    x_coarse = np.linspace(x1 + hh, x2, num_x_coarse)
    y_coarse = np.linspace(y1 + hh, y2, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)  # shape (num_x_coarse, num_y_coarse, 2)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))

    # --- Region A (Dbxy): [x1, x1+hh) x [y1, y1+hh) with spacing h2 x h2 ---
    num_x_Dbxy = int(hh / h2)
    num_y_Dbxy = int(hh / h2)
    x_Dbxy = np.linspace(x1, x1 + hh, num_x_Dbxy, endpoint=False)
    y_Dbxy = np.linspace(y1, y1 + hh, num_y_Dbxy, endpoint=False)
    xx_Dbxy, yy_Dbxy = np.meshgrid(x_Dbxy, y_Dbxy, indexing='ij')
    grid_Dbxy = np.stack((xx_Dbxy, yy_Dbxy), axis=-1)  # shape (num_x_Dbxy, num_y_Dbxy, 2)
    A_Dbxy = h2 * h2 * np.ones((num_x_Dbxy, num_y_Dbxy))

    # --- Region B (Dbx): [x1+hh, x2] x [y1, y1+hh) with spacing h1 x h2 ---
    num_x_Dbx = int((x2 - (x1 + hh)) / h1) + 1
    num_y_Dbx = int(hh / h2)
    x_Dbx = np.linspace(x1 + hh, x2, num_x_Dbx)
    y_Dbx = np.linspace(y1, y1 + hh, num_y_Dbx, endpoint=False)
    xx_Dbx, yy_Dbx = np.meshgrid(x_Dbx, y_Dbx, indexing='ij')
    grid_Dbx = np.stack((xx_Dbx, yy_Dbx), axis=-1)  # shape (num_x_Dbx, num_y_Dbx, 2)
    A_Dbx = h1 * h2 * np.ones((num_x_Dbx, num_y_Dbx))

    # --- Region C (Dby): [x1, x1+hh) x [y1+hh, y2] with spacing h2 x h1 ---
    num_x_Dby = int(hh / h2)
    num_y_Dby = int((y2 - (y1 + hh)) / h1) + 1
    x_Dby = np.linspace(x1, x1 + hh, num_x_Dby, endpoint=False)
    y_Dby = np.linspace(y1 + hh, y2, num_y_Dby)
    xx_Dby, yy_Dby = np.meshgrid(x_Dby, y_Dby, indexing='ij')
    grid_Dby = np.stack((xx_Dby, yy_Dby), axis=-1)  # shape (num_x_Dby, num_y_Dby, 2)
    A_Dby = h2 * h1 * np.ones((num_x_Dby, num_y_Dby))

    print(f"Coarse grid size: {grid_coarse.shape[0]} x {grid_coarse.shape[1]} = {grid_coarse.shape[0]*grid_coarse.shape[1]} points")
    print(f"Region A (Dbxy) grid size: {grid_Dbxy.shape[0]} x {grid_Dbxy.shape[1]} = {grid_Dbxy.shape[0]*grid_Dbxy.shape[1]} points")
    print(f"Region B (Dbx) grid size: {grid_Dbx.shape[0]} x {grid_Dbx.shape[1]} = {grid_Dbx.shape[0]*grid_Dbx.shape[1]} points")
    print(f"Region C (Dby) grid size: {grid_Dby.shape[0]} x {grid_Dby.shape[1]} = {grid_Dby.shape[0]*grid_Dby.shape[1]} points")
    
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

# For the simulation, we combine the three fine regions into one union:
grid_fine = np.concatenate((grid_Dbxy.reshape(-1, 2), grid_Dbx.reshape(-1, 2), grid_Dby.reshape(-1, 2)), axis=0)
A_fine = np.concatenate((A_Dbxy.reshape(-1), A_Dbx.reshape(-1), A_Dby.reshape(-1)), axis=0)

def plot_grid(grid_coarse, grid_Dbxy, grid_Dbx, grid_Dby):
    """
    Plots the coarse grid and the three fine grid regions with different colors.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(region_x[0], region_x[1])
    ax.set_ylim(region_y[0], region_y[1])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Nonuniform Grid Visualization")

    # Coarse grid: reshape to (N,2)
    coarse_points = grid_coarse.reshape(-1, 2)
    ax.scatter(coarse_points[:, 0], coarse_points[:, 1], color='red', label='Coarse Grid', s=20)
    
    # Fine grid regions
    ax.scatter(grid_Dbxy[:, :, 0].reshape(-1), grid_Dbxy[:, :, 1].reshape(-1), color='blue', label='Region A (Dbxy)', s=20)
    ax.scatter(grid_Dbx[:, :, 0].reshape(-1), grid_Dbx[:, :, 1].reshape(-1), color='green', label='Region B (Dbx)', s=20)
    ax.scatter(grid_Dby[:, :, 0].reshape(-1), grid_Dby[:, :, 1].reshape(-1), color='purple', label='Region C (Dby)', s=20)
    
    ax.legend()
    plt.show()

# Plot the grids for visualization
plot_grid(grid_coarse, grid_Dbxy, grid_Dbx, grid_Dby)

# ---------------------------
# Vortex Initialization (for 3D grids)
# ---------------------------
def initialize_vortices_grid(grid):
    """
    Initializes vortex properties (vorticity and velocity) for a grid
    given as a 3D array (n_i, n_j, 2).
    Returns:
      w: 2D array of vorticity values (n_i x n_j)
      u: 3D array of velocities (n_i x n_j x 2)
    """
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

print("* Number of vortices in coarse grid:", grid_coarse.shape[0]*grid_coarse.shape[1])
print("* Number of vortices in Region A:", grid_Dbxy.shape[0]*grid_Dbxy.shape[1])
print("* Number of vortices in Region B:", grid_Dbx.shape[0]*grid_Dbx.shape[1])
print("* Number of vortices in Region C:", grid_Dby.shape[0]*grid_Dby.shape[1])
num_vortices = (grid_coarse.shape[0]*grid_coarse.shape[1] +
                grid_Dbxy.shape[0]*grid_Dbxy.shape[1] +
                grid_Dbx.shape[0]*grid_Dbx.shape[1] +
                grid_Dby.shape[0]*grid_Dby.shape[1])

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

    k1 = -0.5 / np.pi * ((x2 - y2) / r2 - (x2 - y2) / r2_bar1 - (x2 + y2) / r2_bar2 + (x2 + y2) / r2_bar3)
    k2 =  0.5 / np.pi * ((x1 - y1) / r2 - (x1 + y1) / r2_bar1 - (x1 - y1) / r2_bar2 + (x1 + y1) / r2_bar3)

    factor = 1 - np.exp(- (r2 / delta)**2)
    return np.array([k1, k2]) * factor

def indicator_D(point):
    """Indicator for the first quadrant: returns 1 if both coordinates are positive."""
    return 1 if (point[0] > 0 and point[1] > 0) else 0

def Rx(point):
    """Reflects a point across the x-axis: (x1, x2) -> (x1, -x2)."""
    return np.array([point[0], -point[1]])

def Ry(point):
    """Reflects a point across the y-axis: (x1, x2) -> (-x1, point[1])."""
    return np.array([-point[0], point[1]])

# ---------------------------
# Function Factory for u_func
# ---------------------------
def make_u_func(current_coarse, current_fine_list, w0_coarse, w0_fine_list, A_coarse, A_fine_list):
    """
    Creates a u_func that computes the local velocity (drift) at any position x,
    using the current vortex positions from the coarse grid and a list of fine grids.
    Each fine grid is a 3D array (n_i, n_j, 2). Similarly, w0 and A for fine grids
    are provided as lists of 2D arrays.
    """
    current_coarse_copy = current_coarse.copy()
    # current_fine_list is a list of arrays: [current_A, current_B, current_C]
    def u_func(x):
        u_val = np.zeros(2)
        # Loop over the coarse grid (3D array)
        n1, n2, _ = current_coarse_copy.shape
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                term1 = indicator_D(pos) * K_Q(pos, x, delta)
                term2 = indicator_D(Rx(pos)) * K_Q(Rx(pos), x, delta)
                term3 = indicator_D(Ry(pos)) * K_Q(Ry(pos), x, delta)
                term4 = indicator_D(Rx(Ry(pos))) * K_Q(Rx(Ry(pos)), x, delta)
                contrib = term1 - term2 - term3 + term4
                u_val += contrib * w0_coarse[i, j] * A_coarse[i, j]
        # Loop over each fine grid region
        for grid_fine, w_f, A_f in zip(current_fine_list, w0_fine_list, A_fine_list):
            n_i, n_j, _ = grid_fine.shape
            for i in range(n_i):
                for j in range(n_j):
                    pos = grid_fine[i, j]
                    term1 = indicator_D(pos) * K_Q(pos, x, delta)
                    term2 = indicator_D(Rx(pos)) * K_Q(Rx(pos), x, delta)
                    term3 = indicator_D(Ry(pos)) * K_Q(Ry(pos), x, delta)
                    term4 = indicator_D(Rx(Ry(pos))) * K_Q(Rx(Ry(pos)), x, delta)
                    contrib = term1 - term2 - term3 + term4
                    u_val += contrib * w_f[i, j] * A_f[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (for nonuniform grids with 3D indexing)
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates vortex trajectories using the Euler-Maruyama scheme.
    The coarse grid and each fine region (Dbxy, Dbx, and Dby) are updated on their own 3D arrays.
    Returns:
      traj: a tuple (traj_coarse, traj_Dbxy, traj_Dbx, traj_Dby)
      uFuncs: list of functions (one per time step) that compute the local velocity.
      displacements: a tuple (disp_coarse, disp_Dbxy, disp_Dbx, disp_Dby)
    """
    # Get dimensions
    n1, n2, _ = grid_coarse.shape
    n_Dbxy1, n_Dbxy2, _ = grid_Dbxy.shape
    n_Dbx1, n_Dbx2, _ = grid_Dbx.shape
    n_Dby1, n_Dby2, _ = grid_Dby.shape
    
    # Allocate arrays for trajectories (time, i, j, 2)
    traj_coarse = np.zeros((num_steps + 1, n1, n2, 2))
    traj_Dbxy = np.zeros((num_steps + 1, n_Dbxy1, n_Dbxy2, 2))
    traj_Dbx = np.zeros((num_steps + 1, n_Dbx1, n_Dbx2, 2))
    traj_Dby = np.zeros((num_steps + 1, n_Dby1, n_Dby2, 2))
    
    traj_coarse[0] = grid_coarse
    traj_Dbxy[0] = grid_Dbxy
    traj_Dbx[0] = grid_Dbx
    traj_Dby[0] = grid_Dby
    
    uFuncs = []
    
    # Allocate displacement arrays (for logging displacements)
    disp_coarse = np.zeros((num_steps, n1, n2, 2))
    disp_Dbxy = np.zeros((num_steps, n_Dbxy1, n_Dbxy2, 2))
    disp_Dbx = np.zeros((num_steps, n_Dbx1, n_Dbx2, 2))
    disp_Dby = np.zeros((num_steps, n_Dby1, n_Dby2, 2))
    
    # ----- Simulation from step 0 to num_steps - 1 -----
    for step in range(num_steps):
        current_coarse = traj_coarse[step]
        current_Dbxy = traj_Dbxy[step]
        current_Dbx = traj_Dbx[step]
        current_Dby = traj_Dby[step]
        u_func = make_u_func(current_coarse, [current_Dbxy, current_Dbx, current_Dby],
                             w0_coarse, [w0_Dbxy, w0_Dbx, w0_Dby],
                             A_coarse, [A_Dbxy, A_Dbx, A_Dby])
        uFuncs.append(u_func)
        # Update coarse grid
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_coarse[step, i, j] = disp
                traj_coarse[step + 1, i, j] = current_coarse[i, j] + disp
        # Update Region Dbxy
        for i in range(n_Dbxy1):
            for j in range(n_Dbxy2):
                u_val = u_func(current_Dbxy[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_Dbxy[step, i, j] = disp
                traj_Dbxy[step + 1, i, j] = current_Dbxy[i, j] + disp
        # Update Region Dbx
        for i in range(n_Dbx1):
            for j in range(n_Dbx2):
                u_val = u_func(current_Dbx[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_Dbx[step, i, j] = disp
                traj_Dbx[step + 1, i, j] = current_Dbx[i, j] + disp
        # Update Region Dby
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
(trajectories_coarse, trajectories_Dbxy, trajectories_Dbx, trajectories_Dby), uFuncs, (disp_coarse, disp_Dbxy, disp_Dbx, disp_Dby) = simulate_vortex_trajectories()
end_time = time.time()
print(f"Vortex trajectories computed in {end_time - start_time:.2f} seconds")

# ---------------------------
# Boat Simulation (using the velocity functions from vortex simulation)
# ---------------------------
def generate_boat_grid():
    """
    Generates a boat grid by combining the coarse grid and the three fine regions.
    Although the grids are stored as 3D arrays, here they are flattened for the boat simulation.
    Boats on the walls (x-axis or y-axis) are removed.
    """
    boat_grid = np.concatenate((
        grid_coarse.reshape(-1, 2),
        grid_Dbxy.reshape(-1, 2),
        grid_Dbx.reshape(-1, 2),
        grid_Dby.reshape(-1, 2)
    ), axis=0)
    boat_grid = boat_grid[(boat_grid[:, 0] != 0) & (boat_grid[:, 1] != 0)]
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
            boat_displacements[step, b] = dt * vel  # drift only (no noise)
            boat_positions[step + 1, b] = boat_positions[step, b] + dt * vel
    return boat_positions, boat_displacements

print("Simulating boats starting from grid points......")
start_time = time.time()
boat_positions, boat_displacements = simulate_boats(uFuncs)
end_time = time.time()
print(f"Boat simulation completed in {end_time - start_time:.2f} seconds")

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
save_path = os.path.join("animation", "h0h1_Q.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
