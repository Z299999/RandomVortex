import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.015          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter

# Mesh parameters:
h0 = 1.0    # Coarse mesh grid spacing (both x and y)
h1 = 0.5    # Fine mesh grid spacing in the x direction
h2 = 0.1    # Fine mesh grid spacing in the y direction

region_x = [-6, 6]
region_y = [-6, 6]  # overall domain for grid generation
# For visualization we only show the upper half: [-6,6] x [0,6]
window_x = region_x
window_y = [0, 6]

np.random.seed(42)

# ---------------------------
# Vorticity and Velocity Functions
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return -2 * np.cos(y)

# ---------------------------
# Grid Generation (as provided)
# ---------------------------

def generate_nonuniform_grid_D_reflected():
    
    '''
    D01 and D02 are coarser mesh.
    Db1 and Db2 are the finer mesh.
    D01 and Db1 are in lower half plane.
    D02 and Db2 are in upper half plane.
    '''
    x1, x2 = region_x
    y1, y4 = region_y
    N0 = int(x2 / h0) + 1
    N2 = int(y4 / h0) + 1
    N1 = int(y4 / h1) + 1
    y3 = N2 * h2
    y2 = - y3
    N3 = int((y2 - y1) / h0) + 1

    D01x = np.linspace(x1, x2, 2 * N0)
    D01y = np.linspace(y1, y2, N3, endpoint=False)
    xx_D01, yy_D01 = np.meshgrid(D01x, D01y, indexing='ij')
    grid_D01 = np.stack((xx_D01, yy_D01), axis=-1)

    Db1x  = np.linspace(x1, x2, 2 * N1)
    Db1y  = np.linspace(y2, 0, N2, endpoint=False)
    xx_Db1, yy_Db1 = np.meshgrid(Db1x, Db1y, indexing='ij')
    grid_Db1 = np.stack((xx_Db1, yy_Db1), axis=-1)

    Db2x  = np.linspace(x1, x2, 2 * N1)
    Db2y  = np.linspace(0, y3, N2, endpoint=False)
    xx_Db2, yy_Db2 = np.meshgrid(Db2x, Db2y, indexing='ij')
    grid_Db2 = np.stack((xx_Db2, yy_Db2), axis=-1)

    D02x = np.linspace(x1, x2, 2 * N0)
    D02y = np.linspace(y3, y4, N3)
    xx_D02, yy_D02 = np.meshgrid(D02x, D02y, indexing='ij')
    grid_D02 = np.stack((xx_D02, yy_D02), axis=-1)

    A_coarse = h0 * h0
    A_fine   = h1 * h2

    return grid_D01, grid_D02, grid_Db1, grid_Db2, A_coarse, A_fine

# Generate the four grids
grid_D01, grid_D02, grid_Db1, grid_Db2, A_coarse, A_fine = generate_nonuniform_grid_D_reflected()

# ---------------------------
# Vortex Initialization
# ---------------------------
def initialize_vortices(grid):
    # grid shape is (M, N, 2)
    M, N, _ = grid.shape
    w = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y)
    return w

# Initialize vorticity on each grid.
# The physical (upper) vortices carry the weight;
# the lower (auxiliary) vortices are used to compute the mirror contribution.
w0_D01 = initialize_vortices(grid_D01)  # Coarse lower (auxiliary)
w0_D02 = initialize_vortices(grid_D02)  # Coarse upper (physical)
w0_Db1 = initialize_vortices(grid_Db1)  # Fine lower (auxiliary)
w0_Db2 = initialize_vortices(grid_Db2)  # Fine upper (physical)

# ---------------------------
# Helper Functions: Reflection and Indicator
# ---------------------------
def reflect(point):
    """Reflect a point about the x-axis."""
    return np.array([point[0], -point[1]])

def indicator_D(point):
    """Indicator for domain D: return 1 if point is in D (y>0), else 0."""
    return 1 if point[1] > 0 else 0

# ---------------------------
# Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r2 = (x1 - y1)**2 + (x2 - y2)**2
    r2_bar = (x1 - y1)**2 + (-x2 - y2)**2
    if r2 < 1e-10 or r2_bar < 1e-10:
        return np.zeros(2)
    k1 = 0.5 / np.pi * ((y2 - x2) / r2 - (y2 + x2) / r2_bar)
    k2 = 0.5 / np.pi * ((y1 - x1) / r2_bar - (y1 - x1) / r2)
    factor = 1 - np.exp(- (r2 / delta)**2)
    return np.array([k1, k2]) * factor

# ---------------------------
# u_func Factory Following the Given Scheme with Mirror Indexing
# ---------------------------
def make_u_func(curr_D01, curr_D02, curr_Db1, curr_Db2, 
                w0_D02, w0_Db2, A_coarse, A_fine):
    """
    Build the local velocity function u(x) using the scheme:
    
      u(x) = Σ_{(i,j) in upper grid} A * ω_{i,j} *
             [ 1_D(X_upper(i,j)) K_delta(X_upper(i,j), x)
               - 1_D(reflect(X_lower(i, N-1-j))) K_delta(reflect(X_lower(i, N-1-j)), x) ]
               
    where for a given upper vortex at index (i,j) on grid D02 (or Db2),
    the corresponding lower (auxiliary) vortex is taken from grid D01 (or Db1)
    at index (i, N-1-j) so that it is the mirror of the upper one.
    """
    def u_func(x):
        u_val = np.zeros(2)
        # Coarse contribution
        M, N, _ = curr_D02.shape  # N is number of vertical grid points
        for i in range(M):
            for j in range(N):
                pos_upper = curr_D02[i, j]
                pos_lower = curr_D01[i, N - 1 - j]  # mirror index for lower vortex
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_coarse * w0_D02[i, j] * diffK
        # Fine contribution
        M_f, N_f, _ = curr_Db2.shape
        for i in range(M_f):
            for j in range(N_f):
                pos_upper = curr_Db2[i, j]
                pos_lower = curr_Db1[i, N_f - 1 - j]  # mirror index for lower vortex
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_fine * w0_Db2[i, j] * diffK
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama)
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulate the vortex trajectories for all four grids.
    The update for each vortex is:
      X_{t+1} = X_t + dt * u(X_t) + sqrt(2*nu*dt)*ξ.
    The u function is built at each time step using the upper and corresponding mirror (lower) vortices.
    """
    # Allocate trajectory arrays for each grid
    traj_D01 = np.zeros((num_steps+1,) + grid_D01.shape)
    traj_D02 = np.zeros((num_steps+1,) + grid_D02.shape)
    traj_Db1 = np.zeros((num_steps+1,) + grid_Db1.shape)
    traj_Db2 = np.zeros((num_steps+1,) + grid_Db2.shape)
    
    traj_D01[0] = grid_D01
    traj_D02[0] = grid_D02
    traj_Db1[0] = grid_Db1
    traj_Db2[0] = grid_Db2

    uFuncs = []
    
    for step in range(num_steps):
        # Get current positions
        curr_D01 = traj_D01[step].copy()
        curr_D02 = traj_D02[step].copy()
        curr_Db1 = traj_Db1[step].copy()
        curr_Db2 = traj_Db2[step].copy()
        
        # Build u function for the current time step using the upper grids and their mirrors
        u_func = make_u_func(curr_D01, curr_D02, curr_Db1, curr_Db2, w0_D02, w0_Db2, A_coarse, A_fine)
        uFuncs.append(u_func)
        
        # Update coarse grids (both lower and upper)
        M, N, _ = curr_D02.shape
        for i in range(M):
            for j in range(N):
                # Upper (physical) vortex update (D02)
                u_val_upper = u_func(curr_D02[i, j])
                dW_upper = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D02[step+1, i, j] = curr_D02[i, j] + dt * u_val_upper + dW_upper
                # Lower (auxiliary) vortex update (D01)
                u_val_lower = u_func(curr_D01[i, j])
                dW_lower = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D01[step+1, i, j] = curr_D01[i, j] + dt * u_val_lower + dW_lower

        # Update fine grids (both lower and upper)
        M_f, N_f, _ = curr_Db2.shape
        for i in range(M_f):
            for j in range(N_f):
                # Upper (physical) vortex update (Db2)
                u_val_upper = u_func(curr_Db2[i, j])
                dW_upper = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Db2[step+1, i, j] = curr_Db2[i, j] + dt * u_val_upper + dW_upper
                # Lower (auxiliary) vortex update (Db1)
                u_val_lower = u_func(curr_Db1[i, j])
                dW_lower = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Db1[step+1, i, j] = curr_Db1[i, j] + dt * u_val_lower + dW_lower
                
    return (traj_D01, traj_D02, traj_Db1, traj_Db2), uFuncs

print("Simulating vortex trajectories...")
(trajectories_D01, trajectories_D02, trajectories_Db1, trajectories_Db2), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation (using only the upper half-plane vortices)
# ---------------------------
def generate_boat_grid():
    """
    For the boat simulation, use only the upper half-plane grids.
    Combine the coarse upper grid (D02) and the fine upper grid (Db2) into one flat array.
    """
    boat_grid = np.concatenate((grid_D02.reshape(-1, 2), grid_Db2.reshape(-1, 2)), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
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

print("Simulating boat trajectories...")
boat_positions, boat_displacements = simulate_boats(uFuncs)

def compute_velocity_field(u_func, query_points):
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        vel = u_func(query_points[p])
        U[p] = vel[0]
        V[p] = vel[1]
    return U, V

# Use the boat grid (upper half only) as query points
query_grid = generate_boat_grid()

# ---------------------------
# Animation: Velocity Field and Boat Trajectories
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

# Plot boat positions (upper half only)
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

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "corrReflec.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
