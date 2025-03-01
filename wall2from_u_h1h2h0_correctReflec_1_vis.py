import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.15          # viscosity
T = 15              # final time (seconds)
dt = 0.1            # time step
num_steps = int(T / dt)
delta = 0.1         # mollification parameter

# ---------------------------
# Mesh Parameters
# ---------------------------
h0 = 1.0            # coarse grid spacing
h1 = 0.5            # fine grid spacing in x
h2 = 0.1            # fine grid spacing in y (for the fine grid lower half)

region_x = [-6, 6]
region_y = [-6, 6]  # overall domain for grid generation

# For the symmetric grid, we need a splitting value for the coarse lower mesh.
y1 = region_y[0]    # -6
y4 = region_y[1]    # 6
y2 = -0.5           # splitting point for the coarse grid (lower half: [y1,y2))

# ---------------------------
# Grid Generation: Symmetric Meshes
# ---------------------------
def generate_symmetric_grids():
    """
    1. Create lower grids:
       - Coarse: [x1,x2] x [y1,y2)
       - Fine:   [x1,x2] x [y2, 0)
    2. Mirror them to get the upper half.
    3. Add a boundary mesh at y=0 (for the fine grid).
    4. Combine lower, boundary, and upper parts.
    """
    x1, x2 = region_x

    # --- Coarse grid ---
    # Lower half: from y1 to y2 (endpoint not included)
    N_x_coarse = int((x2 - x1) / h0) + 1
    N_y_coarse = int((y2 - y1) / h0)  # number of vertical points in the lower half
    x_coarse = np.linspace(x1, x2, N_x_coarse)
    y_coarse_lower = np.linspace(y1, y2, N_y_coarse, endpoint=False)
    XX_coarse, YY_coarse_lower = np.meshgrid(x_coarse, y_coarse_lower, indexing='ij')
    # D02: lower coarse grid
    D02 = np.stack((XX_coarse, YY_coarse_lower), axis=-1)
    
    # Mirror to get the upper coarse grid: reflect about y=0
    D01 = D02.copy()
    D01[:, :, 1] = -D01[:, :, 1]
    D01 = D01[:, ::-1, :]  # reverse vertical order
    
    # Combine lower and upper coarse grids
    D0 = np.concatenate((D02, D01), axis=1)  # shape: (N_x_coarse, 2*N_y_coarse, 2)
    
    # --- Fine grid ---
    # Lower half: from y2 to 0 (endpoint not included)
    N_x_fine = int((x2 - x1) / h1) + 1
    N_y_fine_lower = int((0 - y2) / h2)  # number of vertical points in the lower half
    x_fine = np.linspace(x1, x2, N_x_fine)
    y_fine_lower = np.linspace(y2, 0, N_y_fine_lower, endpoint=False)
    XX_fine, YY_fine_lower = np.meshgrid(x_fine, y_fine_lower, indexing='ij')
    # Db2: lower fine grid
    Db2 = np.stack((XX_fine, YY_fine_lower), axis=-1)
    
    # Mirror to get the upper fine grid
    Db1 = Db2.copy()
    Db1[:, :, 1] = -Db1[:, :, 1]
    Db1 = Db1[:, ::-1, :]
    
    # Create a boundary mesh at y=0 (a single row)
    Dbd = np.stack((x_fine, np.zeros_like(x_fine)), axis=-1)
    Dbd = Dbd[:, None, :]  # shape: (N_x_fine, 1, 2)
    
    # Combine fine grids: lower + boundary + upper
    Db = np.concatenate((Db2, Dbd, Db1), axis=1)  # shape: (N_x_fine, 2*N_y_fine_lower+1, 2)
    
    return D0, Db

# Generate the symmetric grids
D0, Db = generate_symmetric_grids()

# Set the number of vertical points in the lower halves (for indexing)
N_coarse = int((y2 - y1) / h0)      # coarse grid lower half count
N_fine = int((0 - y2) / h2)         # fine grid lower half count

# ---------------------------
# Vorticity and Velocity Functions
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

# ---------------------------
# Vortex Initialization
# ---------------------------
def initialize_vortices(grid):
    # grid shape: (M, N, 2)
    M, N, _ = grid.shape
    w = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y)
    return w

# Initialize vorticity for coarse and fine grids
w0_D0 = initialize_vortices(D0)
w0_Db = initialize_vortices(Db)

# ---------------------------
# Helper Functions
# ---------------------------
def reflect(point):
    """Reflect a point about the x-axis."""
    return np.array([point[0], -point[1]])

def indicator_D(point):
    """Return 1 if point is in the physical domain (y > 0), else 0."""
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
# u_func Factory on Combined Meshes
# ---------------------------
A_coarse = h0 * h0
A_fine   = h1 * h2

def make_u_func(curr_D0, curr_Db, w0_D0, w0_Db, A_coarse, A_fine, N_coarse, N_fine):
    """
    Build the local velocity function u(x) using the combined grids.
    
    For the coarse grid, the physical vortices are in the upper half (indices j from N_coarse to 2*N_coarse-1).
    Their mirror vortices are taken from indices j = 2*N_coarse-1 - j.
    
    For the fine grid, note that the combined grid has an extra boundary row at y=0.
    The physical vortices are then for indices j from N_fine+1 to 2*N_fine (since the lower half occupies indices 0 to N_fine-1
    and the boundary is at index N_fine). Their mirror vortices are taken as j = 2*N_fine - j.
    """
    def u_func(x):
        u_val = np.zeros(2)
        # Coarse contribution:
        M, total = curr_D0.shape[0], curr_D0.shape[1]  # total = 2*N_coarse
        for i in range(M):
            for j in range(N_coarse, total):  # only physical (upper) vortices
                pos_upper = curr_D0[i, j]
                mirror_index = 2 * N_coarse - 1 - j
                pos_lower = curr_D0[i, mirror_index]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_coarse * w0_D0[i, j] * diffK
        # Fine contribution:
        M_f, total_f = curr_Db.shape[0], curr_Db.shape[1]  # total_f = 2*N_fine + 1 (including boundary)
        for i in range(M_f):
            # Physical vortices: indices from N_fine+1 to 2*N_fine (exclude boundary at index N_fine)
            for j in range(N_fine+1, total_f):
                pos_upper = curr_Db[i, j]
                mirror_index = 2 * N_fine - j  # mirror index for fine grid
                pos_lower = curr_Db[i, mirror_index]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_fine * w0_Db[i, j] * diffK
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama)
# ---------------------------
def simulate_vortex_trajectories():
    traj_D0 = np.zeros((num_steps+1,) + D0.shape)
    traj_Db = np.zeros((num_steps+1,) + Db.shape)
    traj_D0[0] = D0
    traj_Db[0] = Db

    uFuncs = []
    
    for step in range(num_steps):
        curr_D0 = traj_D0[step].copy()
        curr_Db = traj_Db[step].copy()
        # Build u function using current positions.
        u_func = make_u_func(curr_D0, curr_Db, w0_D0, w0_Db, A_coarse, A_fine, N_coarse, N_fine)
        uFuncs.append(u_func)
        
        # Update coarse grid D0:
        M, total = curr_D0.shape[0], curr_D0.shape[1]
        for i in range(M):
            for j in range(total):
                u_val = u_func(curr_D0[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D0[step+1, i, j] = curr_D0[i, j] + dt * u_val + dW
        # Update fine grid Db:
        M_f, total_f = curr_Db.shape[0], curr_Db.shape[1]
        for i in range(M_f):
            for j in range(total_f):
                u_val = u_func(curr_Db[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Db[step+1, i, j] = curr_Db[i, j] + dt * u_val + dW

    return (traj_D0, traj_Db), uFuncs

print("Simulating vortex trajectories...")
(trajectories_D0, trajectories_Db), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation (using only the physical/upper half of the combined meshes)
# ---------------------------
def generate_boat_grid():
    """
    For boat simulation, extract only the physical (upper) vortices.
    - For the coarse grid: physical vortices are in columns from index N_coarse onward.
    - For the fine grid: physical vortices are in columns from index N_fine+1 onward (excluding the boundary at y=0).
    """
    boat_coarse = trajectories_D0[0][:, N_coarse:, :].reshape(-1, 2)
    boat_fine = trajectories_Db[0][:, N_fine+1:, :].reshape(-1, 2)
    boat_grid = np.concatenate((boat_coarse, boat_fine), axis=0)
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

# ---------------------------
# Velocity Field Query
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
    # Use the physical vortices from coarse and fine grids
    boat_coarse = trajectories_D0[0][:, N_coarse:, :].reshape(-1, 2)
    boat_fine = trajectories_Db[0][:, N_fine+1:, :].reshape(-1, 2)
    query_grid = np.concatenate((boat_coarse, boat_fine), axis=0)
    return query_grid

query_grid = generate_query_grid()

# ---------------------------
# Animation: Velocity Field and Boat Trajectories
# ---------------------------
window_x = region_x
window_y = [0, region_y[1]]  # show only the physical (upper) part: y>=0

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

# Plot boat positions (physical vortices)
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
save_path = os.path.join("animation", "combined_symmetric.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
