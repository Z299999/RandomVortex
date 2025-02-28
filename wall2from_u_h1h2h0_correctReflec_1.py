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
h0 = 1.0    # Coarse mesh spacing (both x and y)
h1 = 0.5    # Fine mesh spacing in the x direction
h2 = 0.1    # Fine mesh spacing in the y direction

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
# Grid Generation: Combine Lower and Upper Meshes
# ---------------------------
def generate_combined_grids():
    """
    Generate the coarse and fine grids as combined meshes.
    The original lower and upper coarse grids (D01 and D02) are combined vertically to form D0.
    Similarly, the fine grids (Db1 and Db2) are combined to form Db.
    
    For the coarse grid:
      - D01 (auxiliary) covers y in [y1, y2]
      - D02 (physical) covers y in [y3, y4]
      They are generated with:
         D01x: 2*N0 points in x, D01y: N3 points in y.
         D02x: same as D01x, D02y: N3 points in y.
      Then the combined coarse grid D0 = concatenate(D01, D02) along axis=1.
    
    Similarly for the fine grid.
    
    Returns:
       D0: combined coarse grid, shape (M_coarse, 2*N_coarse_y, 2)
       Db: combined fine grid, shape (M_fine, 2*N_fine_y, 2)
       A_coarse, A_fine: area elements.
       N_coarse: number of vertical points in lower (or upper) part for coarse grid.
       N_fine: number of vertical points in lower (or upper) part for fine grid.
    """
    x1, x2 = region_x
    y1, y4 = region_y
    # Coarse grid:
    N0 = int(x2 / h0) + 1      # number of x points (divided by 2 in original, here we keep same)
    N2 = int(y4 / h0) + 1      # used in computing y3
    N_coarse = int((y4 / h1))  # Actually, we follow the original: N2 is used for computing y3
    # In original code, we had:
    #   y3 = N2 * h2, y2 = -y3, N3 = int((y2-y1)/h0)+1.
    # We use these definitions:
    y3 = N2 * h2
    y2 = - y3
    N3 = int((y2 - y1) / h0) + 1   # number of vertical points in lower half (for coarse)
    
    # Create coarse lower grid (D01) over y in [y1, y2)
    D01x = np.linspace(x1, x2, 2 * N0)  # note: 2*N0 points in x (as in original)
    D01y = np.linspace(y1, y2, N3, endpoint=False)
    xx_D01, yy_D01 = np.meshgrid(D01x, D01y, indexing='ij')
    grid_D01 = np.stack((xx_D01, yy_D01), axis=-1)
    
    # Create coarse upper grid (D02) over y in [y3, y4]
    D02x = np.linspace(x1, x2, 2 * N0)
    D02y = np.linspace(y3, y4, N3, endpoint=True)
    xx_D02, yy_D02 = np.meshgrid(D02x, D02y, indexing='ij')
    grid_D02 = np.stack((xx_D02, yy_D02), axis=-1)
    
    # Combine coarse grids vertically (axis=1: y direction)
    D0 = np.concatenate((grid_D01, grid_D02), axis=1)  # shape: (2*N0, 2*N3, 2)
    
    # Fine grid:
    N1 = int(y4 / h1) + 1  # number of x points for fine grid
    # Let N_fine_y be defined similarly as the number of vertical points for the lower fine mesh.
    N_fine = int(y4 / h1) + 1  # This is not very clear in the original code but we follow original: N2 for fine?
    # In the original code fine grids:
    #   Db1y: np.linspace(y2, 0, N2, endpoint=False)
    #   Db2y: np.linspace(0, y3, N2, endpoint=False)
    # So here we let N_fine = N2.
    N_fine = N2  # using N2 as the number of vertical points in each half for the fine grid.
    
    Db1x  = np.linspace(x1, x2, 2 * N1)
    Db1y  = np.linspace(y2, 0, N_fine, endpoint=False)
    xx_Db1, yy_Db1 = np.meshgrid(Db1x, Db1y, indexing='ij')
    grid_Db1 = np.stack((xx_Db1, yy_Db1), axis=-1)
    
    Db2x  = np.linspace(x1, x2, 2 * N1)
    Db2y  = np.linspace(0, y3, N_fine, endpoint=False)
    xx_Db2, yy_Db2 = np.meshgrid(Db2x, Db2y, indexing='ij')
    grid_Db2 = np.stack((xx_Db2, yy_Db2), axis=-1)
    
    # Combine fine grids vertically
    Db = np.concatenate((grid_Db1, grid_Db2), axis=1)
    
    A_coarse = h0 * h0
    A_fine   = h1 * h2
    
    return D0, Db, A_coarse, A_fine, N3, N_fine

# Generate the combined grids
D0, Db, A_coarse, A_fine, N_coarse, N_fine = generate_combined_grids()
# For D0, shape is (M_coarse, 2*N_coarse, 2); for Db, shape is (M_fine, 2*N_fine, 2)

# ---------------------------
# Vortex Initialization on Combined Meshes
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

# For the combined coarse grid D0 and combined fine grid Db,
# note that the physical (upper) vortices are in the upper half: indices j from N_coarse to 2*N_coarse-1 (coarse)
# and j from N_fine to 2*N_fine-1 (fine).
w0_D0 = initialize_vortices(D0)
w0_Db = initialize_vortices(Db)

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
# u_func Factory on Combined Meshes
# ---------------------------
def make_u_func(curr_D0, curr_Db, w0_D0, w0_Db, A_coarse, A_fine, N_coarse, N_fine):
    """
    Build the local velocity function u(x) using the combined grids.
    
    For the coarse part: for each physical (upper) vortex in D0, i.e. for indices j in [N_coarse, 2*N_coarse),
    the mirror vortex is taken from index (i, 2*N_coarse - 1 - j).
    
    Similarly for the fine part in Db.
    
    Then, 
      u(x) = u_coarse(x) + u_fine(x),
    where
      u_coarse(x) = Σ_{(i,j) with j>=N_coarse} A_coarse * ω_{i,j} [ 1_D(X_{i,j})K_delta(X_{i,j},x)
                      - 1_D(reflect(X_{i,2*N_coarse-1-j}))K_delta(reflect(X_{i,2*N_coarse-1-j}),x) ]
    and similarly for the fine part.
    """
    def u_func(x):
        u_val = np.zeros(2)
        # Coarse contribution:
        M, total = curr_D0.shape[0], curr_D0.shape[1]  # total = 2*N_coarse
        for i in range(M):
            for j in range(N_coarse, total):  # only upper (physical) vortices
                pos_upper = curr_D0[i, j]
                mirror_index = 2 * N_coarse - 1 - j
                pos_lower = curr_D0[i, mirror_index]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                # We reflect the lower vortex before applying the indicator and kernel:
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_coarse * w0_D0[i, j] * diffK
        # Fine contribution:
        M_f, total_f = curr_Db.shape[0], curr_Db.shape[1]  # total_f = 2*N_fine
        for i in range(M_f):
            for j in range(N_fine, total_f):
                pos_upper = curr_Db[i, j]
                mirror_index = 2 * N_fine - 1 - j
                pos_lower = curr_Db[i, mirror_index]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_fine * w0_Db[i, j] * diffK
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama) on Combined Meshes
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulate the vortex trajectories for the combined coarse grid D0 and combined fine grid Db.
    Update is performed for all points (both physical and auxiliary).
    """
    traj_D0 = np.zeros((num_steps+1,) + D0.shape)
    traj_Db = np.zeros((num_steps+1,) + Db.shape)
    traj_D0[0] = D0
    traj_Db[0] = Db

    uFuncs = []
    
    for step in range(num_steps):
        curr_D0 = traj_D0[step].copy()
        curr_Db = traj_Db[step].copy()
        # Build u function using current combined positions.
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
    For boat simulation, we take only the physical (upper) vortices from the combined grids.
    That is, for the coarse grid D0 we take columns from N_coarse to end,
    and similarly for the fine grid Db.
    Then, we flatten and concatenate them.
    """
    boat_coarse = trajectories_D0[0][:, N_coarse:, :].reshape(-1, 2)
    boat_fine = trajectories_Db[0][:, N_fine:, :].reshape(-1, 2)
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

def compute_velocity_field(u_func, query_points):
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        vel = u_func(query_points[p])
        U[p] = vel[0]
        V[p] = vel[1]
    return U, V

# For query points we use the physical (upper) parts of the combined grids.
def generate_query_grid():
    boat_coarse = trajectories_D0[0][:, N_coarse:, :].reshape(-1, 2)
    boat_fine = trajectories_Db[0][:, N_fine:, :].reshape(-1, 2)
    query_grid = np.concatenate((boat_coarse, boat_fine), axis=0)
    return query_grid

query_grid = generate_query_grid()

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
save_path = os.path.join("animation", "combined_corrReflec.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
