import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01           # viscosity
T = 15              # final time (seconds)
dt = 0.1            # time step
num_steps = int(T / dt)
delta = 0.01         # mollification parameter

# ---------------------------
# Mesh Parameters
# ---------------------------
h0 = 1.0            # coarse grid spacing (for coarse meshes)
h1 = 0.5            # fine grid spacing in x (for fine meshes)
h2 = 0.05            # fine grid spacing in y (for fine lower half and boundary)

region_x = [-6, 6]
region_y = [-6, 6]  # overall domain

# For the coarse grid lower half, use y1 to y2 and then mirror about y=0.
y1 = region_y[0]    # -6
y4 = region_y[1]    # 6
y2 = -0.2           # splitting point for the coarse grid lower half

# ---------------------------
# Grid Generation: Split Meshes
# ---------------------------
def generate_split_meshes():
    """
    Generate five separate meshes:
      - D01: Coarse upper grid: mirror of D02 (reflect y-coordinate).
      - Db1: Fine upper grid: mirror of Db2.
      - D0:  Boundary grid along y=0. Now generated using the fine mesh in x for higher density.
      - Db2: Fine lower grid over [x1,x2] x [y2,0) using spacing h1 in x and h2 in y.
      - D02: Coarse lower grid over [x1,x2] x [y1,y2) using spacing h0.
    """
    x1, x2 = region_x

    ## Coarse Meshes (using coarse spacing)
    N_x_coarse = int((x2 - x1) / h0) + 1
    N_y_coarse = int((y2 - y1) / h0)  # number of vertical points in coarse lower half
    x_coarse = np.linspace(x1, x2, N_x_coarse)
    y_coarse_lower = np.linspace(y1, y2, N_y_coarse, endpoint=False)
    XX_coarse, YY_coarse_lower = np.meshgrid(x_coarse, y_coarse_lower, indexing='ij')
    D02 = np.stack((XX_coarse, YY_coarse_lower), axis=-1)  # coarse lower grid

    # Coarse upper grid (D01) by reflecting D02:
    D01 = D02.copy()
    D01[:, :, 1] = -D01[:, :, 1]

    ## Boundary grid (D0) along y = 0: use the fine x-mesh for higher density.
    N_x_fine = int((x2 - x1) / h1) + 1
    x_fine = np.linspace(x1, x2, N_x_fine)
    D0 = np.stack((x_fine, np.zeros_like(x_fine)), axis=-1)
    D0 = D0[:, None, :]  # shape: (N_x_fine, 1, 2)

    ## Fine Meshes (using fine spacing)
    N_y_fine = int((0 - y2) / h2)  # number of vertical points in fine lower half
    y_fine_lower = np.linspace(y2, 0, N_y_fine, endpoint=False)
    XX_fine, YY_fine_lower = np.meshgrid(x_fine, y_fine_lower, indexing='ij')
    Db2 = np.stack((XX_fine, YY_fine_lower), axis=-1)  # fine lower grid

    # Fine upper grid (Db1) by reflecting Db2:
    Db1 = Db2.copy()
    Db1[:, :, 1] = -Db1[:, :, 1]

    return D01, D02, D0, Db1, Db2

# Generate the five meshes.
D01, D02, D0, Db1, Db2 = generate_split_meshes()

# Set counts for indexing.
N_coarse = D02.shape[1]    # vertical count for coarse (D02 and D01)
N_fine   = Db2.shape[1]    # vertical count for fine (Db2 and Db1)

def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

def initialize_vortices(grid):
    M, N, _ = grid.shape
    w = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x_val, y_val = grid[i, j]
            w[i, j] = vorticity(x_val, y_val)
    return w

w0_D01 = initialize_vortices(D01)
w0_D02 = initialize_vortices(D02)
w0_D0  = initialize_vortices(D0)    # Dense boundary grid.
w0_Db1 = initialize_vortices(Db1)
w0_Db2 = initialize_vortices(Db2)

# ---------------------------
# Helper Functions
# ---------------------------
def indicator_D(point):
    """Return 1 if the point is in the physical (upper) domain (y > 0), else 0."""
    return 1 if point[1] > 0 else 0

def reflect(point):
    """Reflect a point across the horizontal axis: [x, y] -> [x, -y]."""
    return np.array([point[0], -point[1]])

def reflect_points(pts):
    """Reflect an array of points of shape (N,2) by flipping the y-coordinate."""
    pts_ref = pts.copy()
    pts_ref[:, 1] = -pts_ref[:, 1]
    return pts_ref

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
# u_func Factory Using All Meshes
# ---------------------------
A_coarse = h0 * h0   # area for coarse and boundary meshes
A_fine   = h1 * h2   # area for fine meshes

def make_u_func(curr_D01, curr_D02, curr_D0, curr_Db1, curr_Db2,
                w_D01, w_D02, w_D0, w_Db1, w_Db2,
                A_coarse, A_fine, delta):
    """
    Build the local velocity function u(x). For x in the upper half-plane, the velocity is computed as
    \[
      \hat{u}(x,t_{k+1}) = \sum_{(i_1,i_2)\in D,\;i_2>0} A_{i_1,i_2}\,\omega_{i_1,i_2}\Bigl[K_\delta\bigl(X^{i_1,i_2},x\bigr)-K_\delta\bigl(X^{i_1,-i_2},x\bigr)\Bigr].
    \]
    For x on the boundary (x[1]=0) the velocity is zero, and for x in the lower half-plane, we define
    \[
      u(x,t)=\text{reflect}\Bigl(u\bigl(\text{reflect}(x),t\bigr)\Bigr).
    \]
    """
    def compute_u(x):
        u_val = np.zeros(2)
        # Coarse contribution:
        M_coarse, N_coarse_local = curr_D02.shape[0], curr_D02.shape[1]
        for i in range(M_coarse):
            for j in range(N_coarse_local):
                pos_upper = curr_D02[i, j]
                pos_lower = reflect(pos_upper) # cheating
                # pos_lower = curr_D01[i, j]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                u_val += A_coarse * w_D01[i, j] * (term_upper - term_lower)
        # Fine contribution:
        M_fine, N_fine_local = curr_Db2.shape[0], curr_Db2.shape[1]
        for i in range(M_fine):
            for j in range(N_fine_local):
                pos_upper = curr_Db2[i, j]
                pos_lower = reflect(pos_upper) # cheating
                # pos_lower = curr_Db1[i, j]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                u_val += A_fine * w_Db1[i, j] * (term_upper - term_lower)
        # The boundary grid D0 contributes zero (since indicator_D is zero at y=0).
        return u_val

    def u_func(x):
        if x[1] > 0:
            return compute_u(x)
        elif np.isclose(x[1], 0.0):
            return np.zeros(2)
        else:
            # For x in the lower half-plane, reflect x, compute u, then reflect the velocity.
            x_ref = reflect(x)
            u_val_upper = compute_u(x_ref)
            return reflect(u_val_upper)
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler–Maruyama) for All Meshes
# ---------------------------
def simulate_vortex_trajectories():
    # Allocate arrays for trajectories.
    traj_D02 = np.zeros((num_steps+1,) + D02.shape)
    traj_Db2 = np.zeros((num_steps+1,) + Db2.shape)
    traj_D0  = np.zeros((num_steps+1,) + D0.shape)
    traj_Db1 = np.zeros((num_steps+1,) + Db1.shape)
    traj_D01 = np.zeros((num_steps+1,) + D01.shape)
    
    # Set initial conditions.
    traj_D02[0] = D02
    traj_Db2[0] = Db2
    traj_D0[0]  = D0
    traj_Db1[0] = Db1
    traj_D01[0] = D01

    uFuncs = []
    
    for step in range(num_steps):
        curr_D02 = traj_D02[step].copy()
        curr_Db2 = traj_Db2[step].copy()
        curr_D0  = traj_D0[step].copy()
        curr_Db1 = traj_Db1[step].copy()
        curr_D01 = traj_D01[step].copy()
        
        u_func = make_u_func(curr_D01, curr_D02, curr_D0, curr_Db1, curr_Db2,
                             w0_D01, w0_D02, w0_D0, w0_Db1, w0_Db2,
                             A_coarse, A_fine, delta)
        uFuncs.append(u_func)
        
        M_f, N_f = curr_Db1.shape[0], curr_Db1.shape[1]
        M, N = curr_D01.shape[0], curr_D01.shape[1]

        # Update coarse lower grid (D02)
        for i in range(M):
            for j in range(N):
                u_val = u_func(curr_D02[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D02[step+1, i, j] = curr_D02[i, j] + dt * u_val + dW

        # Update fine lower grid (Db2)
        for i in range(M_f):
            for j in range(N_f):
                u_val = u_func(curr_Db2[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Db2[step+1, i, j] = curr_Db2[i, j] + dt * u_val + dW

        # Update boundary grid (D0)
        M0, N0 = curr_D0.shape[0], curr_D0.shape[1]  # N0 is 1
        for i in range(M0):
            for j in range(N0):
                u_val = u_func(curr_D0[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D0[step+1, i, j] = curr_D0[i, j] + dt * u_val + dW

        # # Update fine upper grid (Db1)
        # 
        # for i in range(M_f):
        #     for j in range(N_f):
        #         u_val = u_func(curr_Db1[i, j])
        #         dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
        #         traj_Db1[step+1, i, j] = curr_Db1[i, j] + dt * u_val + dW
        
        # # Update coarse upper grid (D01)
        # 
        # for i in range(M):
        #     for j in range(N):
        #         u_val = u_func(curr_D01[i, j])
        #         dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
        #         traj_D01[step+1, i, j] = curr_D01[i, j] + dt * u_val + dW

        

    return (traj_D01, traj_D02, traj_D0, traj_Db1, traj_Db2), uFuncs

print("Simulating vortex trajectories...")
(trajectories_D01, trajectories_D02, trajectories_D0,
 trajectories_Db1, trajectories_Db2), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation: Update Boats Using u_func
# ---------------------------
def generate_initial_boat_grid():
    """
    For boat simulation, initially use points from:
      - The boundary grid D0 (which is at y=0),
      - The coarse lower grid D02 reflected to the upper half-plane,
      - The fine lower grid Db2 reflected to the upper half-plane.
    """
    boat_boundary = D0.reshape(-1, 2)
    boat_coarse = reflect_points(D02.reshape(-1, 2))
    boat_fine = reflect_points(Db2.reshape(-1, 2))
    boat_grid = np.concatenate((boat_boundary, boat_coarse, boat_fine), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    boat_positions = []
    boat_grid = generate_initial_boat_grid()
    boat_positions.append(boat_grid)
    num_boats = boat_grid.shape[0]
    for step in range(num_steps):
        u_func = uFuncs[step]
        new_positions = np.zeros_like(boat_grid)
        for b in range(num_boats):
            vel = u_func(boat_grid[b])
            new_positions[b] = boat_grid[b] + dt * vel
        boat_grid = new_positions
        boat_positions.append(boat_grid)
    return np.array(boat_positions)

print("Simulating boat trajectories...")
boat_positions = simulate_boats(uFuncs)

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
    # For visualization, use the initial boat grid.
    return generate_initial_boat_grid()

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

U, V = compute_velocity_field(uFuncs[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

boat_scatter = ax.scatter(boat_positions[0][:, 0], boat_positions[0][:, 1],
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
save_path = os.path.join("animation", "split_meshes.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
