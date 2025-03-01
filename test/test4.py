import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.015          # viscosity
T = 15              # final time (seconds)
dt = 0.1            # time step
num_steps = int(T / dt)
delta = 0.1         # mollification parameter

# ---------------------------
# Mesh Parameters
# ---------------------------
h0 = 1.0            # coarse grid spacing (for coarse meshes)
h1 = 0.5            # fine grid spacing in x (for fine meshes)
h2 = 0.1            # fine grid spacing in y (for fine lower half and boundary)

region_x = [-6, 6]
region_y = [-6, 6]  # overall domain

# For the coarse grid lower half, use y1 to y2 and then mirror about y=0.
y1 = region_y[0]    # -6
y4 = region_y[1]    # 6
y2 = -0.5           # splitting point for the coarse grid lower half

# ---------------------------
# Grid Generation: Split Meshes
# ---------------------------
def generate_split_meshes():
    """
    Generate five separate meshes:
      - D02: Coarse lower grid over [x1,x2] x [y1,y2) using spacing h0.
      - D01: Coarse upper grid: mirror of D02 (reflect y-coordinate).
      - D0:  Boundary grid along y=0, generated with the fine x-mesh for higher density.
      - Db2: Fine lower grid over [x1,x2] x [y2,0) using spacing h1 in x and h2 in y.
      - Db1: Fine upper grid: mirror of Db2.
    """
    x1, x2 = region_x

    ## Coarse Meshes (using coarse spacing)
    N_x_coarse = int((x2 - x1) / h0) + 1
    N_y_coarse = int((y2 - y1) / h0)  # coarse vertical count
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
    N_y_fine = int((0 - y2) / h2)  # fine vertical count
    y_fine_lower = np.linspace(y2, 0, N_y_fine, endpoint=False)
    XX_fine, YY_fine_lower = np.meshgrid(x_fine, y_fine_lower, indexing='ij')
    Db2 = np.stack((XX_fine, YY_fine_lower), axis=-1)  # fine lower grid

    # Fine upper grid (Db1) by reflecting Db2:
    Db1 = Db2.copy()
    Db1[:, :, 1] = -Db1[:, :, 1]

    return D01, D02, D0, Db1, Db2

# Generate the five meshes.
D01, D02, D0, Db1, Db2 = generate_split_meshes()

# ---------------------------
# Vorticity and Velocity Functions
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

# ---------------------------
# Vortex Initialization for Each Mesh
# ---------------------------
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
w0_D0  = initialize_vortices(D0)    # boundary grid (dense in x)
w0_Db1 = initialize_vortices(Db1)
w0_Db2 = initialize_vortices(Db2)

# ---------------------------
# Helper Functions
# ---------------------------
def indicator_D(point):
    """Return 1 if point is in the physical (upper) domain (y > 0), else 0."""
    return 1 if point[1] > 0 else 0

def reflect(point):
    """Reflect a point across the horizontal axis: [x, y] -> [x, -y]."""
    return np.array([point[0], -point[1]])

def reflect_points(pts):
    """Reflect an array of points of shape (N, 2) by flipping the y-coordinate."""
    pts_ref = pts.copy()
    pts_ref[:, 1] = -pts_ref[:, 1]
    return pts_ref

# ---------------------------
# Mollified Biot–Savart Kernel
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
      \hat{u}(x,t_{k+1}) = \sum_{(i_1,i_2)\in D,\,i_2>0} A_{i_1,i_2}\,\omega_{i_1,i_2}\Bigl[K_\delta\bigl(X^{i_1,i_2},x\bigr) - K_\delta\bigl(X^{i_1,-i_2},x\bigr)\Bigr].
    \]
    For x on the boundary (\(x_2=0\)) the velocity is zero, and for x in the lower half-plane, we set
    \[
      u(x,t)=\text{reflect}\Bigl(u\bigl(\text{reflect}(x),t\bigr)\Bigr).
    \]
    """
    def compute_u(x):
        u_val = np.zeros(2)
        # Coarse contribution:
        M_coarse, N_coarse_local = curr_D01.shape[0], curr_D01.shape[1]
        for i in range(M_coarse):
            for j in range(N_coarse_local):
                pos_upper = curr_D01[i, j]
                pos_lower = curr_D02[i, j]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                u_val += A_coarse * w_D01[i, j] * (term_upper - term_lower)
        # Fine contribution:
        M_fine, N_fine_local = curr_Db1.shape[0], curr_Db1.shape[1]
        for i in range(M_fine):
            for j in range(N_fine_local):
                pos_upper = curr_Db1[i, j]
                pos_lower = curr_Db2[i, j]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                u_val += A_fine * w_Db1[i, j] * (term_upper - term_lower)
        # D0 (boundary) contributes zero.
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
# Vortex Trajectory Simulation (Euler--Maruyama) for All Meshes
# ---------------------------
def simulate_vortex_trajectories():
    # Allocate arrays for trajectories.
    traj_D01 = np.zeros((num_steps+1,) + D01.shape)
    traj_D02 = np.zeros((num_steps+1,) + D02.shape)
    traj_D0  = np.zeros((num_steps+1,) + D0.shape)
    traj_Db1 = np.zeros((num_steps+1,) + Db1.shape)
    traj_Db2 = np.zeros((num_steps+1,) + Db2.shape)
    
    # Set initial conditions.
    traj_D01[0] = D01
    traj_D02[0] = D02
    traj_D0[0]  = D0
    traj_Db1[0] = Db1
    traj_Db2[0] = Db2

    uFuncs = []
    
    for step in range(num_steps):
        curr_D01 = traj_D01[step].copy()
        curr_D02 = traj_D02[step].copy()
        curr_D0  = traj_D0[step].copy()
        curr_Db1 = traj_Db1[step].copy()
        curr_Db2 = traj_Db2[step].copy()
        
        u_func = make_u_func(curr_D01, curr_D02, curr_D0, curr_Db1, curr_Db2,
                             w0_D01, w0_D02, w0_D0, w0_Db1, w0_Db2,
                             A_coarse, A_fine, delta)
        uFuncs.append(u_func)
        
        # Update coarse upper grid (D01)
        M, N = curr_D01.shape[0], curr_D01.shape[1]
        for i in range(M):
            for j in range(N):
                u_val = u_func(curr_D01[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D01[step+1, i, j] = curr_D01[i, j] + dt * u_val + dW

        # Update coarse lower grid (D02)
        for i in range(M):
            for j in range(N):
                u_val = u_func(curr_D02[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D02[step+1, i, j] = curr_D02[i, j] + dt * u_val + dW

        # Update boundary grid (D0) -- now updated with Brownian noise.
        M0, N0 = curr_D0.shape[0], curr_D0.shape[1]  # N0 is 1
        for i in range(M0):
            for j in range(N0):
                u_val = u_func(curr_D0[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_D0[step+1, i, j] = curr_D0[i, j] + dt * u_val + dW

        # Update fine upper grid (Db1)
        M_f, N_f = curr_Db1.shape[0], curr_Db1.shape[1]
        for i in range(M_f):
            for j in range(N_f):
                u_val = u_func(curr_Db1[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Db1[step+1, i, j] = curr_Db1[i, j] + dt * u_val + dW

        # Update fine lower grid (Db2)
        for i in range(M_f):
            for j in range(N_f):
                u_val = u_func(curr_Db2[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                traj_Db2[step+1, i, j] = curr_Db2[i, j] + dt * u_val + dW

    return (traj_D01, traj_D02, traj_D0, traj_Db1, traj_Db2), uFuncs

if __name__ == '__main__':
    print("Simulating vortex trajectories...")
    (trajectories_D01, trajectories_D02, trajectories_D0,
     trajectories_Db1, trajectories_Db2), uFuncs = simulate_vortex_trajectories()

    # ---------------------------
    # Animation of Vortex Trajectories
    # ---------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial scatter plots for each grid.
    scat_D01 = ax.scatter(trajectories_D01[0].reshape(-1, 2)[:, 0],
                          trajectories_D01[0].reshape(-1, 2)[:, 1],
                          s=10, color='blue', label="D01 (upper coarse)")
    scat_D02 = ax.scatter(trajectories_D02[0].reshape(-1, 2)[:, 0],
                          trajectories_D02[0].reshape(-1, 2)[:, 1],
                          s=10, color='red', label="D02 (lower coarse)")
    scat_D0 = ax.scatter(trajectories_D0[0].reshape(-1, 2)[:, 0],
                         trajectories_D0[0].reshape(-1, 2)[:, 1],
                         s=10, color='green', label="D0 (boundary)")
    scat_Db1 = ax.scatter(trajectories_Db1[0].reshape(-1, 2)[:, 0],
                          trajectories_Db1[0].reshape(-1, 2)[:, 1],
                          s=10, color='orange', label="Db1 (upper fine)")
    scat_Db2 = ax.scatter(trajectories_Db2[0].reshape(-1, 2)[:, 0],
                          trajectories_Db2[0].reshape(-1, 2)[:, 1],
                          s=10, color='purple', label="Db2 (lower fine)")

    # Set the plot limits and title.
    ax.set_xlim(region_x[0] - 1, region_x[1] + 1)
    ax.set_ylim(region_y[0] - 1, region_y[1] + 1)
    ax.set_title("Vortex Trajectories at t = 0.00 s")
    ax.legend()

    def update(frame):
        # Extract current positions from each grid and update scatter plots.
        data_D01 = trajectories_D01[frame].reshape(-1, 2)
        data_D02 = trajectories_D02[frame].reshape(-1, 2)
        data_D0  = trajectories_D0[frame].reshape(-1, 2)
        data_Db1 = trajectories_Db1[frame].reshape(-1, 2)
        data_Db2 = trajectories_Db2[frame].reshape(-1, 2)
        
        scat_D01.set_offsets(data_D01)
        scat_D02.set_offsets(data_D02)
        scat_D0.set_offsets(data_D0)
        scat_Db1.set_offsets(data_Db1)
        scat_Db2.set_offsets(data_Db2)
        
        ax.set_title(f"Vortex Trajectories at t = {frame*dt:.2f} s")
        return scat_D01, scat_D02, scat_D0, scat_Db1, scat_Db2

    # Create the animation.
    anim = FuncAnimation(fig, update, frames=num_steps+1, interval=50, blit=True)

    # ---------------------------
    # Save Animation as MP4 into the "animation" subfolder.
    # ---------------------------
    # Create subfolder "animation" if it doesn't exist.
    os.makedirs("animation", exist_ok=True)
    mp4_filename = os.path.join("animation", "vortex_points_traj.mp4")
    
    # Set up the writer. Here we use 20 fps.
    writer = FFMpegWriter(fps=25, metadata=dict(artist='Your Name'), bitrate=1800)
    anim.save(mp4_filename, writer=writer)
    print(f"Animation saved as {mp4_filename}")

    plt.show()
