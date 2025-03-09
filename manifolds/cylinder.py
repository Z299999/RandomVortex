import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import time  # For timing

# ============================
# Simulation Parameters
# ============================
nu = 0.000          # Viscosity (here zero so no noise)
T = 15              # Final time (seconds)
dt = 0.04           # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)

# Mesh spacings:
h0 = 0.8              # Coarse spacing for regions D₁ and D₂
hd = 0.4            # Dense spacing for region D

region_x = [-6, 6]
region_y = [0, 4]
window_x = region_x
window_y = region_y

np.random.seed(42)

# ---------------------------
# Basic Physics Functions
# ---------------------------
def velocity(x, y):
    return np.array([0, -np.sin(2*x)])

def vorticity(x, y):
    return -2 * np.cos(2*x)

def K_R2(x, y, delta=0.01):
    """Mollified Biot–Savart kernel."""
    r = np.linalg.norm(x - y)
    if r < 1e-10:
        return np.zeros(2)
    factor = 1 - np.exp(- (r / delta)**2)
    K1 = - (x[1] - y[1]) / (2 * np.pi * r**2) * factor
    K2 = (x[0] - y[0]) / (2 * np.pi * r**2) * factor
    return np.array([K1, K2])

# ---------------------------
# Grid Generator (for simulation and initialization)
# ---------------------------
def generate_simulation_grid():
    """
    Generates three grids:
      - D: central region (x in [-2,2)) with dense spacing hd (area = hd^2)
      - D₁: left region (x in [-6,-2)) with coarse spacing h0 (area = h0^2)
      - D₂: right region (x in [2,6)) with coarse spacing h0 (area = h0^2)
    In all cases, y runs from 0 to 6.
    Points on the right and top boundaries are removed using endpoint=False.
    Returns:
      grid_D, A_D, grid_D1, A_D1, grid_D2, A_D2.
    """
    # For region D:
    num_x_D = int((2 - (-2)) / hd)    # 4/0.5 = 8
    num_y_D = int((6 - 0) / hd)         # 6/0.5 = 12
    x_D = np.linspace(-2, 2, num_x_D, endpoint=False)
    y_D = np.linspace(0, 6, num_y_D, endpoint=False)
    xx_D, yy_D = np.meshgrid(x_D, y_D, indexing='ij')
    grid_D = np.stack((xx_D, yy_D), axis=-1)
    A_D = (hd**2) * np.ones(xx_D.shape)
    
    # For region D₁ (coarse):
    num_x_D1 = int(( -2 - (-6) ) / h0)  # 4/1 = 4
    num_y_D1 = int((6 - 0) / h0)         # 6/1 = 6
    x_D1 = np.linspace(-6, -2, num_x_D1, endpoint=False)
    y_D1 = np.linspace(0, 6, num_y_D1, endpoint=False)
    xx_D1, yy_D1 = np.meshgrid(x_D1, y_D1, indexing='ij')
    grid_D1 = np.stack((xx_D1, yy_D1), axis=-1)
    A_D1 = (h0**2) * np.ones(xx_D1.shape)
    
    # For region D₂ (coarse):
    num_x_D2 = int((6 - 2) / h0)       # 4/1 = 4
    num_y_D2 = int((6 - 0) / h0)         # 6/1 = 6
    x_D2 = np.linspace(2, 6, num_x_D2, endpoint=False)
    y_D2 = np.linspace(0, 6, num_y_D2, endpoint=False)
    xx_D2, yy_D2 = np.meshgrid(x_D2, y_D2, indexing='ij')
    grid_D2 = np.stack((xx_D2, yy_D2), axis=-1)
    A_D2 = (h0**2) * np.ones(xx_D2.shape)
    
    print(f"Region D: {grid_D.size//2} points, D₁: {grid_D1.size//2} points, D₂: {grid_D2.size//2} points")
    return grid_D, A_D, grid_D1, A_D1, grid_D2, A_D2

# Generate the simulation grids and areas
grid_D, A_D, grid_D1, A_D1, grid_D2, A_D2 = generate_simulation_grid()

# ---------------------------
# Vortex Initialization
# ---------------------------
def initialize_vortices(grid_D, grid_D1, grid_D2):
    """
    For the central region D, initialize vortices with nonzero vorticity.
    For regions D₁ and D₂, initialize with zeros.
    Returns tuples: (w, u) for each region.
    """
    nD = grid_D.shape[:2]
    w_D = np.zeros(nD)
    u_D = np.zeros(grid_D.shape)
    for i in range(nD[0]):
        for j in range(nD[1]):
            x, y = grid_D[i, j]
            w_D[i, j] = vorticity(x, y)
            u_D[i, j] = velocity(x, y)
            
    nD1 = grid_D1.shape[:2]
    w_D1 = np.zeros(nD1)
    u_D1 = np.zeros(grid_D1.shape)
    nD2 = grid_D2.shape[:2]
    w_D2 = np.zeros(nD2)
    u_D2 = np.zeros(grid_D2.shape)
    return (w_D, u_D), (w_D1, u_D1), (w_D2, u_D2)

(w_D, u_D), (w_D1, u_D1), (w_D2, u_D2) = initialize_vortices(grid_D, grid_D1, grid_D2)

# Copy initial vorticity from D to D₁ and D₂.
def copy_initial_vorticity(grid_D, w_D, grid_D1, w_D1, grid_D2, w_D2):
    """
    For each point in D₁, set its vorticity to that of the corresponding point
    in D (with a shift of +4 in x), and for D₂ use a shift of -4.
    The mapping uses the fact that h0 = 2*hd.
    """
    for i in range(grid_D1.shape[0]):
        for j in range(grid_D1.shape[1]):
            x, y = grid_D1[i, j]
            i_dense = int(round((x + 4 - (-2)) / hd))
            j_dense = j * int(h0/hd)
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D1[i, j] = w_D[i_dense, j_dense]
    for i in range(grid_D2.shape[0]):
        for j in range(grid_D2.shape[1]):
            x, y = grid_D2[i, j]
            i_dense = int(round((x - 4 - (-2)) / hd))
            j_dense = j * int(h0/hd)
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D2[i, j] = w_D[i_dense, j_dense]
    return w_D1, w_D2

w_D1, w_D2 = copy_initial_vorticity(grid_D, w_D, grid_D1, w_D1, grid_D2, w_D2)

# ---------------------------
# Velocity Function Factory for the Combined State
# ---------------------------
def make_u_func_sim(current_state, w_state, A_state):
    """
    Given the current vortex positions (a tuple for regions D, D₁, D₂),
    the (constant) vortex strengths, and areas, returns a velocity function u(x)
    computed as the sum over all regions.
    """
    traj_D, traj_D1, traj_D2 = current_state
    w_D, w_D1, w_D2 = w_state
    A_D, A_D1, A_D2 = A_state
    def u_func(x):
        u_val = np.zeros(2)
        for i in range(traj_D.shape[0]):
            for j in range(traj_D.shape[1]):
                pos = traj_D[i, j]
                u_val += K_R2(pos, x, delta) * w_D[i, j] * A_D[i, j]
        for i in range(traj_D1.shape[0]):
            for j in range(traj_D1.shape[1]):
                pos = traj_D1[i, j]
                u_val += K_R2(pos, x, delta) * w_D1[i, j] * A_D1[i, j]
        for i in range(traj_D2.shape[0]):
            for j in range(traj_D2.shape[1]):
                pos = traj_D2[i, j]
                u_val += K_R2(pos, x, delta) * w_D2[i, j] * A_D2[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    """
    Evolves vortex trajectories using Euler–Maruyama.
    The simulation state is maintained separately for regions D, D₁, and D₂.
    After each step, the trajectories in D₁ and D₂ are updated by copying from D.
    """
    traj_D = np.zeros((num_steps+1,) + grid_D.shape)
    traj_D1 = np.zeros((num_steps+1,) + grid_D1.shape)
    traj_D2 = np.zeros((num_steps+1,) + grid_D2.shape)
    traj_D[0] = grid_D
    traj_D1[0] = grid_D1
    traj_D2[0] = grid_D2
    
    uFuncs = []
    w_state = (w_D, w_D1, w_D2)
    A_state = (A_D, A_D1, A_D2)
    
    for step in range(num_steps):
        current_state = (traj_D[step], traj_D1[step], traj_D2[step])
        u_func = make_u_func_sim(current_state, w_state, A_state)
        uFuncs.append(u_func)
        for i in range(traj_D[step].shape[0]):
            for j in range(traj_D[step].shape[1]):
                pos = traj_D[step, i, j]
                traj_D[step+1, i, j] = pos + dt * u_func(pos)
        for i in range(traj_D1[step].shape[0]):
            for j in range(traj_D1[step].shape[1]):
                pos = traj_D1[step, i, j]
                traj_D1[step+1, i, j] = pos + dt * u_func(pos)
        for i in range(traj_D2[step].shape[0]):
            for j in range(traj_D2[step].shape[1]):
                pos = traj_D2[step, i, j]
                traj_D2[step+1, i, j] = pos + dt * u_func(pos)
                
        # Copy/update D₁ from D (shift left by 4)
        for i in range(traj_D1[0].shape[0]):
            for j in range(traj_D1[0].shape[1]):
                x, y = grid_D1[i, j]
                i_dense = int(round((x + 6) / hd))
                j_dense = j * 2
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    traj_D1[step+1, i, j] = traj_D[step+1, i_dense, j_dense] - np.array([4, 0])
        # Copy/update D₂ from D (shift right by 4)
        for i in range(traj_D2[0].shape[0]):
            for j in range(traj_D2[0].shape[1]):
                x, y = grid_D2[i, j]
                i_dense = int(round((x - 2) / hd))
                j_dense = j * 2
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    traj_D2[step+1, i, j] = traj_D[step+1, i_dense, j_dense] + np.array([4, 0])
                    
    return (traj_D, traj_D1, traj_D2), uFuncs

(simTraj_D, simTraj_D1, simTraj_D2), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation (only for region D with periodic boundary conditions)
# ---------------------------
def generate_boat_grid():
    """
    Generates a boat grid using only the central region D.
    """
    return grid_D.reshape(-1, 2)

def simulate_boats(uFuncs):
    boat_grid = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    boat_colors = np.full(num_boats, 'red')
    
    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            pos = boat_positions[step, b]
            new_pos = pos + dt * u_func(pos)
            if new_pos[0] >= 2:
                new_pos[0] -= 4
            elif new_pos[0] < -2:
                new_pos[0] += 4
            boat_positions[step+1, b] = new_pos
    return boat_positions, boat_colors

boat_positions, boat_colors = simulate_boats(uFuncs)

# ---------------------------
# Velocity Query Grids for Visualization (in the original rectangular domain)
# ---------------------------
def generate_velocity_query_grids():
    """
    Generates query grids for computing the velocity field for visualization.
    For region D, use dense spacing hd; for D₁ and D₂, use coarse spacing h0.
    Points on the right and top boundaries are removed (endpoint=False).
    """
    num_x_D = int((2 - (-2)) / hd)
    num_y_D = int((6 - 0) / hd)
    x_D = np.linspace(-2, 2, num_x_D, endpoint=False)
    y_D = np.linspace(0, 6, num_y_D, endpoint=False)
    xx_D, yy_D = np.meshgrid(x_D, y_D, indexing='ij')
    query_D = np.stack((xx_D, yy_D), axis=-1).reshape(-1, 2)
    
    num_x_D1 = int(( -2 - (-6) ) / h0)
    num_y_D1 = int((6 - 0) / h0)
    x_D1 = np.linspace(-6, -2, num_x_D1, endpoint=False)
    y_D1 = np.linspace(0, 6, num_y_D1, endpoint=False)
    xx_D1, yy_D1 = np.meshgrid(x_D1, y_D1, indexing='ij')
    query_D1 = np.stack((xx_D1, yy_D1), axis=-1).reshape(-1, 2)
    
    num_x_D2 = int((6 - 2) / h0)
    num_y_D2 = int((6 - 0) / h0)
    x_D2 = np.linspace(2, 6, num_x_D2, endpoint=False)
    y_D2 = np.linspace(0, 6, num_y_D2, endpoint=False)
    xx_D2, yy_D2 = np.meshgrid(x_D2, y_D2, indexing='ij')
    query_D2 = np.stack((xx_D2, yy_D2), axis=-1).reshape(-1, 2)
    
    return query_D, query_D1, query_D2

query_D, query_D1, query_D2 = generate_velocity_query_grids()
query_all_initial = np.concatenate([query_D, query_D1, query_D2], axis=0)

def compute_velocity_field_regions(u_func, query_D, query_D1, query_D2):
    """
    Computes velocity components at each query point and returns concatenated positions and velocities.
    """
    def compute_for_query(query):
        U = np.array([u_func(q)[0] for q in query])
        V = np.array([u_func(q)[1] for q in query])
        return U, V
    U_D, V_D = compute_for_query(query_D)
    U_D1, V_D1 = compute_for_query(query_D1)
    U_D2, V_D2 = compute_for_query(query_D2)
    
    query_all = np.concatenate([query_D, query_D1, query_D2], axis=0)
    U_all = np.concatenate([U_D, U_D1, U_D2])
    V_all = np.concatenate([V_D, V_D1, V_D2])
    return query_all, U_all, V_all

# ============================
# Cylindrical Mapping Functions
# ============================
def to_cylindrical(point):
    """
    Maps a point (x,y) in [-2,2]x[0,6] to cylindrical coordinates.
    theta = (pi/2)*(x+2); (X,Y,Z) = (cos(theta), sin(theta), y).
    """
    x, y = point
    theta = (np.pi/2) * (x + 2)
    return np.array([np.cos(theta), np.sin(theta), y])

def velocity_to_cylindrical(point, vel):
    """
    Transforms a 2D velocity vector at 'point' from rectangular coordinates to the 3D velocity on the cylinder.
    Using the Jacobian of F(x,y) = (cos(theta), sin(theta), y) with theta=(pi/2)(x+2):
      (u,v) -> ( - (pi/2)*sin(theta)*u, (pi/2)*cos(theta)*u, v ).
    """
    x, y = point
    u, v = vel
    theta = (np.pi/2) * (x + 2)
    return np.array([- (np.pi/2) * np.sin(theta) * u,
                     (np.pi/2) * np.cos(theta) * u,
                     v])

# ============================
# Prepare the Static Cylinder Surface for 3D Plotting
# ============================
u_vals = np.linspace(-2, 2, 50, endpoint=False)
v_vals = np.linspace(0, 6, 50, endpoint=False)
U_mesh, V_mesh = np.meshgrid(u_vals, v_vals, indexing='ij')
theta_mesh = (np.pi/2) * (U_mesh + 2)
X_cyl = np.cos(theta_mesh)
Y_cyl = np.sin(theta_mesh)
Z_cyl = V_mesh

# ============================
# Create the Animation with 2 Subplots: 2D (left) and 3D (right)
# ============================
fig = plt.figure(figsize=(16, 8))
# Left: 2D axes
ax2d = fig.add_subplot(1, 2, 1)
# Right: 3D axes
ax3d = fig.add_subplot(1, 2, 2, projection='3d')

# Initialize objects for 2D view.
vel_quiver_2d = None
boat_scatter_2d = None

# For 3D view, we will plot the cylinder surface first.
def init_3d(ax):
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='cyan', rstride=2, cstride=2, edgecolor='none')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Initial draw for 2D view.
ax2d.set_xlim(window_x)
ax2d.set_ylim(window_y)
ax2d.set_aspect('equal')
ax2d.grid(True)
ax2d.set_title("2D Simulation (Rectangular Domain)")

# ============================
# Combined Update Function for Both Subplots
# ============================
def update(frame):
    global vel_quiver_2d, boat_scatter_2d
    t_current = frame * dt

    # ----- 2D Update (Left subplot) -----
    ax2d.cla()
    ax2d.set_xlim(window_x)
    ax2d.set_ylim(window_y)
    ax2d.set_aspect('equal')
    ax2d.grid(True)
    ax2d.set_title(f"2D Simulation (t={t_current:.2f})")
    
    u_func = uFuncs[frame] if frame < len(uFuncs) else uFuncs[-1]
    query_all, U_all, V_all = compute_velocity_field_regions(u_func, query_D, query_D1, query_D2)
    # Plot velocity field as quiver in 2D.
    vel_quiver_2d = ax2d.quiver(query_all[:, 0], query_all[:, 1], U_all, V_all,
                                 color='black', alpha=0.9, pivot='mid', scale_units='xy')
    # Plot boat positions (from region D).
    ax2d.scatter(boat_positions[frame][:, 0], boat_positions[frame][:, 1],
                 s=20, c=boat_colors, zorder=3)
    
    # ----- 3D Update (Right subplot) -----
    ax3d.cla()
    init_3d(ax3d)
    
    # Transform velocity query points and vectors into cylindrical coordinates.
    cyl_positions = np.array([to_cylindrical(pt) for pt in query_all])
    cyl_velocities = np.array([velocity_to_cylindrical(pt, (u, v)) 
                                 for pt, u, v in zip(query_all, U_all, V_all)])
    Xq = cyl_positions[:, 0]
    Yq = cyl_positions[:, 1]
    Zq = cyl_positions[:, 2]
    Uq = cyl_velocities[:, 0]
    Vq = cyl_velocities[:, 1]
    Wq = cyl_velocities[:, 2]
    
    ax3d.quiver(Xq, Yq, Zq, Uq, Vq, Wq, length=0.5, normalize=True,
                color='black', pivot='middle')
    
    # Transform boat positions into cylindrical coordinates.
    boats_rect = boat_positions[frame]
    boats_cyl = np.array([to_cylindrical(pt) for pt in boats_rect])
    ax3d.scatter(boats_cyl[:, 0], boats_cyl[:, 1], boats_cyl[:, 2],
                 s=20, c=boat_colors, depthshade=True)
    
    ax3d.set_title(f"3D Cylindrical View (t={t_current:.2f})")
    return

# ============================
# Create and Save the Animation
# ============================
anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "2d_3d_cylinder.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.show()
