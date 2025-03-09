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
hd = 0.4              # Dense spacing for region D

# Adjusted simulation region: x in [-6,6] and y in [0,4].
region_x = [-6, 6]
region_y = [0, 4]
window_x = [-6, 6]    # Domain D in x
window_y = [0, 4]     # Domain D in y

np.random.seed(42)

# ---------------------------
# Basic Physics Functions
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(2*y), 0])

def vorticity(x, y):
    return np.cos(2*y)

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
    In all cases, y runs from 0 to 4.
    Points on the right and top boundaries are removed using endpoint=False.
    Returns:
      grid_D, A_D, grid_D1, A_D1, grid_D2, A_D2.
    """
    # For region D:
    num_x_D = int((2 - (-2)) / hd)
    num_y_D = int((4 - 0) / hd)
    x_D = np.linspace(-2, 2, num_x_D, endpoint=False)
    y_D = np.linspace(0, 4, num_y_D, endpoint=False)
    xx_D, yy_D = np.meshgrid(x_D, y_D, indexing='ij')
    grid_D = np.stack((xx_D, yy_D), axis=-1)
    A_D = (hd**2) * np.ones(xx_D.shape)
    
    # For region D₁ (coarse):
    num_x_D1 = int(( -2 - (-6) ) / h0)
    num_y_D1 = int((4 - 0) / h0)
    x_D1 = np.linspace(-6, -2, num_x_D1, endpoint=False)
    y_D1 = np.linspace(0, 4, num_y_D1, endpoint=False)
    xx_D1, yy_D1 = np.meshgrid(x_D1, y_D1, indexing='ij')
    grid_D1 = np.stack((xx_D1, yy_D1), axis=-1)
    A_D1 = (h0**2) * np.ones(xx_D1.shape)
    
    # For region D₂ (coarse):
    num_x_D2 = int((6 - 2) / h0)
    num_y_D2 = int((4 - 0) / h0)
    x_D2 = np.linspace(2, 6, num_x_D2, endpoint=False)
    y_D2 = np.linspace(0, 4, num_y_D2, endpoint=False)
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

# Copy initial vorticity from D to D₁ and D₂ (with sign flip for Mobius)
def copy_initial_vorticity(grid_D, w_D, grid_D1, w_D1, grid_D2, w_D2):
    """
    For each point in D₁ and D₂, set its vorticity to the NEGATIVE of that
    in D (with an appropriate spatial shift). The mapping uses the fact that h0 = 2*hd.
    """
    for i in range(grid_D1.shape[0]):
        for j in range(grid_D1.shape[1]):
            x, y = grid_D1[i, j]
            i_dense = int(round((x + 4 - (-2)) / hd))
            j_dense = j * int(h0/hd)
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D1[i, j] = -w_D[i_dense, j_dense]
    for i in range(grid_D2.shape[0]):
        for j in range(grid_D2.shape[1]):
            x, y = grid_D2[i, j]
            i_dense = int(round((x - 4 - (-2)) / hd))
            j_dense = j * int(h0/hd)
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D2[i, j] = -w_D[i_dense, j_dense]
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
# Vortex Trajectory Simulation (with twisted periodicity)
# ---------------------------
def simulate_vortex_trajectories():
    """
    Evolves vortex trajectories using Euler–Maruyama.
    The simulation state is maintained separately for regions D, D₁, and D₂.
    After each step, the trajectories in D₁ and D₂ are updated by copying from D,
    using a twisted re-entry mapping: (x,y) -> (x±4, 4-y).
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
        # Update region D with twisted boundary conditions
        for i in range(traj_D[step].shape[0]):
            for j in range(traj_D[step].shape[1]):
                pos = traj_D[step, i, j]
                new_pos = pos + dt * u_func(pos)
                # Apply twisted periodic BC on D:
                if new_pos[0] >= 2:
                    new_pos = np.array([new_pos[0] - 4, 4 - new_pos[1]])
                elif new_pos[0] < -2:
                    new_pos = np.array([new_pos[0] + 4, 4 - new_pos[1]])
                traj_D[step+1, i, j] = new_pos
        # Update regions D₁ and D₂ from D using the twisted mapping.
        for i in range(traj_D1[0].shape[0]):
            for j in range(traj_D1[0].shape[1]):
                x, y = grid_D1[i, j]
                i_dense = int(round((x + 6) / hd))
                j_dense = j * 2
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    pos_dense = traj_D[step+1, i_dense, j_dense]
                    # For D₁: shift left by 4 and twist y
                    traj_D1[step+1, i, j] = np.array([pos_dense[0] - 4, 4 - pos_dense[1]])
        for i in range(traj_D2[0].shape[0]):
            for j in range(traj_D2[0].shape[1]):
                x, y = grid_D2[i, j]
                i_dense = int(round((x - 2) / hd))
                j_dense = j * 2
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    pos_dense = traj_D[step+1, i_dense, j_dense]
                    # For D₂: shift right by 4 and twist y
                    traj_D2[step+1, i, j] = np.array([pos_dense[0] + 4, 4 - pos_dense[1]])
                    
    return (traj_D, traj_D1, traj_D2), uFuncs

(simTraj_D, simTraj_D1, simTraj_D2), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation (only for region D with twisted periodic boundary conditions)
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
            # Apply twisted periodic re-entry:
            if new_pos[0] >= 2:
                new_pos = np.array([new_pos[0] - 4, 4 - new_pos[1]])
            elif new_pos[0] < -2:
                new_pos = np.array([new_pos[0] + 4, 4 - new_pos[1]])
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
    num_y_D = int((4 - 0) / hd)
    x_D = np.linspace(-2, 2, num_x_D, endpoint=False)
    y_D = np.linspace(0, 4, num_y_D, endpoint=False)
    xx_D, yy_D = np.meshgrid(x_D, y_D, indexing='ij')
    query_D = np.stack((xx_D, yy_D), axis=-1).reshape(-1, 2)
    
    num_x_D1 = int(( -2 - (-6) ) / h0)
    num_y_D1 = int((4 - 0) / h0)
    x_D1 = np.linspace(-6, -2, num_x_D1, endpoint=False)
    y_D1 = np.linspace(0, 4, num_y_D1, endpoint=False)
    xx_D1, yy_D1 = np.meshgrid(x_D1, y_D1, indexing='ij')
    query_D1 = np.stack((xx_D1, yy_D1), axis=-1).reshape(-1, 2)
    
    num_x_D2 = int((6 - 2) / h0)
    num_y_D2 = int((4 - 0) / h0)
    x_D2 = np.linspace(2, 6, num_x_D2, endpoint=False)
    y_D2 = np.linspace(0, 4, num_y_D2, endpoint=False)
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
# Mobius Strip Mapping Functions for 3D Visualization
# ============================
def to_mobius(point):
    """
    Maps a point (x,y) in D (with x in [-2,2] and y in [0,4])
    onto a Mobius strip. The mapping is chosen so that the left boundary (-2,y)
    is identified with the right boundary (2, 4-y).
    """
    x, y = point
    # Define u so that x=-2 -> u=0 and x=2 -> u=2*pi.
    u = np.pi * (x + 2) / 2.0  # u in [0,2*pi)
    # Use a scaled version of y for the "width" parameter; here y=2 is the midline.
    t = (y - 2) / 4.0
    # Standard Mobius strip parameterization:
    X = (1 + t * np.cos(u/2)) * np.cos(u)
    Y = (1 + t * np.cos(u/2)) * np.sin(u)
    Z = t * np.sin(u/2)
    return np.array([X, Y, Z])

def velocity_to_mobius(point, vel):
    """
    Transforms a 2D velocity vector at 'point' from the simulation domain
    into the corresponding 3D velocity on the Mobius strip.
    This is computed using the Jacobian of the mapping F(x,y) = (X,Y,Z) given in to_mobius.
    """
    x, y = point
    vx, vy = vel
    u = np.pi*(x+2)/2.0
    t = (y - 2)/4.0
    # Partial derivatives with respect to x (note: du/dx = pi/2)
    dXdx = - (np.pi*t/4) * np.sin(u/2)*np.cos(u) - (np.pi/2)*(1 + t*np.cos(u/2))*np.sin(u)
    dYdx = - (np.pi*t/4) * np.sin(u/2)*np.sin(u) + (np.pi/2)*(1 + t*np.cos(u/2))*np.cos(u)
    dZdx = (np.pi*t/4)* np.cos(u/2)
    # Partial derivatives with respect to y (dt/dy = 1/4)
    dXdy = (np.cos(u/2)*np.cos(u))/4.0
    dYdy = (np.cos(u/2)*np.sin(u))/4.0
    dZdy = np.sin(u/2)/4.0
    v3d = np.array([dXdx*vx + dXdy*vy,
                    dYdx*vx + dYdy*vy,
                    dZdx*vx + dZdy*vy])
    return v3d

# ============================
# Prepare the Mobius Surface for 3D Plotting
# ============================
# Create a grid in the simulation (x,y) domain and map it to the Mobius strip.
x_vals = np.linspace(-2, 2, 50)
y_vals = np.linspace(0, 4, 50)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')
mobius_points = np.array([to_mobius((x, y)) for x, y in zip(X_mesh.flatten(), Y_mesh.flatten())])
X_mob = mobius_points[:,0].reshape(X_mesh.shape)
Y_mob = mobius_points[:,1].reshape(X_mesh.shape)
Z_mob = mobius_points[:,2].reshape(X_mesh.shape)

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

# For 3D view, plot the Mobius surface.
def init_3d(ax):
    ax.plot_surface(X_mob, Y_mob, Z_mob, alpha=0.2, color='cyan', rstride=2, cstride=2, edgecolor='none')
    # Set limits that roughly center the Mobius strip.
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1.5)
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
    
    # Transform velocity query points and vectors into Mobius coordinates.
    mobius_positions = np.array([to_mobius(pt) for pt in query_all])
    mobius_velocities = np.array([velocity_to_mobius(pt, (u, v)) 
                                  for pt, u, v in zip(query_all, U_all, V_all)])
    Xq = mobius_positions[:, 0]
    Yq = mobius_positions[:, 1]
    Zq = mobius_positions[:, 2]
    Uq = mobius_velocities[:, 0]
    Vq = mobius_velocities[:, 1]
    Wq = mobius_velocities[:, 2]
    
    ax3d.quiver(Xq, Yq, Zq, Uq, Vq, Wq, length=0.5, normalize=True,
                color='black', pivot='middle')
    
    # Transform boat positions into Mobius coordinates.
    boats_rect = boat_positions[frame]
    boats_mobius = np.array([to_mobius(pt) for pt in boats_rect])
    ax3d.scatter(boats_mobius[:, 0], boats_mobius[:, 1], boats_mobius[:, 2],
                 s=20, c=boat_colors, depthshade=True)
    
    ax3d.set_title(f"3D Mobius Strip View (t={t_current:.2f})")
    return

# ============================
# Create and Save the Animation
# ============================
anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "2d_3d_mobius.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.show()
