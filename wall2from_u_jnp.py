import os
import jax
import jax.numpy as jnp
import numpy as np  # still needed for matplotlib conversion
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

h1 = 1.0
Re = 0.0001 / nu
layer_thickness = 1.0 * jnp.sqrt(Re)
h2_0 = layer_thickness * 0.5  # desired finer layer thickness
h2 = h1 / (h1 // h2_0)

print("boundary layer thickness:", float(layer_thickness))
print("mesh grid: h1 =", h1, ", h2 =", float(h2))

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

# Create an initial PRNG key for JAX random numbers.
key = jax.random.PRNGKey(42)

# ---------------------------
# Vorticity, Velocity and Grid
# ---------------------------
def velocity(x, y):
    return jnp.array([-jnp.sin(y), 0.0])

def vorticity(x, y):
    return -2 * jnp.cos(y)

def generate_nonuniform_grid_D():
    """
    Generates a nonuniform grid in D with a coarse grid over most of the domain 
    and a finer grid near the boundary (y=0). The grids are returned as 2D arrays.
    """
    x1, x2 = region_x
    y1, y2 = region_y
    y3 = float(layer_thickness)

    # Coarse grid: y in [y3, y2] with spacing h1
    num_x_coarse = int((x2 - x1) / h1) + 1
    num_y_coarse = int((y2 - y3) / h1)
    x_coarse = jnp.linspace(x1, x2, num_x_coarse)
    y_coarse = jnp.linspace(y3, y2, num_y_coarse)
    xx_coarse, yy_coarse = jnp.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = jnp.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h1 * h1 * jnp.ones((num_x_coarse, num_y_coarse))
    
    # Fine grid: y in [y1, y3] with spacing h2
    num_x_fine = int((x2 - x1) / h2) + 1
    num_y_fine = int((y3 - y1) / h2) + 1
    x_fine = jnp.linspace(x1, x2, num_x_fine)
    y_fine = jnp.linspace(y1, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = jnp.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = jnp.stack((xx_fine, yy_fine), axis=-1)
    A_fine = h2 * h2 * jnp.ones((num_x_fine, num_y_fine))
    
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
    w = jnp.zeros((n_x, n_y))
    u = jnp.zeros((n_x, n_y, 2))
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            w = w.at[i, j].set(vorticity(x, y))
            u = u.at[i, j].set(velocity(x, y))
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
        return jnp.zeros(2)
    k1 = 0.5 / jnp.pi * ((y2 - x2)/r2 - (y2 + x2)/r2_bar)
    k2 = 0.5 / jnp.pi * ((y1 - x1)/r2_bar - (y1 - x1)/r2)
    factor = 1 - jnp.exp(- (r2 / delta)**2)
    return jnp.array([k1, k2]) * factor

def indicator_D(point):
    # Returns 0 if point[1] <= 0, else 1.
    return jnp.where(point[1] <= 0, 0, 1)

def reflect(point):
    return jnp.array([point[0], -point[1]])

# ---------------------------
# Function Factory for u_func
# ---------------------------
def make_u_func(current_coarse, current_fine, w0_coarse, w0_fine, A_coarse, A_fine):
    current_coarse_copy = jnp.copy(current_coarse)
    current_fine_copy = jnp.copy(current_fine)
    def u_func(x):
        u_val = jnp.zeros(2)
        n1, n2 = current_coarse_copy.shape[0], current_coarse_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val = u_val + (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        n1_f, n2_f = current_fine_copy.shape[0], current_fine_copy.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                pos = current_fine_copy[i, j]
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val = u_val + (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (using double-index for positions)
# ---------------------------
def simulate_vortex_trajectories(key):
    # Initialize trajectory arrays for coarse and fine grids.
    traj_coarse = jnp.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_fine   = jnp.zeros((num_steps + 1, grid_fine.shape[0], grid_fine.shape[1], 2))
    traj_coarse = traj_coarse.at[0].set(grid_coarse)
    traj_fine   = traj_fine.at[0].set(grid_fine)
    
    uFuncs = []
    disp_coarse = jnp.zeros((num_steps, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    disp_fine   = jnp.zeros((num_steps, grid_fine.shape[0], grid_fine.shape[1], 2))
    
    # ----- Pre-compute two initial steps (l=1 and l=2) using velocity field at t=0 -----
    # Step 1:
    u_func_0 = make_u_func(traj_coarse[0], traj_fine[0], w0_coarse, w0_fine, A_coarse, A_fine)
    uFuncs.append(u_func_0)
    n1, n2 = traj_coarse.shape[1], traj_coarse.shape[2]
    for i in range(n1):
        for j in range(n2):
            u_val = u_func_0(traj_coarse[0][i, j])
            key, subkey = jax.random.split(key)
            dW = jnp.sqrt(2 * nu * dt) * jax.random.normal(subkey, shape=(2,))
            disp = dt * u_val + dW
            disp_coarse = disp_coarse.at[0, i, j].set(disp)
            traj_coarse = traj_coarse.at[1, i, j].set(traj_coarse[0, i, j] + disp)
    n1_f, n2_f = traj_fine.shape[1], traj_fine.shape[2]
    for i in range(n1_f):
        for j in range(n2_f):
            u_val = u_func_0(traj_fine[0][i, j])
            key, subkey = jax.random.split(key)
            dW = jnp.sqrt(2 * nu * dt) * jax.random.normal(subkey, shape=(2,))
            disp = dt * u_val + dW
            disp_fine = disp_fine.at[0, i, j].set(disp)
            traj_fine = traj_fine.at[1, i, j].set(traj_fine[0, i, j] + disp)
    
    # Step 2:
    u_func_1 = make_u_func(traj_coarse[1], traj_fine[1], w0_coarse, w0_fine, A_coarse, A_fine)
    uFuncs.append(u_func_1)
    for i in range(n1):
        for j in range(n2):
            u_val = u_func_1(traj_coarse[1][i, j])
            key, subkey = jax.random.split(key)
            dW = jnp.sqrt(2 * nu * dt) * jax.random.normal(subkey, shape=(2,))
            disp = dt * u_val + dW
            disp_coarse = disp_coarse.at[1, i, j].set(disp)
            traj_coarse = traj_coarse.at[2, i, j].set(traj_coarse[1, i, j] + disp)
    for i in range(n1_f):
        for j in range(n2_f):
            u_val = u_func_1(traj_fine[1][i, j])
            key, subkey = jax.random.split(key)
            dW = jnp.sqrt(2 * nu * dt) * jax.random.normal(subkey, shape=(2,))
            disp = dt * u_val + dW
            disp_fine = disp_fine.at[1, i, j].set(disp)
            traj_fine = traj_fine.at[2, i, j].set(traj_fine[1, i, j] + disp)
    
    # ----- Continue simulation from l = 2 to num_steps -----
    for step in range(2, num_steps):
        current_coarse = traj_coarse[step]
        current_fine = traj_fine[step]
        u_func = make_u_func(current_coarse, current_fine, w0_coarse, w0_fine, A_coarse, A_fine)
        uFuncs.append(u_func)
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                key, subkey = jax.random.split(key)
                dW = jnp.sqrt(2 * nu * dt) * jax.random.normal(subkey, shape=(2,))
                disp = dt * u_val + dW
                disp_coarse = disp_coarse.at[step, i, j].set(disp)
                traj_coarse = traj_coarse.at[step+1, i, j].set(current_coarse[i, j] + disp)
        for i in range(n1_f):
            for j in range(n2_f):
                u_val = u_func(current_fine[i, j])
                key, subkey = jax.random.split(key)
                dW = jnp.sqrt(2 * nu * dt) * jax.random.normal(subkey, shape=(2,))
                disp = dt * u_val + dW
                disp_fine = disp_fine.at[step, i, j].set(disp)
                traj_fine = traj_fine.at[step+1, i, j].set(current_fine[i, j] + disp)
                
    return (traj_coarse, traj_fine), uFuncs, (disp_coarse, disp_fine), key

print("Computing vortex trajectories......")
(trajectories_coarse, trajectories_fine), uFuncs, (disp_coarse, disp_fine), key = simulate_vortex_trajectories(key)

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
    boat_grid = jnp.concatenate((grid_coarse.reshape(-1, 2), grid_fine.reshape(-1, 2)), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    """
    Simulate boat trajectories on the grid.
    Boat positions are updated using the precomputed uFuncs (drift only, no noise).
    """
    boat_grid = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = jnp.zeros((num_steps + 1, num_boats, 2))
    boat_positions = boat_positions.at[0].set(boat_grid)
    boat_displacements = jnp.zeros((num_steps, num_boats, 2))
    
    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            vel = u_func(boat_positions[step, b])
            boat_displacements = boat_displacements.at[step, b].set(dt * vel)
            boat_positions = boat_positions.at[step+1, b].set(boat_positions[step, b] + dt * vel)
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
    U = jnp.zeros(P)
    V = jnp.zeros(P)
    for p in range(P):
        vel = u_func(query_points[p])
        U = U.at[p].set(vel[0])
        V = V.at[p].set(vel[1])
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
# Convert JAX arrays to NumPy for matplotlib.
U_np = np.array(U)
V_np = np.array(V)
query_grid_np = np.array(query_grid)
vel_quiver = ax.quiver(query_grid_np[:, 0], query_grid_np[:, 1], U_np, V_np,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

# Plot boat positions as a scatter plot.
boat_positions_np = np.array(boat_positions[0])
boat_scatter = ax.scatter(boat_positions_np[:, 0], boat_positions_np[:, 1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    # Use the appropriate u_func (if frame exceeds available functions, use the last one)
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    U, V = compute_velocity_field(u_func, query_grid)
    U_np = np.array(U)
    V_np = np.array(V)
    vel_quiver.set_UVC(U_np, V_np)
    
    boat_pos = np.array(boat_positions[frame])
    boat_scatter.set_offsets(boat_pos)
    
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps + 1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex7.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
