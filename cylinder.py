import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time  # Add import for time module

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)

# Mesh parameters:
# h0: partition size in the coarse mesh (uniform in x and y)
h0 = 0.8    # Coarse mesh grid spacing (both x and y)

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Vorticity, Velocity and Grid
# ---------------------------
def velocity(x, y):
    return np.array([0, -np.sin(2*x)])

def vorticity(x, y):
    return -2*np.cos(2*x)

def generate_uniform_grid():
    """
    Generates a uniform grid covering the entire region with coarse grid spacing h0.
    
    Returns:
      grid: numpy array of shape (n_x, n_y, 2) containing the grid points
      A: corresponding area array
    """
    x1, x2 = region_x
    y1, y2 = region_y

    num_x = int((x2 - x1) / h0) + 1
    num_y = int((y2 - y1) / h0) + 1
    x_vals = np.linspace(x1, x2, num_x, endpoint=False)
    y_vals = np.linspace(y1, y2, num_y, endpoint=False)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    grid = np.stack((xx, yy), axis=-1)
    A = h0 * h0 * np.ones((num_x, num_y))
    
    print(f"Grid shape: {grid.shape}, number of points: {grid.shape[0]*grid.shape[1]}")
    
    return grid, A

# Generate the grid (for vortex initialization)
grid, A = generate_uniform_grid()

# Initialize vortex particles on the grid
def initialize_vortices(grid):
    # grid has shape (n_x, n_y, 2)
    n_x, n_y, _ = grid.shape
    w = np.zeros((n_x, n_y))
    u = np.zeros((n_x, n_y, 2))
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            if -2 <= x < 2 and 0 <= y <= 6:
                w[i, j] = vorticity(x, y)
                u[i, j] = velocity(x, y)
            else:
                w[i, j] = 0.0
                u[i, j] = np.array([0, 0])
    return w, u

w0, u0 = initialize_vortices(grid)

# Assign values to [-6, -2] and [2, 6] by copying from [-2, 2]
def copy_vorticity_velocity(w, u):
    """
    Copies vorticity and velocity from [-2, 2] to [-6, -2] and [2, 6].
    
    Parameters:
      w: numpy array of shape (n_x, n_y) representing the vorticity field.
      u: numpy array of shape (n_x, n_y, 2) representing the velocity field.
      
    Returns:
      w: updated vorticity field with copied values.
      u: updated velocity field with copied values.
    """
    n_x, n_y = w.shape
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            if -2 <= x < 2 and 0 <= y <= 6:
                if -6 <= x - 4 < -2:
                    w[i - int(4 / h0), j] = w[i, j]  # Copy from [-2, 2) to [-6, -2)
                    u[i - int(4 / h0), j] = u[i, j]
                if 2 <= x + 4 < 6:
                    w[i + int(4 / h0), j] = w[i, j]  # Copy from [-2, 2) to [2, 6)
                    u[i + int(4 / h0), j] = u[i, j]
    return w, u

w0, u0 = copy_vorticity_velocity(w0, u0)

print("* Number of vortices in grid:", grid.shape[0]*grid.shape[1])
num_vortices = grid.shape[0]*grid.shape[1]

# Visualize w0
plt.figure(figsize=(10, 8))
plt.imshow(w0.T, extent=(region_x[0], region_x[1], region_y[0], region_y[1]), origin='lower', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Initial Vorticity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# ---------------------------
# Mollified Biot-Savart Kernel for Entire R^2
# ---------------------------
def K_R2(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r = np.linalg.norm(x - y)
    if r < 1e-10:
        return np.zeros(2)
    factor = 1 - np.exp(- (r / delta)**2)
    K1 = - (x2 - y2) / (2 * np.pi * r**2) * factor
    K2 = (x1 - y1) / (2 * np.pi * r**2) * factor
    return np.array([K1, K2])

def indicator_D(point):
    return 0 if point[1] <= 0 else 1

def reflect(point):
    return np.array([point[0], -point[1]])

# ---------------------------
# Function Factory for u_func
# ---------------------------
def make_u_func(current, w0, A):
    """
    Creates a u_func that computes the local velocity (drift) at any position x,
    using a frozen copy of the current vortex positions.
    """
    current_copy = current.copy()
    def u_func(x):
        u_val = np.zeros(2)
        # Loop over the grid (double index)
        n1, n2 = current_copy.shape[0], current_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_copy[i, j]
                contrib1 = indicator_D(pos) * K_R2(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_R2(reflect(pos), x, delta)
                u_val += (contrib1 - contrib2) * w0[i, j] * A[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (using double-index for positions)
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates vortex trajectories using the Euler-Maruyama scheme.
    
    Returns:
      traj: array of shape (num_steps+1, n_x, n_y, 2) representing the trajectories on the grid.
      uFuncs: list of functions (one per time step) that compute the local velocity.
      displacements: array storing the displacement for each vortex.
    """
    # Initialize trajectory array
    traj = np.zeros((num_steps + 1, grid.shape[0], grid.shape[1], 2))
    traj[0] = grid
    
    uFuncs = []
    displacements = np.zeros((num_steps, grid.shape[0], grid.shape[1], 2))
    
    for step in range(num_steps):
        current = traj[step]
        u_func = make_u_func(current, w0, A)
        uFuncs.append(u_func)
        n1, n2 = current.shape[0], current.shape[1]
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                displacements[step, i, j] = disp
                traj[step + 1, i, j] = current[i, j] + disp
        
        # Copy vorticity and velocity from D to D1 and D2
        traj[step + 1], u0 = copy_vorticity_velocity(traj[step + 1], u0)
                
    return traj, uFuncs, displacements

# ---------------------------
# Boat Simulation (using the velocity functions from vortex simulation)
# ---------------------------
def generate_boat_grid():
    """
    Generates a boat grid by flattening the double-index array into a single flat array of query points.
    """
    boat_grid = grid.reshape(-1, 2)
    return boat_grid

def simulate_boats(uFuncs):
    """
    Simulate boat trajectories on the grid.
    Boat positions are updated using the precomputed uFuncs (drift only, no noise).
    
    Returns:
      boat_positions: array of shape (num_steps+1, num_boats, 2)
      boat_displacements: array of shape (num_steps, num_boats, 2)
      boat_colors: array of shape (num_boats,) indicating the color of each boat
    """
    boat_grid = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps + 1, num_boats, 2))
    boat_positions[0] = boat_grid
    boat_displacements = np.zeros((num_steps, num_boats, 2))
    boat_colors = np.full(num_boats, 'blue')  # Default color for boats in [-2, 2) x [0, 6]

    for b in range(num_boats):
        x, y = boat_grid[b]
        if -2 <= x < 2:
            boat_colors[b] = 'red'  # Different color for boats in [-2, 2) x [0, 6]

    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            vel = u_func(boat_positions[step, b])
            boat_displacements[step, b] = dt * vel  # boats follow the drift (no noise)
            boat_positions[step + 1, b] = boat_positions[step, b] + dt * vel
    return boat_positions, boat_displacements, boat_colors

# ---------------------------
# Main Simulation
# ---------------------------
start_time = time.time()  # Start timer for trajectory computation
print("Computing vortex trajectories......")
trajectories, uFuncs, displacements = simulate_vortex_trajectories()
end_time = time.time()  # End timer for trajectory computation
print(f"Time for trajectory computation: {end_time - start_time:.2f} seconds")

start_time = time.time()  # Start timer for boat simulation
print("Simulating vortex boats......")
boat_positions, boat_displacements, boat_colors = simulate_boats(uFuncs)
end_time = time.time()  # End timer for boat simulation
print(f"Time for boat simulation: {end_time - start_time:.2f} seconds")

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

# Plot boat positions as a scatter plot with different colors
boat_scatter = ax.scatter(boat_positions[0, :, 0], boat_positions[0, :, 1],
                          s=10, c=boat_colors, zorder=3)

def update(frame):
    t_current = frame * dt
    # Use the appropriate u_func (if frame exceeds available functions, use the last one)
    u_func = uFuncs[frame if frame < len(uFuncs) else -1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    
    # Update boat positions via scatter plot.
    boat_scatter.set_offsets(boat_positions[frame])
    
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps + 1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "cylinder.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
