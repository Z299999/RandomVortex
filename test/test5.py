import os
import numpy as np
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

# Mesh parameters:
h0 = 1.0    # Coarse mesh grid spacing (both x and y)
h1 = 0.5    # Fine mesh grid spacing in the x direction
h2 = 0.1    # Fine mesh grid spacing in the y direction

Re = 0.0001 / nu
layer_thickness = 1 * np.sqrt(Re)

region_x = [-6, 6]
region_y = [0, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

np.random.seed(42)

# ---------------------------
# Vorticity, Velocity and Grid
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return np.cos(y)

def generate_nonuniform_grid_D():
    x1, x2 = region_x
    y1, y2 = region_y
    y3 = layer_thickness

    # Coarse grid: y in [y3, y2] with spacing h0
    num_x_coarse = int((x2 - x1) / h0) + 1
    num_y_coarse = int((y2 - y3) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))
    
    # Fine grid: y in [y1, y3] with spacing h1 (x) and h2 (y)
    num_x_fine = int((x2 - x1) / h1) + 1
    num_y_fine = int((y3 - y1) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y1, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = np.stack((xx_fine, yy_fine), axis=-1)
    A_fine = h1 * h2 * np.ones((num_x_fine, num_y_fine))
    
    print(f"Coarse grid shape: {grid_coarse.shape}, number of points: {grid_coarse.shape[0]*grid_coarse.shape[1]}")
    print(f"Fine grid shape: {grid_fine.shape}, number of points: {grid_fine.shape[0]*grid_fine.shape[1]}")
    
    return {'coarse': (grid_coarse, A_coarse), 'fine': (grid_fine, A_fine)}

grids = generate_nonuniform_grid_D()
grid_coarse, A_coarse = grids['coarse']
grid_fine, A_fine = grids['fine']

def initialize_vortices(grid):
    n_x, n_y, _ = grid.shape
    w = np.zeros((n_x, n_y))
    u = np.zeros((n_x, n_y, 2))
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y)
            u[i, j] = velocity(x, y)
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
        return np.zeros(2)
    k1 = 0.5 / np.pi * ((y2 - x2)/r2 - (y2 + x2)/r2_bar)
    k2 = 0.5 / np.pi * ((y1 - x1)/r2_bar - (y1 - x1)/r2)
    factor = 1 - np.exp(- (r2 / delta)**2)
    return np.array([k1, k2]) * factor

def indicator_D(point):
    return 0 if point[1] <= 0 else 1

def reflect(point):
    return np.array([point[0], -point[1]])

def make_u_func(current_coarse, current_fine, w0_coarse, w0_fine, A_coarse, A_fine):
    # Copy current positions for evaluation
    current_coarse_copy = current_coarse.copy()
    current_fine_copy = current_fine.copy()
    def u_func(x):
        u_val = np.zeros(2)
        n1, n2 = current_coarse_copy.shape[0], current_coarse_copy.shape[1]
        for i in range(n1):
            for j in range(n2):
                pos = current_coarse_copy[i, j]
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val += (contrib1 - contrib2) * w0_coarse[i, j] * A_coarse[i, j]
        n1_f, n2_f = current_fine_copy.shape[0], current_fine_copy.shape[1]
        for i in range(n1_f):
            for j in range(n2_f):
                pos = current_fine_copy[i, j]
                contrib1 = indicator_D(pos) * K_delta(pos, x, delta)
                contrib2 = indicator_D(reflect(pos)) * K_delta(reflect(pos), x, delta)
                u_val += (contrib1 - contrib2) * w0_fine[i, j] * A_fine[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Direct Update)
# ---------------------------
def simulate_vortex_trajectories():
    # Allocate arrays for trajectories
    traj_coarse = np.zeros((num_steps + 1, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    traj_fine = np.zeros((num_steps + 1, grid_fine.shape[0], grid_fine.shape[1], 2))
    traj_coarse[0] = grid_coarse
    traj_fine[0] = grid_fine
    
    uFuncs = []
    disp_coarse = np.zeros((num_steps, grid_coarse.shape[0], grid_coarse.shape[1], 2))
    disp_fine = np.zeros((num_steps, grid_fine.shape[0], grid_fine.shape[1], 2))
    
    n1, n2 = grid_coarse.shape[0], grid_coarse.shape[1]
    n1_f, n2_f = grid_fine.shape[0], grid_fine.shape[1]
    
    # Directly update trajectories for each time step
    for step in range(num_steps):
        current_coarse = traj_coarse[step]
        current_fine = traj_fine[step]
        # Compute the velocity function based on current positions
        u_func = make_u_func(current_coarse, current_fine, w0_coarse, w0_fine, A_coarse, A_fine)
        uFuncs.append(u_func)
        # Update coarse grid vortices
        for i in range(n1):
            for j in range(n2):
                u_val = u_func(current_coarse[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_coarse[step, i, j] = disp
                traj_coarse[step + 1, i, j] = current_coarse[i, j] + disp
        # Update fine grid vortices
        for i in range(n1_f):
            for j in range(n2_f):
                u_val = u_func(current_fine[i, j])
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                disp = dt * u_val + dW
                disp_fine[step, i, j] = disp
                traj_fine[step + 1, i, j] = current_fine[i, j] + disp
                
    return (traj_coarse, traj_fine), uFuncs, (disp_coarse, disp_fine)

print("Computing vortex trajectories......")
(trajectories_coarse, trajectories_fine), uFuncs, (disp_coarse, disp_fine) = simulate_vortex_trajectories()


u_func = uFuncs[0]
test_points = np.array([[0.0, 0.05], [0.0, 0.1], [0.0, 0.15], [0.0, 0.2], [0.0, 0.3]])  # A point very near the boundary
for test_point in test_points:
    u_val = u_func(test_point)
    print("At", test_point, "direct contribution:", indicator_D(test_point) * K_delta(test_point, test_point, delta))
    print("At", test_point, "reflected contribution:", indicator_D(reflect(test_point)) * K_delta(reflect(test_point), test_point, delta))
    print("Total u:", u_val)

# # ---------------------------
# # Animate Vortex Trajectories
# # ---------------------------
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_xlim(window_x)
# ax.set_ylim(window_y)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Vortex Trajectories')
# ax.grid(True)

# # Create scatter objects for coarse and fine grid vortices
# coarse_scatter = ax.scatter([], [], color='red', s=10, label='Coarse Grid')
# fine_scatter = ax.scatter([], [], color='blue', s=10, label='Fine Grid')
# ax.legend()

# def init():
#     coarse_scatter.set_offsets(np.empty((0, 2)))
#     fine_scatter.set_offsets(np.empty((0, 2)))
#     return coarse_scatter, fine_scatter

# def update(frame):
#     coarse_positions = trajectories_coarse[frame].reshape(-1, 2)
#     fine_positions = trajectories_fine[frame].reshape(-1, 2)
#     coarse_scatter.set_offsets(coarse_positions)
#     fine_scatter.set_offsets(fine_positions)
#     ax.set_title(f'Vortex Trajectories at t = {frame*dt:.2f}s')
#     return coarse_scatter, fine_scatter

# frames = trajectories_coarse.shape[0]
# ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

# os.makedirs('animation', exist_ok=True)
# writer = FFMpegWriter(fps=25)
# ani.save(os.path.join('animation', 'vortex_animation.mp4'), writer=writer)

# plt.close(fig)
