import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import time

# ============================
# Simulation Parameters
# ============================
nu = 0.000          # Viscosity (here zero so no noise)
T = 15              # Final time (seconds)
dt = 0.04           # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter

# Mesh spacings:
h0 = 1.0            # Coarse spacing for regions D₁, D₂, D₃, D₄
hd = 0.5            # Dense spacing for region D

# For Klein bottle simulation, we take the fundamental domain D as:
#   x in [-2,2) and y in [0,4)
# and attach four coarse copies:
#   D₁: left:   x in [-6,-2), y in [0,4)
#   D₂: right:  x in [2,6),   y in [0,4)
#   D₃: bottom: x in [-2,2),  y in [-4,0)
#   D₄: top:    x in [-2,2),  y in [4,8)
window_x = [-6, 6]
window_y = [-4, 8]

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
# Grid Generator for Klein Bottle Simulation
# ---------------------------
def generate_simulation_grid():
    """
    Generates five grids for Klein bottle simulation:
      - D: central region, x in [-2,2) and y in [0,4) with dense spacing hd.
      - D₁: left region, x in [-6,-2) and y in [0,4) with coarse spacing h0.
      - D₂: right region, x in [2,6) and y in [0,4) with coarse spacing h0.
      - D₃: bottom region, x in [-2,2) and y in [-4,0) with coarse spacing h0.
      - D₄: top region, x in [-2,2) and y in [4,8) with coarse spacing h0.
    """
    # Region D (dense):
    num_x_D = int((2 - (-2)) / hd)    # 4/0.5 = 8
    num_y_D = int((4 - 0) / hd)         # 4/0.5 = 8
    x_D = np.linspace(-2, 2, num_x_D, endpoint=False)
    y_D = np.linspace(0, 4, num_y_D, endpoint=False)
    xx_D, yy_D = np.meshgrid(x_D, y_D, indexing='ij')
    grid_D = np.stack((xx_D, yy_D), axis=-1)
    A_D = (hd**2) * np.ones(xx_D.shape)
    
    # Region D₁ (left, coarse):
    num_x_D1 = int(( -2 - (-6) ) / h0)  # 4/1 = 4
    num_y_D1 = int((4 - 0) / h0)         # 4/1 = 4
    x_D1 = np.linspace(-6, -2, num_x_D1, endpoint=False)
    y_D1 = np.linspace(0, 4, num_y_D1, endpoint=False)
    xx_D1, yy_D1 = np.meshgrid(x_D1, y_D1, indexing='ij')
    grid_D1 = np.stack((xx_D1, yy_D1), axis=-1)
    A_D1 = (h0**2) * np.ones(xx_D1.shape)
    
    # Region D₂ (right, coarse):
    num_x_D2 = int((6 - 2) / h0)       # 4/1 = 4
    num_y_D2 = int((4 - 0) / h0)         # 4/1 = 4
    x_D2 = np.linspace(2, 6, num_x_D2, endpoint=False)
    y_D2 = np.linspace(0, 4, num_y_D2, endpoint=False)
    xx_D2, yy_D2 = np.meshgrid(x_D2, y_D2, indexing='ij')
    grid_D2 = np.stack((xx_D2, yy_D2), axis=-1)
    A_D2 = (h0**2) * np.ones(xx_D2.shape)
    
    # Region D₃ (bottom, coarse):
    num_x_D3 = int((2 - (-2)) / h0)    # 4/1 = 4
    num_y_D3 = int((0 - (-4)) / h0)      # 4/1 = 4
    x_D3 = np.linspace(-2, 2, num_x_D3, endpoint=False)
    y_D3 = np.linspace(-4, 0, num_y_D3, endpoint=False)
    xx_D3, yy_D3 = np.meshgrid(x_D3, y_D3, indexing='ij')
    grid_D3 = np.stack((xx_D3, yy_D3), axis=-1)
    A_D3 = (h0**2) * np.ones(xx_D3.shape)
    
    # Region D₄ (top, coarse):
    num_x_D4 = int((2 - (-2)) / h0)    # 4/1 = 4
    num_y_D4 = int((8 - 4) / h0)         # 4/1 = 4
    x_D4 = np.linspace(-2, 2, num_x_D4, endpoint=False)
    y_D4 = np.linspace(4, 8, num_y_D4, endpoint=False)
    xx_D4, yy_D4 = np.meshgrid(x_D4, y_D4, indexing='ij')
    grid_D4 = np.stack((xx_D4, yy_D4), axis=-1)
    A_D4 = (h0**2) * np.ones(xx_D4.shape)
    
    print(f"Region D: {grid_D.size//2} points, D₁: {grid_D1.size//2} points, D₂: {grid_D2.size//2} points, D₃: {grid_D3.size//2} points, D₄: {grid_D4.size//2} points")
    return grid_D, A_D, grid_D1, A_D1, grid_D2, A_D2, grid_D3, A_D3, grid_D4, A_D4

grid_D, A_D, grid_D1, A_D1, grid_D2, A_D2, grid_D3, A_D3, grid_D4, A_D4 = generate_simulation_grid()

# ---------------------------
# Vortex Initialization
# ---------------------------
def initialize_vortices(grid_D, grid_D1, grid_D2, grid_D3, grid_D4):
    # For region D, initialize using the vorticity function.
    nD = grid_D.shape[:2]
    w_D = np.zeros(nD)
    u_D = np.zeros(grid_D.shape)
    for i in range(nD[0]):
        for j in range(nD[1]):
            x, y = grid_D[i, j]
            w_D[i, j] = vorticity(x, y)
            u_D[i, j] = velocity(x, y)
            
    # For regions D₁, D₂, D₃, D₄, initialize with zeros.
    nD1 = grid_D1.shape[:2]
    w_D1 = np.zeros(nD1)
    u_D1 = np.zeros(grid_D1.shape)
    
    nD2 = grid_D2.shape[:2]
    w_D2 = np.zeros(nD2)
    u_D2 = np.zeros(grid_D2.shape)
    
    nD3 = grid_D3.shape[:2]
    w_D3 = np.zeros(nD3)
    u_D3 = np.zeros(grid_D3.shape)
    
    nD4 = grid_D4.shape[:2]
    w_D4 = np.zeros(nD4)
    u_D4 = np.zeros(grid_D4.shape)
    
    return (w_D, u_D), (w_D1, u_D1), (w_D2, u_D2), (w_D3, u_D3), (w_D4, u_D4)

(w_D, u_D), (w_D1, u_D1), (w_D2, u_D2), (w_D3, u_D3), (w_D4, u_D4) = initialize_vortices(grid_D, grid_D1, grid_D2, grid_D3, grid_D4)

# ---------------------------
# Copy Initial Vorticity from D to Other Regions
# (Modifications for Klein bottle: D₁, D₂ same as D; D₃, D₄ are opposite sign and twisted mapping)
# ---------------------------
def copy_initial_vorticity(grid_D, w_D, grid_D1, w_D1, grid_D2, w_D2, grid_D3, w_D3, grid_D4, w_D4):
    # For D₁ (left): map (x,y) -> (x+4, y)
    for i in range(grid_D1.shape[0]):
        for j in range(grid_D1.shape[1]):
            x, y = grid_D1[i, j]
            x_mapped = x + 4
            i_dense = int(round((x_mapped - (-2)) / hd))
            j_dense = j * int(h0/hd)
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D1[i, j] = w_D[i_dense, j_dense]
    # For D₂ (right): map (x,y) -> (x-4, y)
    for i in range(grid_D2.shape[0]):
        for j in range(grid_D2.shape[1]):
            x, y = grid_D2[i, j]
            x_mapped = x - 4
            i_dense = int(round((x_mapped - (-2)) / hd))
            j_dense = j * int(h0/hd)
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D2[i, j] = w_D[i_dense, j_dense]
    # For D₃ (bottom, twisted): map (x,y) -> (-x, y+4) and take opposite vorticity.
    for i in range(grid_D3.shape[0]):
        for j in range(grid_D3.shape[1]):
            x, y = grid_D3[i, j]
            x_mapped = -x
            y_mapped = y + 4
            i_dense = int(round((x_mapped - (-2)) / hd))
            j_dense = int(round((y_mapped - 0) / hd))
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D3[i, j] = - w_D[i_dense, j_dense]
    # For D₄ (top, twisted): map (x,y) -> (-x, y-4) and take opposite vorticity.
    for i in range(grid_D4.shape[0]):
        for j in range(grid_D4.shape[1]):
            x, y = grid_D4[i, j]
            x_mapped = -x
            y_mapped = y - 4
            i_dense = int(round((x_mapped - (-2)) / hd))
            j_dense = int(round((y_mapped - 0) / hd))
            if i_dense < grid_D.shape[0] and j_dense < grid_D.shape[1]:
                w_D4[i, j] = - w_D[i_dense, j_dense]
    return w_D1, w_D2, w_D3, w_D4

w_D1, w_D2, w_D3, w_D4 = copy_initial_vorticity(grid_D, w_D, grid_D1, w_D1, grid_D2, w_D2, grid_D3, w_D3, grid_D4, w_D4)

# ---------------------------
# Velocity Function Factory for the Combined State
# ---------------------------
def make_u_func_sim(current_state, w_state, A_state):
    traj_D, traj_D1, traj_D2, traj_D3, traj_D4 = current_state
    w_D, w_D1, w_D2, w_D3, w_D4 = w_state
    A_D, A_D1, A_D2, A_D3, A_D4 = A_state
    def u_func(x):
        u_val = np.zeros(2)
        # Sum contributions from all regions:
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
        for i in range(traj_D3.shape[0]):
            for j in range(traj_D3.shape[1]):
                pos = traj_D3[i, j]
                u_val += K_R2(pos, x, delta) * w_D3[i, j] * A_D3[i, j]
        for i in range(traj_D4.shape[0]):
            for j in range(traj_D4.shape[1]):
                pos = traj_D4[i, j]
                u_val += K_R2(pos, x, delta) * w_D4[i, j] * A_D4[i, j]
        return u_val
    return u_func

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    traj_D = np.zeros((num_steps+1,) + grid_D.shape)
    traj_D1 = np.zeros((num_steps+1,) + grid_D1.shape)
    traj_D2 = np.zeros((num_steps+1,) + grid_D2.shape)
    traj_D3 = np.zeros((num_steps+1,) + grid_D3.shape)
    traj_D4 = np.zeros((num_steps+1,) + grid_D4.shape)
    
    traj_D[0] = grid_D
    traj_D1[0] = grid_D1
    traj_D2[0] = grid_D2
    traj_D3[0] = grid_D3
    traj_D4[0] = grid_D4
    
    uFuncs = []
    w_state = (w_D, w_D1, w_D2, w_D3, w_D4)
    A_state = (A_D, A_D1, A_D2, A_D3, A_D4)
    
    for step in range(num_steps):
        current_state = (traj_D[step], traj_D1[step], traj_D2[step], traj_D3[step], traj_D4[step])
        u_func = make_u_func_sim(current_state, w_state, A_state)
        uFuncs.append(u_func)
        
        # Update trajectories for region D:
        for i in range(traj_D[step].shape[0]):
            for j in range(traj_D[step].shape[1]):
                pos = traj_D[step, i, j]
                traj_D[step+1, i, j] = pos + dt * u_func(pos)
                
        # Update trajectories for the coarse regions:
        for region in (traj_D1, traj_D2, traj_D3, traj_D4):
            for i in range(region[step].shape[0]):
                for j in range(region[step].shape[1]):
                    pos = region[step, i, j]
                    region[step+1, i, j] = pos + dt * u_func(pos)
                    
        # Copy/update D₁ from D (shift left by 4):
        for i in range(traj_D1[0].shape[0]):
            for j in range(traj_D1[0].shape[1]):
                x, y = grid_D1[i, j]
                # Mapping: corresponding point in D is (x+4, y)
                i_dense = int(round((x + 4 - (-2)) / hd))
                j_dense = j * int(h0/hd)
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    traj_D1[step+1, i, j] = traj_D[step+1, i_dense, j_dense] - np.array([4, 0])
        # Copy/update D₂ from D (shift right by 4):
        for i in range(traj_D2[0].shape[0]):
            for j in range(traj_D2[0].shape[1]):
                x, y = grid_D2[i, j]
                i_dense = int(round((x - 4 - (-2)) / hd))
                j_dense = j * int(h0/hd)
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    traj_D2[step+1, i, j] = traj_D[step+1, i_dense, j_dense] + np.array([4, 0])
        # Copy/update D₃ from D (bottom copy, twisted identification):
        for i in range(traj_D3[0].shape[0]):
            for j in range(traj_D3[0].shape[1]):
                x, y = grid_D3[i, j]
                # Mapping for Klein bottle: (x,y) in D₃ corresponds to (-x, y+4) in D.
                x_mapped = -x
                y_mapped = y + 4
                i_dense = int(round((x_mapped - (-2)) / hd))
                j_dense = int(round((y_mapped - 0) / hd))
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    new_pos = traj_D[step+1, i_dense, j_dense]
                    traj_D3[step+1, i, j] = np.array([-new_pos[0], new_pos[1] - 4])
        # Copy/update D₄ from D (top copy, twisted identification):
        for i in range(traj_D4[0].shape[0]):
            for j in range(traj_D4[0].shape[1]):
                x, y = grid_D4[i, j]
                # Mapping for Klein bottle: (x,y) in D₄ corresponds to (-x, y-4) in D.
                x_mapped = -x
                y_mapped = y - 4
                i_dense = int(round((x_mapped - (-2)) / hd))
                j_dense = int(round((y_mapped - 0) / hd))
                if i_dense < traj_D[step+1].shape[0] and j_dense < traj_D[step+1].shape[1]:
                    new_pos = traj_D[step+1, i_dense, j_dense]
                    traj_D4[step+1, i, j] = np.array([-new_pos[0], new_pos[1] + 4])
                    
    return (traj_D, traj_D1, traj_D2, traj_D3, traj_D4), uFuncs

(simTraj_D, simTraj_D1, simTraj_D2, simTraj_D3, simTraj_D4), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation on the Klein Bottle (for region D with twisted boundary conditions)
# ---------------------------
def generate_boat_grid():
    """Generates a boat grid using only the central region D."""
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
            # Apply periodic boundaries in x:
            if new_pos[0] >= 2:
                new_pos[0] -= 4
            elif new_pos[0] < -2:
                new_pos[0] += 4
            # Apply twisted boundaries in y for Klein bottle:
            if new_pos[1] >= 4:
                new_pos[1] -= 4
                new_pos[0] = -new_pos[0]
            elif new_pos[1] < 0:
                new_pos[1] += 4
                new_pos[0] = -new_pos[0]
            boat_positions[step+1, b] = new_pos
    return boat_positions, boat_colors

boat_positions, boat_colors = simulate_boats(uFuncs)

# ---------------------------
# Velocity Query Grids for Visualization (for all regions)
# ---------------------------
def generate_velocity_query_grids():
    # Region D (dense):
    num_x_D = int((2 - (-2)) / hd)
    num_y_D = int((4 - 0) / hd)
    x_D = np.linspace(-2, 2, num_x_D, endpoint=False)
    y_D = np.linspace(0, 4, num_y_D, endpoint=False)
    xx_D, yy_D = np.meshgrid(x_D, y_D, indexing='ij')
    query_D = np.stack((xx_D, yy_D), axis=-1).reshape(-1, 2)
    
    # Region D₁ (left, coarse):
    num_x_D1 = int(( -2 - (-6) ) / h0)
    num_y_D1 = int((4 - 0) / h0)
    x_D1 = np.linspace(-6, -2, num_x_D1, endpoint=False)
    y_D1 = np.linspace(0, 4, num_y_D1, endpoint=False)
    xx_D1, yy_D1 = np.meshgrid(x_D1, y_D1, indexing='ij')
    query_D1 = np.stack((xx_D1, yy_D1), axis=-1).reshape(-1, 2)
    
    # Region D₂ (right, coarse):
    num_x_D2 = int((6 - 2) / h0)
    num_y_D2 = int((4 - 0) / h0)
    x_D2 = np.linspace(2, 6, num_x_D2, endpoint=False)
    y_D2 = np.linspace(0, 4, num_y_D2, endpoint=False)
    xx_D2, yy_D2 = np.meshgrid(x_D2, y_D2, indexing='ij')
    query_D2 = np.stack((xx_D2, yy_D2), axis=-1).reshape(-1, 2)
    
    # Region D₃ (bottom, coarse):
    num_x_D3 = int((2 - (-2)) / h0)
    num_y_D3 = int((0 - (-4)) / h0)
    x_D3 = np.linspace(-2, 2, num_x_D3, endpoint=False)
    y_D3 = np.linspace(-4, 0, num_y_D3, endpoint=False)
    xx_D3, yy_D3 = np.meshgrid(x_D3, y_D3, indexing='ij')
    query_D3 = np.stack((xx_D3, yy_D3), axis=-1).reshape(-1, 2)
    
    # Region D₄ (top, coarse):
    num_x_D4 = int((2 - (-2)) / h0)
    num_y_D4 = int((8 - 4) / h0)
    x_D4 = np.linspace(-2, 2, num_x_D4, endpoint=False)
    y_D4 = np.linspace(4, 8, num_y_D4, endpoint=False)
    xx_D4, yy_D4 = np.meshgrid(x_D4, y_D4, indexing='ij')
    query_D4 = np.stack((xx_D4, yy_D4), axis=-1).reshape(-1, 2)
    
    query_all = np.concatenate([query_D, query_D1, query_D2, query_D3, query_D4], axis=0)
    return query_all, query_D, query_D1, query_D2, query_D3, query_D4

query_all_initial, query_D, query_D1, query_D2, query_D3, query_D4 = generate_velocity_query_grids()

def compute_velocity_field_regions(u_func, queries):
    """Compute velocity at each query point (for visualization)."""
    all_query = np.concatenate(queries, axis=0)
    U_all = np.array([u_func(q)[0] for q in all_query])
    V_all = np.array([u_func(q)[1] for q in all_query])
    return all_query, U_all, V_all

# ---------------------------
# Klein Bottle Mapping Functions
# ---------------------------
def to_klein(point):
    """
    Maps a point (x,y) in [-2,2)×[0,4) to a Klein bottle in ℝ³.
    We set u = (x+2)*(π/2) and v = y*(π/2), then use the parameterization:
      X = (r + cos(u/2)*sin(v) - sin(u/2)*sin(2*v)) * cos(u)
      Y = (r + cos(u/2)*sin(v) - sin(u/2)*sin(2*v)) * sin(u)
      Z = sin(u/2)*sin(v) + cos(u/2)*sin(2*v)
    """
    x, y = point
    u = (x + 2) * (np.pi/2)
    v = y * (np.pi/2)
    r = 2
    X = (r + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)) * np.cos(u)
    Y = (r + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)) * np.sin(u)
    Z = np.sin(u/2)*np.sin(v) + np.cos(u/2)*np.sin(2*v)
    return np.array([X, Y, Z])

def velocity_to_klein(point, vel):
    """
    Transforms a 2D velocity vector at 'point' from rectangular coordinates
    to the corresponding 3D velocity on the Klein bottle using the Jacobian.
    """
    x, y = point
    u = (x + 2) * (np.pi/2)
    v = y * (np.pi/2)
    u_vel, v_vel = vel
    r = 2
    # Parameterization of Klein bottle:
    # Let A = r + cos(u/2)*sin(v) - sin(u/2)*sin(2*v)
    A = r + np.cos(u/2)*np.sin(v) - np.sin(u/2)*np.sin(2*v)
    # dF/du:
    dA_du = -0.5*np.sin(u/2)*np.sin(v) - 0.5*np.cos(u/2)*np.sin(2*v)
    dF_du = np.array([
        dA_du*np.cos(u) - A*np.sin(u),
        dA_du*np.sin(u) + A*np.cos(u),
        0.5*np.cos(u/2)*np.sin(v) - 0.5*np.sin(u/2)*np.sin(2*v)
    ])
    # dF/dv:
    A_v = np.cos(u/2)*np.cos(v) - 2*np.sin(u/2)*np.cos(2*v)
    dF_dv = np.array([
        A_v*np.cos(u),
        A_v*np.sin(u),
        np.sin(u/2)*np.cos(v) + 2*np.cos(u/2)*np.cos(2*v)
    ])
    # Chain rule: du/dx = π/2, du/dy = 0, dv/dx = 0, dv/dy = π/2
    dF_dx = (np.pi/2) * dF_du
    dF_dy = (np.pi/2) * dF_dv
    return u_vel * dF_dx + v_vel * dF_dy

# ---------------------------
# Prepare the Static Klein Bottle Surface for 3D Plotting
# ---------------------------
u_vals = np.linspace(0, 2*np.pi, 50)
v_vals = np.linspace(0, 2*np.pi, 50)
U_mesh, V_mesh = np.meshgrid(u_vals, v_vals, indexing='ij')
r = 2
X_klein = (r + np.cos(U_mesh/2)*np.sin(V_mesh) - np.sin(U_mesh/2)*np.sin(2*V_mesh)) * np.cos(U_mesh)
Y_klein = (r + np.cos(U_mesh/2)*np.sin(V_mesh) - np.sin(U_mesh/2)*np.sin(2*V_mesh)) * np.sin(U_mesh)
Z_klein = np.sin(U_mesh/2)*np.sin(V_mesh) + np.cos(U_mesh/2)*np.sin(2*V_mesh)

# ---------------------------
# Create the Animation with 2 Subplots: 2D (left) and 3D (right)
# ---------------------------
fig = plt.figure(figsize=(16, 8))
# Left: 2D axes
ax2d = fig.add_subplot(1, 2, 1)
# Right: 3D axes
ax3d = fig.add_subplot(1, 2, 2, projection='3d')

# Initial draw for 2D view.
ax2d.set_xlim(window_x)
ax2d.set_ylim(window_y)
ax2d.set_aspect('equal')
ax2d.grid(True)
ax2d.set_title("2D Simulation (Rectangular Domain) - Klein Bottle")

def init_3d(ax):
    ax.plot_surface(X_klein, Y_klein, Z_klein, alpha=0.2, rstride=2, cstride=2, edgecolor='none')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# ---------------------------
# Combined Update Function for Both Subplots
# ---------------------------
def update(frame):
    t_current = frame * dt

    # ----- 2D Update -----
    ax2d.cla()
    ax2d.set_xlim(window_x)
    ax2d.set_ylim(window_y)
    ax2d.set_aspect('equal')
    ax2d.grid(True)
    ax2d.set_title(f"2D Simulation (t={t_current:.2f}) - Klein Bottle")
    
    u_func = uFuncs[frame] if frame < len(uFuncs) else uFuncs[-1]
    queries = (query_D, query_D1, query_D2, query_D3, query_D4)
    query_all, U_all, V_all = compute_velocity_field_regions(u_func, queries)
    # Plot velocity field as quiver in 2D.
    ax2d.quiver(query_all[:, 0], query_all[:, 1], U_all, V_all,
                color='black', alpha=0.9, pivot='mid', scale_units='xy')
    # Plot boat positions (from region D).
    ax2d.scatter(boat_positions[frame][:, 0], boat_positions[frame][:, 1],
                 s=20, c=boat_colors, zorder=3)
    
    # ----- 3D Update -----
    ax3d.cla()
    init_3d(ax3d)
    
    # Transform velocity query points and vectors into Klein bottle coordinates.
    klein_positions = np.array([to_klein(pt) for pt in query_all])
    klein_velocities = np.array([velocity_to_klein(pt, (u, v)) 
                                 for pt, u, v in zip(query_all, U_all, V_all)])
    Xq = klein_positions[:, 0]
    Yq = klein_positions[:, 1]
    Zq = klein_positions[:, 2]
    Uq = klein_velocities[:, 0]
    Vq = klein_velocities[:, 1]
    Wq = klein_velocities[:, 2]
    
    ax3d.quiver(Xq, Yq, Zq, Uq, Vq, Wq, length=0.5, normalize=True,
                color='black', pivot='middle')
    
    # Transform boat positions into Klein bottle coordinates.
    boats_rect = boat_positions[frame]
    boats_klein = np.array([to_klein(pt) for pt in boats_rect])
    ax3d.scatter(boats_klein[:, 0], boats_klein[:, 1], boats_klein[:, 2],
                 s=20, c=boat_colors, depthshade=True)
    
    ax3d.set_title(f"3D Klein Bottle View (t={t_current:.2f})")
    return

# ---------------------------
# Create and Save the Animation
# ---------------------------
anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "2d_3d_klein.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.show()