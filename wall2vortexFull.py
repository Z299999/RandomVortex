import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.01          # Viscosity
T = 20.0           # Final time (seconds)
dt = 0.1           # Time step
num_steps = int(T/dt)
delta = 0.1        # Mollification parameter (δ < 0.15)
N = 10             # Number of sample paths per vortex
original_num_vortices = 2   # Original vortices in D
num_vortices = 2 * original_num_vortices  # Including images

np.random.seed(42)

# ---------------------------
# Vortex Initialization
# ---------------------------
# Original vortices in D (x2 < 0)
original_vortex_positions = np.array([[-0.5, -1.0], [0.5, -1.0]])
# Image vortices (reflected across x2=0)
image_vortex_positions = original_vortex_positions * np.array([1, -1])
vortex_positions = np.concatenate([original_vortex_positions, image_vortex_positions])

# Vortex strengths (original vortices: -1 and 1, images: 1 and -1)
w0_original = np.array([-1, 1])
w0_image = -w0_original
w0 = np.concatenate([w0_original, w0_image])

def reflect(point):
    """Reflects a point across the wall: (x1, x2) -> (x1, -x2)"""
    return np.array([point[0], -point[1]])

# ---------------------------
# Modified Biot-Savart Kernel with Reflection
# ---------------------------
def modified_K(x, y, delta):
    """Computes the modified kernel K(x,y) with reflection and mollification."""
    x_reflect = np.array([x[0], -x[1]])
    r1 = y - x
    r2 = y - x_reflect
    
    # Compute norms and mollification factors
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    # Avoid division by zero
    if r1_norm < 1e-10:
        term1 = 0.0
    else:
        term1 = (r1[1] / (r1_norm**2)) * (1 - np.exp(-(r1_norm/delta)**2))
    
    if r2_norm < 1e-10:
        term2 = 0.0
    else:
        term2 = ((y[1] + x[1]) / (r2_norm**2)) * (1 - np.exp(-(r2_norm/delta)**2))
    
    K1 = (term1 - term2) / (2 * np.pi)
    
    # Compute K2 component
    if r1_norm < 1e-10:
        term1_k2 = 0.0
    else:
        term1_k2 = ((x[0] - y[0]) / (r1_norm**2)) * (1 - np.exp(-(r1_norm/delta)**2))
    
    if r2_norm < 1e-10:
        term2_k2 = 0.0
    else:
        term2_k2 = ((x[0] - y[0]) / (r2_norm**2)) * (1 - np.exp(-(r2_norm/delta)**2))
    
    K2 = (term1_k2 - term2_k2) / (2 * np.pi)
    
    return np.array([K1, K2])

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    """Simulates trajectories for original and image vortices."""
    traj = np.zeros((num_steps+1, num_vortices, N, 2))
    # Initialize positions
    for i in range(num_vortices):
        traj[0, i, :, :] = vortex_positions[i]
    
    for step in range(num_steps):
        current = traj[step]
        new = np.zeros_like(current)
        
        for i in range(num_vortices):
            for rho in range(N):
                current_pos = current[i, rho]
                drift = np.zeros(2)
                
                # Accumulate velocity contributions from all vortices
                for j in range(num_vortices):
                    tmp = np.zeros(2)
                    for sigma in range(N):
                        pos_j = current[j, sigma]
                        # Check if vortex j's position is in D
                        if pos_j[1] < 0:
                            contrib = modified_K(current_pos, pos_j, delta) * w0[j]
                            tmp += contrib
                    drift += tmp / N  # Average over samples
                
                # Update position with drift and diffusion
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new[i, rho] = current_pos + dt * drift + dW
        traj[step+1] = new
    return traj

# ---------------------------
# Grid Generation
# ---------------------------
def generate_nonuniform_grid():
    # Fine grid near boundary (x2=0)
    x1 = np.linspace(-1, 1, 41)
    x2_coarse = np.linspace(-1, -0.15, 20, endpoint=False)
    x2_fine = np.linspace(-0.15, 0, 15)
    x2 = np.concatenate((x2_coarse, x2_fine))
    xx, yy = np.meshgrid(x1, x2)
    return np.column_stack((xx.flatten(), yy.flatten())), xx, yy

# ---------------------------
# Boat Simulation
# ---------------------------
def simulate_boats(vortex_traj):
    """Simulates boats on a non-uniform grid."""
    boat_grid, _, _ = generate_nonuniform_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    arrow_len = 0.001  # Visualization parameter
    
    for step in range(num_steps+1):
        current_vortex = vortex_traj[step]
        for b in range(num_boats):
            pos = boat_positions[step, b]
            vel = np.zeros(2)
            
            # Compute velocity contribution from all vortices
            for j in range(num_vortices):
                tmp = np.zeros(2)
                for rho in range(N):
                    pos_j = current_vortex[j, rho]
                    if pos_j[1] < 0:  # Check if in D
                        contrib = modified_K(pos, pos_j, delta) * w0[j]
                        tmp += contrib
                vel += tmp / N
            
            # Update boat position and direction
            if step < num_steps:
                boat_positions[step+1, b] = pos + dt * vel
    return boat_positions

# ---------------------------
# Velocity Field Computation
# ---------------------------
def compute_velocity_field(vortex_positions_t, query_points):
    """Computes velocity field using modified kernel."""
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    
    for p in range(P):
        x = query_points[p]
        vel = np.zeros(2)
        for j in range(num_vortices):
            tmp = np.zeros(2)
            for rho in range(N):
                pos_j = vortex_positions_t[j, rho]
                if pos_j[1] < 0:  # Check if in D
                    contrib = modified_K(x, pos_j, delta) * w0[j]
                    tmp += contrib
            vel += tmp / N
        U[p], V[p] = vel[0], vel[1]
    return U, V

# ---------------------------
# Main Simulation
# ---------------------------
trajectories = simulate_vortex_trajectories()
boat_positions = simulate_boats(trajectories)
query_grid, xx_query, yy_query = generate_nonuniform_grid()

# ---------------------------
# Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 0.2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Initialize velocity quiver
U, V = compute_velocity_field(trajectories[0], query_grid)
vel_quiver = ax.quiver(xx_query, yy_query, U.reshape(xx_query.shape), V.reshape(yy_query.shape), 
                       color='black', pivot='mid', scale=30)

# Initialize boats
boats = boat_positions[0]
boat_quiv = ax.quiver(boats[:,0], boats[:,1], np.zeros_like(boats), np.zeros_like(boats),
                      color='red', scale=50, alpha=0.6)

def update(frame):
    t = frame * dt
    # Update velocity field
    U, V = compute_velocity_field(trajectories[frame], query_grid)
    vel_quiver.set_UVC(U.reshape(xx_query.shape), V.reshape(yx_query.shape))
    
    # Update boats
    boats = boat_positions[frame]
    boat_quiv.set_offsets(boats)
    
    ax.set_title(f"Vortex and Boat Animation (t={t:.2f})")
    return vel_quiver, boat_quiv

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=50, blit=False)

# Save animation
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex7_corrected.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved to {save_path}")

plt.close()