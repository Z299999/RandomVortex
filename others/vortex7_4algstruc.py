import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu        = 0.001      # Viscosity
T         = 30.0       # Final time (seconds)
dt        = 0.1        # Time step
num_steps = int(T/dt)
delta     = 0.1        # Mollification parameter
N         = 10         # Number of sample paths per vortex
num_vortices = 7       # Number of vortices
grid_size = 51        # Grid size for boat simulation

np.random.seed(42)

# ---------------------------
# Vortex Initialization
# ---------------------------
# Seven random vortex positions in [-1,1]^2
vortex_positions = np.random.uniform(-1.0, 1.0, size=(num_vortices, 2))
# Random vortex strengths (vorticity) in [-1,1]
w0 = np.random.uniform(-1.0, 1.0, size=num_vortices)

# ---------------------------
# Define Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, delta):
    """
    Mollified Biot--Savart kernel.
    K(x) = (1/(2*pi)) * (-x_2/|x|^2, x_1/|x|^2)
    K_delta(x) = K(x) * [1 - exp(-(|x|/delta)**2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.zeros(2)
    factor = 1.0 - np.exp(- (r/delta)**2)
    return (1.0/(2*np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulate seven vortices, each with N sample paths,
    using Euler–Maruyama.
    Returns:
      trajectories: shape (num_steps+1, num_vortices, N, 2)
    """
    traj = np.zeros((num_steps+1, num_vortices, N, 2))
    # Initialize: each vortex starts at its random position
    for i in range(num_vortices):
        for rho in range(N):
            traj[0, i, rho, :] = vortex_positions[i]
    
    for step in range(num_steps):
        current = traj[step]  # shape (num_vortices, N, 2)
        new = np.zeros_like(current)
        for i in range(num_vortices):
            for rho in range(N):
                drift = np.zeros(2)
                # Sum contributions from all vortices (including self if desired)
                for j in range(num_vortices):
                    tmp = np.zeros(2)
                    for sigma in range(N):
                        diff = current[j, sigma] - current[i, rho]
                        tmp += K_delta(diff, delta)
                    drift += (tmp / N) * w0[j]
                dW = np.sqrt(2*nu*dt) * np.random.randn(2)
                new[i, rho] = current[i, rho] + dt * drift + dW
        traj[step+1] = new
    return traj

# ---------------------------
# Boat (Particle) Simulation on a Grid
# ---------------------------
def simulate_boats():
    """
    Initialize boats on a grid and update their positions using
    the local velocity computed from vortex trajectories.
    Returns:
      boat_positions: shape (num_steps+1, num_boats, 2)
      boat_directions: shape (num_steps+1, num_boats, 2)
    """
    # Create a grid of boats in [-1,1]^2
    x_vals = np.linspace(-1, 1, grid_size)
    y_vals = np.linspace(-1, 1, grid_size)
    xx_b, yy_b = np.meshgrid(x_vals, y_vals)
    boats0 = np.column_stack((xx_b.flatten(), yy_b.flatten()))
    num_boats = boats0.shape[0]
    
    boat_positions  = np.zeros((num_steps+1, num_boats, 2))
    boat_directions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boats0
    arrow_len = 0.05  # constant arrow length

    for step in range(num_steps+1):
        # For each boat, compute velocity using the vortex positions at this time step.
        current_vortex = trajectories[step]  # shape (num_vortices, N, 2)
        for b in range(num_boats):
            pos_b = boat_positions[step, b]
            vel = np.zeros(2)
            for i in range(num_vortices):
                tmp = np.zeros(2)
                for rho in range(N):
                    diff = current_vortex[i, rho] - pos_b
                    tmp += K_delta(diff, delta)
                vel += (tmp / N) * w0[i]
            # Store the boat's direction (normalized arrow of fixed length)
            norm_v = np.linalg.norm(vel)
            if norm_v > 1e-10:
                boat_directions[step, b] = (vel / norm_v) * arrow_len
            else:
                boat_directions[step, b] = np.zeros(2)
            # Update boat position using Euler forward
            if step < num_steps:
                boat_positions[step+1, b] = pos_b + dt * vel
    return boat_positions, boat_directions

# ---------------------------
# Compute Velocity Field
# ---------------------------
def compute_velocity_field(vortex_positions_t, query_points, delta):
    """
    vortex_positions_t: shape (num_vortices, N, 2) at a given time t.
    query_points: shape (P, 2)
    Returns:
       U, V: each shape (P,)
    """
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        pos = query_points[p]
        vel = np.zeros(2)
        for i in range(num_vortices):
            tmp = np.zeros(2)
            for rho in range(N):
                diff = vortex_positions_t[i, rho] - pos
                tmp += K_delta(diff, delta)
            vel += (tmp / N) * w0[i]
        U[p], V[p] = vel[0], vel[1]
    return U, V

# ---------------------------
# Main Simulation
# ---------------------------
trajectories = simulate_vortex_trajectories()  # shape (num_steps+1, num_vortices, N, 2)
boat_positions, boat_dirs = simulate_boats()    # shapes (num_steps+1, num_boats, 2)

# Create a grid for velocity field animation (using the same grid as boats)
xq = np.linspace(-1, 1, grid_size)
yq = np.linspace(-1, 1, grid_size)
xx, yy = np.meshgrid(xq, yq)
query_points = np.column_stack((xx.flatten(), yy.flatten()))

# ---------------------------
# Animation: Combined Velocity Field and Boat Simulation
# ---------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.0)")

# Initial velocity field (black arrows)
U, V = compute_velocity_field(trajectories[0], query_points, delta)
U_plot = U.reshape(xx.shape)
V_plot = V.reshape(yy.shape)
vel_quiver = ax.quiver(xx, yy, U_plot, V_plot, color='black', pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

# Initial boat positions and directions (red arrows, slightly transparent)
boat_pos = boat_positions[0]
boat_dir = boat_dirs[0]
boat_quiver = ax.quiver(boat_pos[:,0], boat_pos[:,1], boat_dir[:,0], boat_dir[:,1],
                        color='red', pivot='tail', alpha=0.5, scale=None, angles='xy', scale_units='xy')

def update(frame):
    t_current = frame * dt
    # Update velocity field quiver
    current_vortex = trajectories[frame]  # shape (num_vortices, N, 2)
    U, V = compute_velocity_field(current_vortex, query_points, delta)
    U_plot = U.reshape(xx.shape)
    V_plot = V.reshape(yy.shape)
    vel_quiver.set_UVC(U_plot.flatten(), V_plot.flatten())
    # Update boat quiver
    boat_pos = boat_positions[frame]
    boat_dir = boat_dirs[frame]
    boat_quiver.set_offsets(boat_pos)
    boat_quiver.set_UVC(boat_dir[:,0], boat_dir[:,1])
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_quiver

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

# Save animation in subfolder "animation" as "vortex7.mp4"
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "vortex7.mp4")
writer = FFMpegWriter(fps=30)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)