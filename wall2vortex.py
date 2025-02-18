import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.00          # Viscosity
T = 20.0           # Final time (seconds)
dt = 0.1           # Time step
num_steps = int(T/dt)
delta = 0.1        # Mollification parameter (choose δ < 0.15 for the boundary layer)
N = 10             # Number of sample paths per vortex
num_vortices = 2   # Total number of vortices

np.random.seed(42)

# ---------------------------
# Vortex Initialization in D (x2 < 0)
# ---------------------------
# We generate vortex positions with x1 in [-1,1] and x2 in [-1,0]
vortex_positions = np.zeros((num_vortices, 2))
# vortex_positions[:, 0] = np.random.uniform(-1, 1, num_vortices)
# vortex_positions[:, 1] = np.random.uniform(-1, 0, num_vortices)

vortex_positions[:, 0] = np.array([-0.5, 0.5]) # x axis
vortex_positions[:, 1] = np.array([-1, -1]) # y axis

# Random vortex strengths (vorticity) in [-1,1]
# w0 = np.random.uniform(-1, 1, num_vortices)

w0 = np.array([-1, 1])



# ---------------------------
# Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, delta):
    """
    Computes the mollified Biot--Savart kernel.
    
    K(x) = (1/(2*pi)) * (-x_2/|x|^2, x_1/|x|^2)
    K_delta(x) = K(x) * [1 - exp(-(|x|/delta)^2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.zeros(2)
    factor = 1.0 - np.exp(- (r/delta)**2)
    return (1.0/(2*np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

def indicator_D(point):
    "Returns 1 if point is in D (i.e. if x2 < 0), else 0."
    return 1 if point[1] < 0 else 0

def reflect(point):
    "Reflects a point across the wall: (x1, x2) -> (x1, -x2)."
    return np.array([point[0], -point[1]])

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulates the trajectories of 'num_vortices' vortices,
    each with N sample paths, using Euler-Maruyama.
    Returns a 4D array of shape (num_steps+1, num_vortices, N, 2).
    """
    traj = np.zeros((num_steps+1, num_vortices, N, 2))
    # Initialize: each vortex's sample paths are set to its initial position.
    for i in range(num_vortices):
        for rho in range(N):
            traj[0, i, rho, :] = vortex_positions[i]
    for step in range(num_steps):
        current = traj[step]  # shape: (num_vortices, N, 2)
        new = np.zeros_like(current)
        for i in range(num_vortices):
            for rho in range(N):
                drift = np.zeros(2)
                # Sum over contributions from all vortices using method of images.
                for j in range(num_vortices):
                    tmp = np.zeros(2)
                    for sigma in range(N):
                        pos_j = current[j, sigma]
                        # Contribution from vortex j if it is in D.
                        contrib1 = indicator_D(pos_j) * K_delta(pos_j - current[i, rho], delta)
                        # Contribution from its reflected image.
                        contrib2 = indicator_D(reflect(pos_j)) * K_delta(reflect(pos_j) - current[i, rho], delta)
                        tmp += (contrib1 - contrib2)
                    drift += (tmp / N) * w0[j]
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new[i, rho] = current[i, rho] + dt * drift + dW
        traj[step+1] = new
    return traj

# ---------------------------
# Nonuniform Grid for Query Points (Finer near boundary x2=0)
# ---------------------------
def generate_nonuniform_grid_D():
    # x1: 41 uniformly spaced points in [-1,1]
    x1 = np.linspace(-1, 1, 41)
    # x2: use coarse grid for x2 in [-1, -0.15] and fine grid for x2 in [-0.15, 0]
    x2_coarse = np.linspace(-1, -0.15, 20, endpoint=False)
    x2_fine = np.linspace(-0.15, 0, 15)
    x2 = np.concatenate((x2_coarse, x2_fine))
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack((xx.flatten(), yy.flatten()))
    return grid, xx, yy

# ---------------------------
# Boat Simulation on a Nonuniform Grid in D
# ---------------------------
def generate_boat_grid():
    # Similar to query grid for velocity but may use even finer density if desired.
    # Here, we use the same nonuniform grid.
    return generate_nonuniform_grid_D()  # returns (grid, xx, yy)

def simulate_boats(vortex_traj):
    """
    Simulate boat trajectories on the nonuniform grid in D.
    Boats move according to the local velocity computed from the vortex simulation.
    Returns:
      boat_positions: shape (num_steps+1, num_boats, 2)
      boat_directions: shape (num_steps+1, num_boats, 2)
    """
    boat_grid, _, _ = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_directions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    arrow_len = 0.05  # constant arrow length
    for step in range(num_steps+1):
        current_vortex = vortex_traj[step]  # shape: (num_vortices, N, 2)
        for b in range(num_boats):
            pos = boat_positions[step, b]
            vel = np.zeros(2)
            for i in range(num_vortices):
                tmp = np.zeros(2)
                for rho in range(N):
                    pos_i = current_vortex[i, rho]
                    contrib1 = indicator_D(pos_i) * K_delta(pos_i - pos, delta)
                    contrib2 = indicator_D(reflect(pos_i)) * K_delta(reflect(pos_i) - pos, delta)
                    tmp += (contrib1 - contrib2)
                vel += (tmp / N) * w0[i]
            norm_vel = np.linalg.norm(vel)
            if norm_vel > 1e-10:
                boat_directions[step, b] = (vel / norm_vel) * arrow_len
            else:
                boat_directions[step, b] = np.zeros(2)
            if step < num_steps:
                boat_positions[step+1, b] = pos + dt * vel
    return boat_positions, boat_directions

# ---------------------------
# Velocity Field Computation
# ---------------------------
def compute_velocity_field(vortex_positions_t, query_points):
    """
    Computes the velocity field at the given query points using
    the method of images.
    vortex_positions_t: shape (num_vortices, N, 2) at time t.
    query_points: shape (P,2) in D.
    Returns U, V (each of shape (P,)).
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
                pos_i = vortex_positions_t[i, rho]
                contrib1 = indicator_D(pos_i) * K_delta(pos_i - pos, delta)
                contrib2 = indicator_D(reflect(pos_i)) * K_delta(reflect(pos_i) - pos, delta)
                tmp += (contrib1 - contrib2)
            vel += (tmp / N) * w0[i]
        U[p], V[p] = vel[0], vel[1]
    return U, V

# ---------------------------
# Main Simulation
# ---------------------------
trajectories = simulate_vortex_trajectories()  # shape (num_steps+1, num_vortices, N, 2)
boat_positions, boat_dirs = simulate_boats(trajectories)

# Create nonuniform grid for velocity field query (in D)
query_grid, xx_query, yy_query = generate_nonuniform_grid_D()

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 0.2)  # Since domain D is x2<0, show a little above the boundary
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

# Initialize velocity field quiver (black arrows)
U, V = compute_velocity_field(trajectories[0], query_grid)
U_plot = U.reshape(xx_query.shape)
V_plot = V.reshape(yy_query.shape)
vel_quiver = ax.quiver(xx_query, yy_query, U_plot, V_plot, color='black', pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

# Initialize boat quiver (red arrows, semi-transparent)
boats_init = boat_positions[0]
boat_dirs_init = boat_dirs[0]
boat_quiver = ax.quiver(boats_init[:,0], boats_init[:,1], boat_dirs_init[:,0], boat_dirs_init[:,1],
                        color='red', pivot='tail', alpha=0.5, scale=None, angles='xy', scale_units='xy')

def update(frame):
    t_current = frame * dt
    # Update velocity field
    current_vortex = trajectories[frame]  # shape (num_vortices, N, 2)
    U, V = compute_velocity_field(current_vortex, query_grid)
    U_plot = U.reshape(xx_query.shape)
    V_plot = V.reshape(yy_query.shape)
    vel_quiver.set_UVC(U_plot.flatten(), V_plot.flatten())
    
    # Update boat positions and directions
    boats = boat_positions[frame]
    dirs = boat_dirs[frame]
    boat_quiver.set_offsets(boats)
    boat_quiver.set_UVC(dirs[:,0], dirs[:,1])
    
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_quiver

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

# Save the animation to a subfolder "animation" as "vortex7.mp4"
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "wall2.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)