import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.001         # Viscosity
T = 10.0           # Final time (seconds)
dt = 0.1           # Time step
num_steps = int(T/dt)
delta = 0.1        # Mollification parameter (choose δ < 0.15 for the boundary layer)
N = 10             # Number of sample paths per vortex
num_vortices = 7   # Total number of vortices

np.random.seed(42)

# ---------------------------
# Vortex Initialization in D (x2 < 0)
# ---------------------------
# Generate seven random vortex positions in D (x2 < 0)
vortex_positions = np.zeros((num_vortices, 2))
vortex_positions[:, 0] = np.random.uniform(-1, 1, num_vortices)
vortex_positions[:, 1] = np.random.uniform(-1, 0, num_vortices)
# Random vortex strengths (vorticity) in [-1,1]
w0 = np.random.uniform(-1, 1, num_vortices)

# ---------------------------
# Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, delta):
    """
    Mollified Biot--Savart kernel.
    K(x) = (1/(2*pi)) * (-x2/|x|^2, x1/|x|^2)
    K_delta(x) = K(x) * [1 - exp(-(|x|/delta)**2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.zeros(2)
    factor = 1.0 - np.exp(- (r/delta)**2)
    return (1.0/(2*np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

def indicator_D(point):
    "Return 1 if point is in D (i.e., if x2 < 0), else 0."
    return 1 if point[1] < 0 else 0

def reflect(point):
    "Reflect a point across the wall: (x1, x2) -> (x1, -x2)."
    return np.array([point[0], -point[1]])

# ---------------------------
# Vortex Trajectory Simulation
# ---------------------------
def simulate_vortex_trajectories():
    """
    Simulate 'num_vortices' vortices, each with N sample paths, using Euler–Maruyama.
    Returns an array of shape (num_steps+1, num_vortices, N, 2).
    """
    traj = np.zeros((num_steps+1, num_vortices, N, 2))
    for i in range(num_vortices):
        for rho in range(N):
            traj[0, i, rho, :] = vortex_positions[i]
    for step in range(num_steps):
        current = traj[step]  # shape: (num_vortices, N, 2)
        new = np.zeros_like(current)
        for i in range(num_vortices):
            for rho in range(N):
                drift = np.zeros(2)
                for j in range(num_vortices):
                    tmp = np.zeros(2)
                    for sigma in range(N):
                        diff = current[j, sigma] - current[i, rho]
                        # Use method of images: subtract contribution from reflection.
                        contrib1 = indicator_D(current[j, sigma]) * K_delta(diff, delta)
                        contrib2 = indicator_D(reflect(current[j, sigma])) * K_delta(reflect(current[j, sigma]) - current[i, rho], delta)
                        tmp += (contrib1 - contrib2)
                    drift += (tmp / N) * w0[j]
                dW = np.sqrt(2*nu*dt) * np.random.randn(2)
                new[i, rho] = current[i, rho] + dt * drift + dW
        traj[step+1] = new
    return traj

# ---------------------------
# Generate Nonuniform Grid in D (Finer Near Boundary x2=0)
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
# Boat Simulation on Nonuniform Grid in D
# ---------------------------
def simulate_boats(vortex_traj):
    """
    Simulate boat trajectories on the nonuniform grid in D.
    Boats move according to the local velocity computed from vortex trajectories.
    Returns:
      boat_positions: shape (num_steps+1, num_boats, 2)
      boat_directions: shape (num_steps+1, num_boats, 2)
    """
    boat_grid, xx_b, yy_b = generate_nonuniform_grid_D()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_directions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    arrow_len = 0.05  # constant arrow length

    for step in range(num_steps+1):
        current_vortex = vortex_traj[step]  # shape (num_vortices, N, 2)
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
            if np.linalg.norm(vel) > 1e-10:
                boat_directions[step, b] = (vel/np.linalg.norm(vel)) * arrow_len
            else:
                boat_directions[step, b] = np.zeros(2)
            if step < num_steps:
                boat_positions[step+1, b] = pos + dt * vel
    return boat_positions, boat_directions

# ---------------------------
# Compute Velocity Field at Query Points in D
# ---------------------------
def compute_velocity_field(vortex_positions_t, query_points):
    """
    Compute the velocity field at given query points in D using the method of images.
    vortex_positions_t: shape (num_vortices, N, 2) at time t.
    query_points: shape (P,2)
    Returns U, V arrays of shape (P,).
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
# Identify Special Boats (Vortex Boats)
# ---------------------------
def identify_special_boats(boat_positions, threshold=0.1):
    """
    Identify boats whose initial positions are close to any vortex position.
    Returns a boolean mask (length = num_boats).
    """
    boats = boat_positions[0]  # initial boat positions
    mask = np.zeros(boats.shape[0], dtype=bool)
    for b, pos in enumerate(boats):
        for vortex in vortex_positions:
            if np.linalg.norm(pos - vortex) < threshold:
                mask[b] = True
                break
    return mask

# ---------------------------
# Main Simulation
# ---------------------------
trajectories = simulate_vortex_trajectories()  # shape: (num_steps+1, num_vortices, N, 2)
boat_positions, boat_dirs = simulate_boats(trajectories)
query_grid, xx_query, yy_query = generate_nonuniform_grid_D()

# Identify special boats (boats initially at vortex locations)
special_mask = identify_special_boats(boat_positions, threshold=0.1)

# ---------------------------
# Animation: Combined Velocity Field and Boat Animation
# ---------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 0.2)  # domain D: x2 < 0; show a bit above the wall
ax.set_aspect('equal')
ax.grid(True)
# Add parameter annotation in bottom left corner (small, thin font)
param_text = ax.text(-1.15, -1.15, f"$\\nu={nu}$, $M={num_vortices}$", fontsize=8, color='black',
                     verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

ax.set_title("Vortex & Boat Animation (t=0.00)")

# Initialize velocity field quiver (black arrows)
U, V = compute_velocity_field(trajectories[0], query_grid)
U_plot = U.reshape(xx_query.shape)
V_plot = V.reshape(yy_query.shape)
vel_quiver = ax.quiver(xx_query, yy_query, U_plot, V_plot, color='black',
                       pivot='mid', scale=None, angles='xy', scale_units='xy')

# Initialize boat quivers:
# - Special boats (vortex boats) as blue dots
# - Ordinary boats as red arrows with high transparency
boats_init = boat_positions[0]
dirs_init = boat_dirs[0]
special_boats = boats_init[special_mask]
ordinary_boats = boats_init[~special_mask]
ordinary_dirs = boat_dirs[0][~special_mask]

spec_scatter = ax.scatter(special_boats[:,0], special_boats[:,1], color='blue', s=30)
ord_quiv = ax.quiver(ordinary_boats[:,0], ordinary_boats[:,1], ordinary_dirs[:,0], ordinary_dirs[:,1],
                     color='red', pivot='tail', alpha=0.2, scale=None, angles='xy', scale_units='xy')

# Initialize vortex boats scatter: mean vortex positions (blue dots)
mean_vortex = np.mean(trajectories[0], axis=1)  # shape: (num_vortices, 2)
vortex_scatter = ax.scatter(mean_vortex[:,0], mean_vortex[:,1], color='blue', s=40)

def update(frame):
    t_current = frame * dt
    # Update velocity field
    current_vortex = trajectories[frame]  # shape: (num_vortices, N, 2)
    U, V = compute_velocity_field(current_vortex, query_grid)
    U_plot = U.reshape(xx_query.shape)
    V_plot = V.reshape(yy_query.shape)
    vel_quiver.set_UVC(U_plot.flatten(), V_plot.flatten())
    
    # Update ordinary boats
    boats = boat_positions[frame]
    dirs = boat_dirs[frame]
    ordinary_boats = boats[~special_mask]
    ordinary_dirs  = dirs[~special_mask]
    ord_quiv.set_offsets(ordinary_boats)
    ord_quiv.set_UVC(ordinary_dirs[:,0], ordinary_dirs[:,1])
    
    # Update special boats (vortex boats) scatter: we take mean of boat positions for those marked special
    special_boats = boats[special_mask]
    spec_scatter.set_offsets(special_boats)
    
    # Update vortex boats: mean vortex positions
    mean_vortex = np.mean(current_vortex, axis=1)
    vortex_scatter.set_offsets(mean_vortex)
    
    # Update parameter annotation (static in this case)
    param_text.set_text(f"$\\nu={nu}$, $M={num_vortices}$")
    
    ax.set_title(f"Vortex & Boat Animation (t={t_current:.2f})")
    return vel_quiver, ord_quiv, spec_scatter, vortex_scatter, param_text

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)

# Save the animation in a subfolder "animation" as "vortex7.mp4"
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "wall2v.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)