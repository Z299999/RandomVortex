import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

np.random.seed(42)  # For reproducibility; remove for new randomness each run

# ---------------------------
# PARAMETERS
# ---------------------------
nu    = 0.001    # Viscosity
T     = 10.0     # Final time
dt    = 0.1      # Time step
num_steps = int(T / dt)
h     = 0.1      # Spatial mesh size
delta = 0.1      # Mollification parameter
N     = 10       # Number of sample paths per particle

# Create a 21x21 grid on [-1,1]^2
x_values = np.linspace(-1, 1, 21)
y_values = np.linspace(-1, 1, 21)
xx, yy = np.meshgrid(x_values, y_values)
grid_points = np.column_stack((xx.flatten(), yy.flatten()))
num_particles = grid_points.shape[0]

def find_closest_index(target, points):
    dists = np.linalg.norm(points - target, axis=1)
    return np.argmin(dists)

# Choose two active vortex points:
index1 = find_closest_index(np.array([-0.5, 0]), grid_points)
index2 = find_closest_index(np.array([0.5, 0]), grid_points)
active_indices = [index1, index2]
print("Active vortex indices:", active_indices)
print("Active vortex positions:", grid_points[active_indices])

def K_delta(x, delta):
    """
    Mollified Biot–Savart kernel:
      K(x) = (1/(2*pi)) * (-x2/|x|^2, x1/|x|^2)
      K_delta(x) = K(x) * [1 - exp(-(|x|/delta)^2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(- (r / delta)**2)
    return (1 / (2 * np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

# Folder to save static images
save_folder = os.path.join("figure", "two_vortices")
os.makedirs(save_folder, exist_ok=True)

# Define a function to run the simulation for a given vortex type
def run_simulation(vortex_type):
    w0 = np.zeros(num_particles)
    if vortex_type == 'same':
        for idx in active_indices:
            w0[idx] = 1.0 / (h**2)
    elif vortex_type == 'opposite':
        w0[index1] =  1.0 / (h**2)
        w0[index2] = -1.0 / (h**2)
    else:
        raise ValueError("Invalid vortex type. Please enter 'same' or 'opposite'.")

    # Initialize trajectories for all particles.
    trajectories = np.zeros((num_steps + 1, num_particles, N, 2))
    for i in range(num_particles):
        trajectories[0, i, :, :] = grid_points[i]

    # Time-stepping: Update every particle.
    for step in range(num_steps):
        current_positions = trajectories[step]
        new_positions = np.zeros_like(current_positions)
        for i in range(num_particles):
            for rho in range(N):
                drift = np.zeros(2)
                for z in active_indices:
                    temp = np.zeros(2)
                    for sigma in range(N):
                        diff = current_positions[z, sigma] - current_positions[i, rho]
                        temp += K_delta(diff, delta)
                    drift += (temp / N) * (w0[z] * h**2)
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new_positions[i, rho] = current_positions[i, rho] + dt * drift + dW
        trajectories[step + 1] = new_positions

    # Compute global maximum velocity magnitude over all frames,
    # so that we use the same color bar scale.
    global_max = 0.0
    for frame in range(num_steps + 1):
        U_frame = np.zeros_like(xx)
        V_frame = np.zeros_like(yy)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                pos = np.array([xx[i, j], yy[i, j]])
                temp = np.zeros(2)
                for z in active_indices:
                    vortex_positions = trajectories[frame, z]
                    inner = np.zeros(2)
                    for sigma in range(N):
                        diff = vortex_positions[sigma] - pos
                        inner += K_delta(diff, delta)
                    temp += (inner / N) * (w0[z] * h**2)
                U_frame[i, j] = temp[0]
                V_frame[i, j] = temp[1]
        mag_frame = np.sqrt(U_frame**2 + V_frame**2).max()
        if mag_frame > global_max:
            global_max = mag_frame

    # ------------------------------
    # Static Visualization (several time stages)
    # ------------------------------
    time_indices = np.linspace(0, num_steps, 5, dtype=int)
    for t_idx in time_indices:
        U = np.zeros_like(xx)
        V = np.zeros_like(yy)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                pos = np.array([xx[i, j], yy[i, j]])
                temp = np.zeros(2)
                for z in active_indices:
                    vortex_positions = trajectories[t_idx, z]
                    inner = np.zeros(2)
                    for sigma in range(N):
                        diff = vortex_positions[sigma] - pos
                        inner += K_delta(diff, delta)
                    temp += (inner / N) * (w0[z] * h**2)
                U[i, j] = temp[0]
                V[i, j] = temp[1]
        velocity_magnitude = np.sqrt(U**2 + V**2)
        fig, ax = plt.subplots(figsize=(6, 6))
        ctf = ax.contourf(xx, yy, velocity_magnitude, levels=100, cmap='coolwarm',
                          vmin=0, vmax=global_max, alpha=0.6)
        cb = fig.colorbar(ctf, ax=ax)
        cb.set_label("Velocity Magnitude")
        ax.streamplot(xx, yy, U, V, color='purple')
        ax.set_title(f'Streamlines at t = {t_idx*dt:.2f} ({vortex_type} vortices)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid(True)
        save_path = os.path.join(save_folder, f"streamlines_t{t_idx*dt:.2f}_{vortex_type}.png")
        plt.savefig(save_path)
        plt.close(fig)

# Run the simulation for both 'same' and 'opposite' vortex types
run_simulation('same')
run_simulation('opposite')
