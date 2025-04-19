import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter      # <‑‑ new
import os

np.random.seed(42)  # For reproducibility; remove for new randomness each run

# ---------------------------
# PARAMETERS
# ---------------------------
nu    = 0.001   # Viscosity
T     = 10.0    # Final time
dt    = 0.1     # Time step
num_steps = int(T / dt)
h     = 0.1     # Spatial mesh size
delta = 0.1     # Mollification parameter
N     = 10      # Number of sample paths per particle

# Create a 21x21 grid on [-1, 1]^2.
x_values = np.linspace(-1, 1, 21)
y_values = np.linspace(-1, 1, 21)
xx, yy = np.meshgrid(x_values, y_values)
grid_points = np.column_stack((xx.flatten(), yy.flatten()))
num_particles = grid_points.shape[0]

def find_closest_index(target, points):
    """Return the index of the point in 'points' closest to the 'target'."""
    dists = np.linalg.norm(points - target, axis=1)
    return np.argmin(dists)

# Choose two active vortex points based on proximity.
index1 = find_closest_index(np.array([-0.5, 0]), grid_points)
index2 = find_closest_index(np.array([0.5, 0]), grid_points)
active_indices = [index1, index2]
print("Active vortex indices:", active_indices)
print("Active vortex positions:", grid_points[active_indices])

def K_delta(x, delta):
    """
    Mollified Biot–Savart kernel:
        K(x) = (1/(2*pi)) * (-x2/|x|^2, x1/|x|^2)
        K_delta(x) = K(x) * [1 - exp(-( |x|/delta )^2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(- (r / delta) ** 2)
    return (1 / (2 * np.pi)) * np.array([-x[1], x[0]]) / (r ** 2) * factor

# Output folder updated to "figure/two_vortices".
save_folder = os.path.join("figure", "two_vortices")
os.makedirs(save_folder, exist_ok=True)

# Define a custom red‑white colormap.
cmap = LinearSegmentedColormap.from_list('red_white', ['white', 'red'])

def run_simulation(vortex_type):
    """
    Run the simulation for the specified vortex type ('same' or 'opposite').
    """
    w0 = np.zeros(num_particles)
    if vortex_type == 'same':
        for idx in active_indices:
            w0[idx] = 1.0 / (h ** 2)
    elif vortex_type == 'opposite':
        w0[index1] = 1.0 / (h ** 2)
        w0[index2] = -1.0 / (h ** 2)
    else:
        raise ValueError("Invalid vortex type. Please enter 'same' or 'opposite'.")

    # Initialise trajectories.
    trajectories = np.zeros((num_steps + 1, num_particles, N, 2))
    for i in range(num_particles):
        trajectories[0, i, :, :] = grid_points[i]

    # Time stepping.
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
                    drift += (temp / N) * (w0[z] * h ** 2)
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new_positions[i, rho] = current_positions[i, rho] + dt * drift + dW
        trajectories[step + 1] = new_positions

    # Consistent colour scaling.
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
                    temp += (inner / N) * (w0[z] * h ** 2)
                U_frame[i, j] = temp[0]
                V_frame[i, j] = temp[1]
        global_max = max(global_max, np.sqrt(U_frame**2 + V_frame**2).max())

    # ------------------------------
    # Visualisation snapshots
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
                    temp += (inner / N) * (w0[z] * h ** 2)
                U[i, j] = temp[0]
                V[i, j] = temp[1]
        velocity_magnitude = np.sqrt(U**2 + V**2)

        # ----------------------------
        # VISUALISATION (updated)
        # ----------------------------
        fig, ax = plt.subplots(figsize=(6, 6))

        # Domain and aspect.
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', 'box')

        # Hide all spines.
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Major ticks every 0.25 and 2‑dp labels.
        tick_vals = np.linspace(-1, 1, 9)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        formatter = FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', direction='out', length=3, pad=2)

        ax.set_xlabel(r'$x$', labelpad=2)
        ax.set_ylabel(r'$y$', labelpad=2)

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

        # Flow field.
        ax.contourf(xx, yy, velocity_magnitude, levels=100, cmap=cmap,
                    vmin=0, vmax=global_max, alpha=0.6)
        ax.streamplot(xx, yy, U, V, color='black')

        # Annotation.
        ax.text(0.02, 0.98,
                f"t = {t_idx * dt:.2f}\n{vortex_type} vortices",
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Margins: more space on the right for the full '1.00'.
        plt.subplots_adjust(left=0.1, right=0.95,  # <-- right margin enlarged
                            bottom=0.05, top=0.98)

        # Save.
        save_path = os.path.join(
            save_folder, f"streamlines_t{t_idx*dt:.2f}_nu{nu:.2f}_{vortex_type}.svg")
        fig.savefig(save_path, format='svg', dpi=300)
        plt.close(fig)

# Run both scenarios.
run_simulation('same')
run_simulation('opposite')
