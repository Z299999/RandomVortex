import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 21
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
h = x[1] - x[0]  # Grid spacing (0.1)
dt = 0.1
T = 10
nu = 0.1
delta = 0.1  # Mollification parameter
sqrt_2nu = np.sqrt(2 * nu)
steps = int(T / dt)
np.random.seed(42)  # Reproducibility

# Initial positions (21x21 grid)
X, Y = np.meshgrid(x, y)
positions = np.stack([X, Y], axis=-1)  # Shape (21, 21, 2)

# Vorticity-carrying particle (center)
center_idx = (grid_size // 2, grid_size // 2)  # (10,10)

# Store trajectories (including initial positions)
trajectories = np.zeros((steps + 1, grid_size, grid_size, 2))
trajectories[0] = positions.copy()

# Mollified Biot-Savart kernel
def K_delta(dx, delta):
    x, y = dx
    denom = x**2 + y**2 + delta**2
    return np.array([-y / denom, x / denom]) / (2 * np.pi)

# Simulate trajectories
for step in range(steps):
    drifts = np.zeros_like(positions)
    X0_old = trajectories[step, center_idx[0], center_idx[1]]
    
    # Compute drifts for all particles except the center
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == center_idx:
                continue
            current_pos = trajectories[step, i, j]
            dx = X0_old - current_pos
            K = K_delta(dx, delta)
            drifts[i, j] = 0.01 * K * dt  # h²=0.01, drift * dt
    
    # Brownian increments
    dW = np.random.randn(grid_size, grid_size, 2) * sqrt_2nu * np.sqrt(dt)
    
    # Update positions (including center)
    new_positions = trajectories[step] + drifts + dW
    trajectories[step + 1] = new_positions.copy()

# Plot particle trajectories (center and a few others)
plt.figure(figsize=(10, 6))
for i in [5, 10, 15]:
    for j in [5, 10, 15]:
        plt.plot(trajectories[:, i, j, 0], trajectories[:, i, j, 1], 
                 lw=0.5, label=f'Particle ({x[i]:.1f}, {y[j]:.1f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Particle Trajectories')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Compute final velocity field on the original grid
X0_final = trajectories[-1, center_idx[0], center_idx[1]]
velocity_field = np.zeros((grid_size, grid_size, 2))
for i in range(grid_size):
    for j in range(grid_size):
        grid_point = np.array([x[i], y[j]])
        dx = X0_final - grid_point
        K = K_delta(dx, delta)
        velocity_field[i, j] = 0.01 * K  # h²=0.01

# Plot velocity field
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, velocity_field[:, :, 0], velocity_field[:, :, 1], 
           scale=30, color='r')
plt.scatter(X0_final[0], X0_final[1], c='blue', s=100, 
            label='Final Vortex Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Velocity Field at Final Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()