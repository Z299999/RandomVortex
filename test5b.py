# one vortex no slip boundary condition simulation

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility
nu = 0.001           # Viscosity
T = 10.0             # Final time
dt = 0.1             # Time step
num_steps = int(T/dt)
h = 0.1              # Spatial mesh size
delta = 0.1          # Mollification parameter

# Create a 21x21 grid on [-1,1]^2
x_values = np.linspace(-1,1,21)
y_values = np.linspace(-1,1,21)
xx, yy = np.meshgrid(x_values, y_values)
grid_points = np.column_stack((xx.flatten(), yy.flatten()))
num_particles = grid_points.shape[0]

# Identify the "center" particle (closest to (0,0))
distances = np.linalg.norm(grid_points, axis=1)
center_index = np.argmin(distances)

# Only the center has nonzero vorticity.
w0 = np.zeros(num_particles)
w0[center_index] = 100.0 / (h**2)

def K_delta(x, delta):
    """
    Mollified Biot--Savart kernel.
    
    K(x) = (1/(2π)) * (-x2/|x|^2, x1/|x|^2)
    K_delta(x) = K(x) * [1 - exp(-(|x|/delta)^2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(- (r/delta)**2)
    return (1 / (2 * np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

def enforce_noslip_boundary(pos, prev_pos):
    """
    Enforce no-slip (viscous) boundary conditions:
    If the updated position 'pos' lies outside [-1,1]^2,
    then set it equal to the previous position 'prev_pos'.
    This mimics zero velocity at the boundary.
    """
    x, y = pos
    if (x < -1) or (x > 1) or (y < -1) or (y > 1):
        return prev_pos
    else:
        return pos

# Use N = 10 sample paths per grid point.
N = 10

# Initialize trajectories:
# trajectories shape: (num_steps+1, num_particles, N, 2)
trajectories = np.zeros((num_steps + 1, num_particles, N, 2))
for i in range(num_particles):
    trajectories[0, i, :, :] = grid_points[i]

for step in range(num_steps):
    current_positions = trajectories[step]  # shape: (num_particles, N, 2)
    new_positions = np.zeros_like(current_positions)
    # For the center, collect its N sample positions.
    center_positions = current_positions[center_index]  # shape: (N, 2)
    
    for i in range(num_particles):
        for rho in range(N):
            if i == center_index:
                drift = np.array([0.0, 0.0])
            else:
                temp = np.zeros(2)
                for sigma in range(N):
                    diff = center_positions[sigma] - current_positions[i, rho]
                    temp += K_delta(diff, delta)
                drift = temp / N
            dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
            pos_new = current_positions[i, rho] + dt * drift + dW
            # Enforce no-slip boundary: if pos_new is outside, reset to previous position.
            pos_new = enforce_noslip_boundary(pos_new, current_positions[i, rho])
            new_positions[i, rho] = pos_new
    trajectories[step + 1] = new_positions

# Compute the final velocity field using:
#   u(x, T) = (1/N) * sum_{sigma=1}^N K_delta(X_center^(sigma)(T) - x)
center_final = trajectories[-1, center_index]  # shape: (N, 2)
U = np.zeros_like(xx)
V = np.zeros_like(yy)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        pos = np.array([xx[i, j], yy[i, j]])
        temp = np.zeros(2)
        for sigma in range(N):
            diff = center_final[sigma] - pos
            temp += K_delta(diff, delta)
        vel = temp / N
        U[i, j] = vel[0]
        V[i, j] = vel[1]
    
plt.figure(figsize=(6,6))
plt.quiver(xx, yy, U, V, pivot='mid', color='red')
plt.title("Final Velocity Field (No-slip Boundary Condition)")
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.grid(True)
plt.show()