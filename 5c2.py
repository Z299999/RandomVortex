import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(42)  # For reproducibility

# ---------------------------
# PARAMETERS
# ---------------------------
nu = 0.001           # Viscosity
T = 20.0             # Final time (seconds)
dt = 0.1             # Time step
num_steps = int(T/dt)  # 200 steps
h = 0.1              # Spatial mesh size
delta = 0.1          # Mollification parameter
N = 10               # Number of sample paths per particle

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
index2 = find_closest_index(np.array([ 0.5, 0]), grid_points)
active_indices = [index1, index2]
print("Active vortex indices:", active_indices)
print("Active vortex positions:", grid_points[active_indices])

# Define the discrete vorticity with opposite signs:
w0 = np.zeros(num_particles)
w0[index1] =  1.0 / (h**2)
w0[index2] = -1.1 / (h**2)

def K_delta(x, delta):
    """
    Mollified Biot--Savart kernel:
      K(x) = (1/(2*pi)) * (-x2/|x|^2, x1/|x|^2)
      K_delta(x) = K(x) * [1 - exp(-(|x|/delta)^2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(- (r/delta)**2)
    return (1/(2*np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

def enforce_noslip_boundary(pos, prev_pos):
    """
    Enforce no-slip boundary condition:
    If pos is outside [-1,1]^2, return the previous position (i.e. freeze the particle).
    """
    x, y = pos
    if x < -1 or x > 1 or y < -1 or y > 1:
        return prev_pos
    return pos

# Initialize trajectories:
# Shape: (num_steps+1, num_particles, N, 2)
trajectories = np.zeros((num_steps+1, num_particles, N, 2))
for i in range(num_particles):
    trajectories[0, i, :, :] = grid_points[i]

# Time-stepping simulation:
for step in range(num_steps):
    current_positions = trajectories[step]  # shape: (num_particles, N, 2)
    new_positions = np.zeros_like(current_positions)
    for i in range(num_particles):
        for rho in range(N):
            drift = np.zeros(2)
            for z in active_indices:
                temp = np.zeros(2)
                for sigma in range(N):
                    diff = current_positions[z, sigma] - current_positions[i, rho]
                    temp += K_delta(diff, delta)
                drift += (temp / N) * (w0[z]*h**2)
            dW = np.sqrt(2*nu*dt) * np.random.randn(2)
            pos_new = current_positions[i, rho] + dt * drift + dW
            pos_new = enforce_noslip_boundary(pos_new, current_positions[i, rho])
            new_positions[i, rho] = pos_new
    trajectories[step+1] = new_positions

def compute_velocity_field(positions, xx, yy, delta, active_indices, w0, h, N):
    U = np.zeros_like(xx)
    V = np.zeros_like(yy)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pos = np.array([xx[i,j], yy[i,j]])
            temp = np.zeros(2)
            for z in active_indices:
                vortex_positions = positions[z]
                inner = np.zeros(2)
                for sigma in range(N):
                    diff = vortex_positions[sigma] - pos
                    inner += K_delta(diff, delta)
                temp += (inner / N) * (w0[z]*h**2)
            U[i,j] = temp[0]
            V[i,j] = temp[1]
    return U, V

# Plot velocity fields at times 0,2,4,...,20
times = np.arange(0, T+1, 2)
step_indices = (times/dt).astype(int)
for idx, step in enumerate(step_indices):
    positions = trajectories[step]
    U, V = compute_velocity_field(positions, xx, yy, delta, active_indices, w0, h, N)
    plt.figure(figsize=(6,6))
    plt.quiver(xx, yy, U, V, pivot='mid', color='red')
    plt.title(f"Velocity Field at t = {step*dt:.1f} (No-slip Boundary)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.grid(True)
    plt.show()

# Create a 25 fps animation of the velocity field from t=0 to t=20
fig, ax = plt.subplots(figsize=(6,6))
quiv = ax.quiver(xx, yy, np.zeros_like(xx), np.zeros_like(yy), pivot='mid', color='red')
ax.set_title("Velocity Field Animation (No-slip Boundary)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.grid(True)

def update(frame):
    positions = trajectories[frame]
    U, V = compute_velocity_field(positions, xx, yy, delta, active_indices, w0, h, N)
    quiv.set_UVC(U, V)
    ax.set_title(f"t = {frame*dt:.1f} (No-slip)")
    return quiv,

anim = animation.FuncAnimation(fig, update, frames=range(0, num_steps+1), interval=40)
# To save the animation, uncomment the following line:
# anim.save('velocity_field_noslip.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
plt.show()