import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# Parameters
# ---------------------------
np.random.seed(42)
nu = 0.001          # Viscosity
T = 10.0            # Final time
dt = 0.1
num_steps = int(T/dt)
N = 10              # Number of sample paths per vortex
num_vortices = 5   # Number of vortices
delta = 0.1         # Mollification parameter

# ---------------------------
# Define Mollified Biot--Savart Kernel
# ---------------------------
def K_delta(x, delta):
    """
    Computes the mollified Biot--Savart kernel.
    K(x) = (1/(2π)) * (-x_2/|x|², x_1/|x|²)
    K_δ(x) = K(x)*[1 - exp(-( |x|/δ )²)].
    """
    r = np.linalg.norm(x)
    if r < 1e-6:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(-(r/delta)**2)
    return (1/(2*np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

# ---------------------------
# Initialize Vortices
# ---------------------------
# Random initial positions in [-1,1]×[-1,1]
vortex_positions = np.random.uniform(-1, 1, (num_vortices, 2))
# Random strengths (circulations) for each vortex (positive or negative)
# vortex_strengths = np.array([1,-1,1,-1,1])
vortex_strengths = np.random.uniform(-1, 1, num_vortices)

print("Initial vortex positions:")
print(vortex_positions)
print("Vortex strengths:")
print(vortex_strengths)

# ---------------------------
# Initialize Trajectories
# ---------------------------
# trajectories: shape (num_steps+1, num_vortices, N, 2)
trajectories = np.zeros((num_steps+1, num_vortices, N, 2))
for i in range(num_vortices):
    for rho in range(N):
        trajectories[0, i, rho, :] = vortex_positions[i]

# ---------------------------
# Time-stepping: Update Each Vortex's Sample Paths
# ---------------------------
for step in range(num_steps):
    current = trajectories[step]  # shape: (num_vortices, N, 2)
    new = np.zeros_like(current)
    for i in range(num_vortices):
        for rho in range(N):
            drift = np.zeros(2)
            # Sum contributions from all other vortices
            for z in range(num_vortices):
                if z == i:
                    continue
                sum_over_samples = np.zeros(2)
                for sigma in range(N):
                    diff = current[z, sigma] - current[i, rho]
                    sum_over_samples += K_delta(diff, delta)
                drift += (sum_over_samples / N) * vortex_strengths[z]
            # Brownian increment
            noise = np.sqrt(2 * nu * dt) * np.random.randn(2)
            new[i, rho] = current[i, rho] + dt*drift + noise
    trajectories[step+1] = new

# ---------------------------
# Compute the Velocity Field
# ---------------------------
def compute_velocity_field(trajectories_at_t, grid_x, grid_y, delta, vortex_strengths, N):
    """
    Computes the velocity field u(x,t) at a given time step.
    u(x,t) = sum_{z=1..num_vortices} [ (1/N) sum_{sigma=1}^N K_δ( X_z^σ(t) - x ) * Γ_z ].
    """
    U = np.zeros_like(grid_x)
    V = np.zeros_like(grid_y)
    num_vortices_local = len(vortex_strengths)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            pos = np.array([grid_x[i, j], grid_y[i, j]])
            velocity = np.zeros(2)
            for z in range(num_vortices_local):
                sum_over_samples = np.zeros(2)
                for sigma in range(N):
                    diff = trajectories_at_t[z, sigma] - pos
                    sum_over_samples += K_delta(diff, delta)
                velocity += (sum_over_samples / N) * vortex_strengths[z]
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
    return U, V

# ---------------------------
# Set up Grid for Velocity Field Plots
# ---------------------------
grid_res = 50
x_vals = np.linspace(-1,1,grid_res)
y_vals = np.linspace(-1,1,grid_res)
X, Y = np.meshgrid(x_vals, y_vals)

# ---------------------------
# 1) Static Plots of Velocity Field + Vortex Trajectories
#    at selected times.
# ---------------------------
times_to_plot = [0, 2, 4, 6, 8, 10]  # in time units
step_indices = [int(t/dt) for t in times_to_plot]

for step_idx, t_val in zip(step_indices, times_to_plot):
    U, V = compute_velocity_field(trajectories[step_idx], X, Y, delta, vortex_strengths, N)

    plt.figure(figsize=(7,7))
    plt.quiver(X, Y, U, V, pivot='mid', color='blue')
    plt.title(f'Velocity Field at t = {t_val:.1f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.grid(True)

    # Plot the trajectory for each vortex sample path from time 0 to step_idx
    for i in range(num_vortices):
        for rho in range(N):
            x_traj = trajectories[:step_idx+1, i, rho, 0]
            y_traj = trajectories[:step_idx+1, i, rho, 1]
            plt.plot(x_traj, y_traj, color='red', alpha=0.3, linewidth=1)

    # Mark the final position at this time
    for i in range(num_vortices):
        for rho in range(N):
            xf = trajectories[step_idx, i, rho, 0]
            yf = trajectories[step_idx, i, rho, 1]
            plt.plot(xf, yf, 'ko', markersize=2)

    plt.show()

# ---------------------------
# 2) Animation of the Velocity Field + Vortex Paths
# ---------------------------
animate = True
if animate:
    fig, ax = plt.subplots(figsize=(7,7))

    # 2a) Create initial velocity field and quiver
    U0, V0 = compute_velocity_field(trajectories[0], X, Y, delta, vortex_strengths, N)
    quiv = ax.quiver(X, Y, U0, V0, pivot='mid', color='blue')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Field at t = 0.00')
    ax.grid(True)

    # 2b) Create line objects for each vortex's sample path
    #     We'll have num_vortices*N lines in total.
    lines = []
    for i in range(num_vortices):
        for rho in range(N):
            (line,) = ax.plot([], [], color='red', lw=1, alpha=0.5)
            lines.append(line)

    def update(frame):
        """
        For each frame = step index, update:
          1) The velocity field quiver
          2) Each vortex path line from step=0..frame
        """
        # (1) Update the velocity field
        U, V = compute_velocity_field(trajectories[frame], X, Y, delta, vortex_strengths, N)
        quiv.set_UVC(U, V)

        # (2) Update each vortex path line
        # lines[n] corresponds to the path for vortex i, sample rho
        # where i = n // N, rho = n % N
        for n, line in enumerate(lines):
            i = n // N
            rho = n % N
            x_data = trajectories[:frame+1, i, rho, 0]
            y_data = trajectories[:frame+1, i, rho, 1]
            line.set_data(x_data, y_data)

        # Update plot title
        ax.set_title(f'Velocity Field at t = {frame*dt:.2f}')
        return [quiv, *lines]

    # 2c) Create and show the animation
    anim = animation.FuncAnimation(
        fig, update,
        frames=range(0, num_steps+1),  # or skip frames if desired
        interval=40,                  # 40 ms/frame => 25 fps
        blit=False
    )
    plt.show()
    # To save the animation (requires ffmpeg), do something like:
    # anim.save('velocity_field_animation.mp4', fps=25, extra_args=['-vcodec', 'libx264'])