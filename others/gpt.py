import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

# ---------------------------
# PARAMETERS
# ---------------------------
nu = 0.001            # viscosity (adjust as desired)
T = 10.0              # final time
dt = 0.1              # time step, so that t0=0, t1=0.1, ..., T=10
num_steps = int(T/dt)
h = 0.1               # spatial mesh size
delta = 0.1           # mollification parameter

# In the discrete approximation the integral is approximated by a sum.
# To have unit circulation we want: w0(center)*h^2 = 1.
# Thus we set:
w0_center = 1.0/(h**2)   # equals 100

# ---------------------------
# CREATE SPATIAL GRID
# ---------------------------
# 21 points in each direction on [-1,1]
x_values = np.linspace(-1, 1, 21)
y_values = np.linspace(-1, 1, 21)
xx, yy = np.meshgrid(x_values, y_values)
grid_points = np.column_stack((xx.flatten(), yy.flatten()))
num_particles = grid_points.shape[0]

# Identify the "center" particle (closest to (0,0))
distances = np.linalg.norm(grid_points, axis=1)
center_index = np.argmin(distances)
print("Center particle index:", center_index, "position:", grid_points[center_index])

# ---------------------------
# INITIAL CONDITIONS
# ---------------------------
# For each particle, the initial position is the grid point.
# (Note: In a Lagrangian method, these particles will move.)
positions = grid_points.copy()  # shape (num_particles, 2)

# We store the entire trajectory for each particle:
# trajectories[time_index, particle_index, coordinate]
trajectories = np.zeros((num_steps + 1, num_particles, 2))
trajectories[0] = positions

# ---------------------------
# DEFINE THE MOLLIFIED KERNEL
# ---------------------------
def K_delta(x, delta):
    """
    Mollified Biot-Savart kernel.
    
    Parameters:
      x : 2D numpy array (vector)
      delta : mollification parameter
     
    Returns:
      2D numpy array: K_delta(x)
      
    Here K(x) = (1/(2*pi)) * (-x2/|x|^2, x1/|x|^2) and we set
      K_delta(x)=K(x)*(1-exp[-(|x|/delta)^2]).
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(- (r / delta)**2)
    return (1/(2*np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

# ---------------------------
# TIME-STEPPING: SIMULATE THE SDE
# ---------------------------
# According to the discretization, for each particle (indexed by i) at time step r:
#   if i is the center: drift = 0.
#   else: drift = K_delta( X_center - X_i ) * (w0_center*h^2).
# (Note: w0_center*h^2 = 1.)
# Then add the Brownian increment with variance 2*nu*dt.
for step in range(num_steps):
    current_positions = trajectories[step].copy()  # shape (num_particles, 2)
    new_positions = np.zeros_like(current_positions)
    # Get the current center position
    center_pos = current_positions[center_index]
    
    # Update each particle:
    for i in range(num_particles):
        if i == center_index:
            # Center particle: no drift (only Brownian motion)
            drift = np.array([0.0, 0.0])
        else:
            # Only the center has nonzero vorticity.
            # So the drift for particle i is: K_delta(center - X_i)
            diff = center_pos - current_positions[i]
            drift = K_delta(diff, delta) * (w0_center * h**2)  # equals K_delta(diff, delta)
        # Brownian increment: sample from N(0, dt*I) and scale by sqrt(2*nu)
        dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
        new_positions[i] = current_positions[i] + dt * drift + dW
        
    trajectories[step + 1] = new_positions

# ---------------------------
# PLOTTING THE RESULTS
# ---------------------------

# (A) Plot particle trajectories.
plt.figure(figsize=(8,8))
for i in range(num_particles):
    # Plot each trajectory in light blue.
    plt.plot(trajectories[:, i, 0], trajectories[:, i, 1],
             lw=0.5, color='blue', alpha=0.5)
# Highlight the center particle in red.
plt.plot(trajectories[:, center_index, 0],
         trajectories[:, center_index, 1],
         lw=2, color='red', label='Center Particle')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Particle Trajectories')
plt.legend()
plt.axis('equal')
plt.grid(True)

# (B) Compute and plot the velocity field at final time T.
# According to the formula,
#   u(x, T) = sum_z [K_delta(X_delta(z,T)-x) w0(x_z) h^2]
# but only the center has nonzero w0, so for any x,
#   u(x, T) = K_delta(X_center(T) - x) * (w0_center*h^2) = K_delta(X_center(T) - x).
# We compute this on the same grid as before.
X, Y = np.meshgrid(x_values, y_values)
U = np.zeros_like(X)
V = np.zeros_like(Y)

center_final = trajectories[-1, center_index]  # final position of the center particle
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i, j], Y[i, j]])
        diff = center_final - pos
        velocity = K_delta(diff, delta) * (w0_center * h**2)  # equals K_delta(diff, delta)
        U[i, j] = velocity[0]
        V[i, j] = velocity[1]

plt.figure(figsize=(8,8))
plt.quiver(X, Y, U, V, pivot='mid', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity Field at Final Time T')
plt.axis('equal')
plt.grid(True)

plt.show()





'''
1. does trajectory represents evolution of random variables, or fluid blobs in the vortex dynamics?
2. why does the center position move? in perfect vortex dynamic, should the center position be fixed thus the dynamic system is steady state (independent of time)
3. i run this code for several times, but the trajectories seem to be the same every time. that is happening? where is randomness?
4. i think this code may work. this means you quite understand my document. in this case. we may increase difficulty:
4a) remain other conditions, only increase number N: to 10, and to 100 respectively. then compare the cases of 1, 10, 100 (by plotting and comparing the final velocity field or some other methods)
4b) ramain other conditions, only increase number of vortices. let's say, there are two points with initial vortices 1, non zero. so these two vortices will act to each other. (For the plot, plot time evolution at 5 different time stages) 
'''