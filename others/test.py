import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters and domain setup
# -----------------------------
nu = 0.1          # kinematic viscosity
U0 = 0.01         # magnitude for initial velocity / vorticity
dt = 0.01         # time step size
T = 1.0           # total simulation time
Nsteps = int(T/dt)
Nparticles = 1000  # number of vortex particles

# Domain: x1 in [-H, H], x2 in [0, H]
H = 6.0

# -----------------------------
# Initialize vortex particles
# -----------------------------
# For simplicity we initialize particles uniformly in the domain.
x_particles = np.random.uniform(-H, H, size=(Nparticles, 1))
y_particles = np.random.uniform(0, H, size=(Nparticles, 1))
particles = np.hstack((x_particles, y_particles))

# All particles are assigned the same vorticity weight.
weights = U0 * np.ones(Nparticles)

# -----------------------------
# Define the Biot--Savart kernel
# -----------------------------
def biot_savart(x, y, delta=0.1):
    """
    Compute the 2D Biot-Savart kernel contribution from a vortex at y to a field point x.
    A regularization parameter delta is used to avoid singularities.
    """
    diff = x - y
    r2 = np.sum(diff**2) + delta**2
    # The perpendicular vector: (-diff[1], diff[0])
    vel = (1/(2*np.pi)) * np.array([-diff[1], diff[0]]) / r2
    return vel

def compute_velocity(x, particles, weights, delta=0.1):
    """
    Compute the velocity at point x due to all vortex particles.
    """
    vel = np.zeros(2)
    for i in range(len(particles)):
        vel += weights[i] * biot_savart(x, particles[i], delta)
    return vel

# -----------------------------
# Euler–Maruyama update for particles
# -----------------------------
def update_particles(particles, weights, dt, nu, delta=0.1):
    """
    Update the positions of the vortex particles.
    The position is updated via:
      x_{n+1} = x_n + u(x_n)*dt + sqrt(2*nu*dt)*N(0, I)
    The wall is enforced by reflecting any particle that crosses x2 < 0.
    """
    new_particles = particles.copy()
    for i in range(len(particles)):
        x = particles[i]
        u = compute_velocity(x, particles, weights, delta)
        noise = np.sqrt(2 * nu * dt) * np.random.randn(2)
        new_particles[i] = x + u * dt + noise
        # Reflect particles that cross the wall (x2 < 0)
        if new_particles[i, 1] < 0:
            new_particles[i, 1] = -new_particles[i, 1]
    return new_particles

# -----------------------------
# Update boundary vorticity theta
# -----------------------------
# We discretize the boundary (x2=0) along x1.
Nx = 100
x_boundary = np.linspace(-H, H, Nx)
theta = U0 * np.ones(Nx)  # initial boundary vorticity

def update_theta(theta, dx, dt, nu):
    """
    Update the boundary vorticity theta using an explicit finite difference scheme 
    for the heat equation: dtheta/dt = 2*nu * d^2 theta/dx^2.
    Here, psi and higher-order terms are neglected for simplicity.
    """
    theta_new = theta.copy()
    # Use central differences for the second derivative.
    for i in range(1, len(theta)-1):
        d2theta = (theta[i+1] - 2*theta[i] + theta[i-1]) / dx**2
        theta_new[i] = theta[i] + dt * (2 * nu * d2theta)
    # Apply Neumann boundary conditions (zero-flux)
    theta_new[0] = theta_new[1]
    theta_new[-1] = theta_new[-2]
    return theta_new

dx = (2*H) / (Nx - 1)

# -----------------------------
# Main simulation loop
# -----------------------------
particles_list = []  # store particle positions for visualization
theta_list = []      # store boundary vorticity profiles

for n in range(Nsteps):
    # Update the vortex particle positions
    particles = update_particles(particles, weights, dt, nu)
    # Update the boundary vorticity
    theta = update_theta(theta, dx, dt, nu)
    
    particles_list.append(particles.copy())
    theta_list.append(theta.copy())

# -----------------------------
# Plotting results
# -----------------------------
# Final particle positions
plt.figure(figsize=(8,6))
plt.scatter(particles[:,0], particles[:,1], s=5, alpha=0.5)
plt.xlim(-H, H)
plt.ylim(0, H)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Particle positions at final time')
plt.show()

# Final boundary vorticity profile
plt.figure(figsize=(8,4))
plt.plot(x_boundary, theta, 'b-', lw=2)
plt.xlabel('$x_1$')
plt.ylabel(r'$\theta(x_1,t)$')
plt.title('Boundary vorticity at final time')
plt.show()
