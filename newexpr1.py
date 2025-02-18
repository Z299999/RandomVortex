import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Problem Setup
# --------------------------------------------------------
# Domain: [-1,1] x [-1,1], discretized into 21 x 21 points
nx, ny = 21, 21
h = 0.1  # mesh size (distance between grid points)

x_vals = np.linspace(-1, 1, nx)
y_vals = np.linspace(-1, 1, ny)

# Create a list of (x, y) grid coordinates
grid_points = []
for j in range(ny):
    for i in range(nx):
        grid_points.append([x_vals[i], y_vals[j]])
grid_points = np.array(grid_points)  # shape = (441, 2)
M = len(grid_points)                 # total number of vortices

# We assume w0(0,0) = 1, zero elsewhere
w0 = np.zeros(M)
# Find index of the point (0,0) and set w0=1 there
for idx, (xx, yy) in enumerate(grid_points):
    if abs(xx) < 1e-14 and abs(yy) < 1e-14:
        w0[idx] = 1.0
        break

# Circulation (vortex strength) at each grid point
Gamma = w0 * (h**2)

# --------------------------------------------------------
# 2. Mollified Biot--Savart Kernel
# --------------------------------------------------------
def K(x):
    """
    Standard 2D Biot--Savart core (singular at x=0).
    Returns a 2D array.
    """
    r2 = x[0]**2 + x[1]**2
    if r2 < 1e-16:
        return np.array([0.0, 0.0])
    factor = 1.0/(2.0*np.pi)
    return factor * np.array([-x[1]/r2, x[0]/r2])

def K_delta(x, delta):
    """
    Mollified Biot--Savart kernel:
      K_delta(x) = K(x) * (1 - exp(-(r/delta)^2))
    """
    r = np.sqrt(x[0]**2 + x[1]**2)
    return (1 - np.exp(-(r/delta)**2)) * K(x)

# --------------------------------------------------------
# 3. Time Discretization & Brownian Motion
# --------------------------------------------------------
T = 10.0        # final time
dt = 0.1        # time step
Ksteps = int(T / dt)  # total number of steps

nu = 0.01       # viscosity
delta = 0.05    # mollification parameter

# We have N=1 path for each vortex, so we do not need a second dimension for "paths."
# We'll store each vortex's position for all time steps:
X = np.zeros((M, Ksteps+1, 2))  # shape: (numVortices, numSteps+1, spaceDim)
X[:, 0, :] = grid_points        # initial positions X(i,0) = x_i

# --------------------------------------------------------
# 4. Main Loop: Update Vortex Positions
# --------------------------------------------------------
for r in range(Ksteps):
    # For each vortex i, compute the drift from all other vortices
    for i in range(M):
        # Summation of velocity from all other vortices
        drift = np.array([0.0, 0.0])
        for z in range(M):
            if z == i:
                continue
            diff = X[z, r, :] - X[i, r, :]
            # Because N=1, we skip the average over sample paths
            drift += K_delta(diff, delta) * Gamma[z]
        # Brownian increment
        xi = np.random.normal(loc=0.0, scale=1.0, size=2)
        # Update position using Euler--Maruyama
        X[i, r+1, :] = X[i, r, :] \
                       + dt * drift \
                       + np.sqrt(2.0 * nu * dt) * xi

# --------------------------------------------------------
# 5. Compute Velocity Field at Final Time
# --------------------------------------------------------
u_final = np.zeros((M, 2))
r_final = Ksteps  # index for t = T
for i in range(M):
    tmp = np.array([0.0, 0.0])
    for z in range(M):
        diff = X[z, r_final, :] - X[i, r_final, :]
        tmp += K_delta(diff, delta) * Gamma[z]
    u_final[i, :] = tmp

# --------------------------------------------------------
# 6. (Optional) Visualization
# --------------------------------------------------------
# Example 1: Plot final vortex trajectories
plt.figure(figsize=(6,6))
for i in range(M):
    # Plot the entire trajectory of vortex i as a line
    plt.plot(X[i, :, 0], X[i, :, 1], '-', linewidth=0.8)
    # Mark final position with a dot
    plt.plot(X[i, r_final, 0], X[i, r_final, 1], 'ko', markersize=2)

plt.title("Vortex Trajectories (N=1 path each)")
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.gca().set_aspect('equal', 'box')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Example 2: Plot the final velocity field via quiver
# (In practice, we might want to use fewer points for clarity.)
plt.figure(figsize=(6,6))
plt.quiver(X[:, r_final, 0], X[:, r_final, 1], u_final[:,0], u_final[:,1],
           color='r', angles='xy', scale_units='xy')
plt.title("Final Velocity Field at t=T")
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.gca().set_aspect('equal', 'box')
plt.xlabel("x")
plt.ylabel("y")
plt.show()