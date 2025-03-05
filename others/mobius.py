import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # ensure 3D plotting

# ------------------------------
# Parameters and grid setup
# ------------------------------
a = 0.25               # half-width of the strip (ζ ∈ [-a, a])
N_zeta = 100           # number of grid points in ζ
N_theta = 200          # number of grid points in θ
T_final = 0.1          # total simulation time
dt = 0.001             # time step for vorticity update
epsilon = 1e-7         # small-scale dissipation coefficient

# Create grids in ζ and θ.
zeta = np.linspace(-a, a, N_zeta)
theta = np.linspace(0, np.pi, N_theta)
dz = zeta[1] - zeta[0]
dtheta = theta[1] - theta[0]
# Meshgrid with indexing='ij': first index is ζ, second is θ.
Z, Theta = np.meshgrid(zeta, theta, indexing='ij')

# ------------------------------
# Compute metric factors from the paper
# |g|(ζ,θ) = 4 + 8*cos(θ)*ζ + (3 + 2*cos(2θ))*ζ².
# ------------------------------
g = 4 + 8 * np.cos(Theta) * Z + (3 + 2 * np.cos(2 * Theta)) * Z**2
A = np.sqrt(g)       # A = |g|^(1/2)
B = 1.0 / A          # B = |g|^(-1/2)

# ------------------------------
# Define Laplacian, Poisson solver, and time-stepping routines
# ------------------------------
def laplacian_np(psi, dz, dtheta, A, B):
    """
    Compute the Laplacian ∆ψ defined as:
      ∆ψ = B * [∂ζ(A ∂ζψ) + ∂θ(B ∂θψ)]
    using np.gradient.
    """
    dpsi_dz = np.gradient(psi, dz, axis=0)
    dpsi_dtheta = np.gradient(psi, dtheta, axis=1)
    term_z = np.gradient(A * dpsi_dz, dz, axis=0)
    term_theta = np.gradient(B * dpsi_dtheta, dtheta, axis=1)
    return B * (term_z + term_theta)

def poisson_solver_np(omega, psi_init, dz, dtheta, A, B, tol=1e-6, max_iter=1000, relax=0.1):
    """
    Solve for ψ in the Poisson equation ∆ψ = ω via fixed-point iteration:
      ψ ← ψ + relax*(ω - ∆ψ)
    """
    psi = psi_init.copy()
    for it in range(max_iter):
        lap = laplacian_np(psi, dz, dtheta, A, B)
        res = omega - lap
        psi += relax * res
        err = np.linalg.norm(res)
        if err < tol:
            # Uncomment the next line for convergence info:
            # print(f"Poisson solver converged in {it} iterations, residual {err:.2e}")
            break
    return psi

def step_omega(omega, psi, dz, dtheta, A, B, dt, epsilon):
    """
    One time-step update:
      1. Solve ∆ψ = ω for ψ.
      2. Compute the Jacobian J(ψ,ω) = ψ_ζ ω_θ - ψ_θ ω_z.
      3. Update ω via: ω_new = ω - dt*(B*J) + dt*ε*∆ω.
         (B appears as a factor in the advection term.)
    """
    psi = poisson_solver_np(omega, psi, dz, dtheta, A, B)
    
    psi_z = np.gradient(psi, dz, axis=0)
    psi_theta = np.gradient(psi, dtheta, axis=1)
    omega_z = np.gradient(omega, dz, axis=0)
    omega_theta = np.gradient(omega, dtheta, axis=1)
    
    J = psi_z * omega_theta - psi_theta * omega_z
    lap_omega = laplacian_np(omega, dz, dtheta, A, B)
    
    omega_new = omega - dt * (B * J) + dt * epsilon * lap_omega
    # Enforce ω = 0 on ζ boundaries.
    omega_new[0, :] = 0
    omega_new[-1, :] = 0
    return omega_new, psi

# ------------------------------
# Initial condition for vorticity ω
# A Gaussian vortex patch placed near ζ = a.
# ------------------------------
zeta0 = a - 0.05
theta0 = np.pi / 2
sigma = 0.02
omega = np.exp(-(((Z - zeta0)**2 + (Theta - theta0)**2) / (2 * sigma**2)))
omega[0, :] = 0
omega[-1, :] = 0
psi = np.zeros_like(omega)

# ------------------------------
# Parameterization of the Möbius band in 3D
# Given: x = (1 + ζ cos θ) cos(2θ),  y = (1 + ζ cos θ) sin(2θ),  z = ζ sin θ.
# ------------------------------
def mobius_coordinates(Z, Theta):
    X = (1 + Z * np.cos(Theta)) * np.cos(2 * Theta)
    Y = (1 + Z * np.cos(Theta)) * np.sin(2 * Theta)
    Z3 = Z * np.sin(Theta)
    return X, Y, Z3

# ------------------------------
# 3D Visualization setup using mplot3d
# ------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
cmap = cm.get_cmap('RdBu')
norm = plt.Normalize(vmin=-1, vmax=1)

# Compute initial 3D coordinates.
X, Y, Z3 = mobius_coordinates(Z, Theta)
facecolors = cmap(norm(omega))

# Initial 3D surface plot.
surf = ax.plot_surface(X, Y, Z3, facecolors=facecolors, rstride=1, cstride=1, antialiased=False)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-0.5, 0.5])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# ------------------------------
# Animation update function
# ------------------------------
def update(frame):
    global omega, psi, surf
    # Evolve the simulation a few time-steps per frame.
    for _ in range(5):
        omega, psi = step_omega(omega, psi, dz, dtheta, A, B, dt, epsilon)
    # Update face colors using the updated ω.
    facecolors = cmap(norm(omega))
    # The 3D coordinates remain the same (static Möbius band), so recompute for clarity.
    X, Y, Z3 = mobius_coordinates(Z, Theta)
    
    ax.clear()  # Clear the previous frame.
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-0.5, 0.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Plot the surface with updated colors.
    surf = ax.plot_surface(X, Y, Z3, facecolors=facecolors, rstride=1, cstride=1, antialiased=False)
    return surf,

# ------------------------------
# Create animation
# ------------------------------
ani = FuncAnimation(fig, update, frames=int(T_final/dt/5), interval=50, blit=False)
plt.show()
