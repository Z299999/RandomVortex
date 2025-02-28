import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------
# Simulation Parameters
# ---------------------------
nu = 0.015          # viscosity
T = 15              # final time (seconds)
dt = 0.1            # time step
num_steps = int(T/dt)
delta = 0.1         # mollification parameter for kernel
epsilon = 0.1       # smoothing parameter in viscosity term
h_vis = dt          # time quadrature step for viscosity term

# Mesh parameters
h0 = 1.0            # coarse mesh spacing (both x and y)
h1 = 0.5            # fine mesh spacing in x
h2 = 0.1            # fine mesh spacing in y

region_x = [-6, 6]
region_y = [-6, 6]  # overall y-domain
# For visualization, display only the upper half (physical domain)
window_x = region_x
window_y = [0, 6]

np.random.seed(42)

# ---------------------------
# Vorticity and Velocity Functions
# ---------------------------
def velocity(x, y):
    return np.array([-np.sin(y), 0])

def vorticity(x, y):
    return -2 * np.cos(y)

# ---------------------------
# Offline Computation of θ
# ---------------------------
def compute_theta_offline(num_steps, dt, nu, N_samples=1000):
    """
    Compute theta(·, t_k) on the boundary offline using:
      theta(x,t0) = -2
      theta(x,t_{k+1}) = theta(x,t_k) + (dt/N) sum psi(Y_i^{t_k}, t_k)
    Here psi is a placeholder returning 0.
    """
    Theta = []
    theta0 = -2
    Theta.append(lambda x, val=theta0: val)
    def psi(x, t):
        return 0
    theta_prev = theta0
    for k in range(1, num_steps+1):
        t_k = k * dt
        samples = np.random.normal(loc=0, scale=np.sqrt(4*nu*t_k), size=N_samples)
        psi_vals = np.array([psi(s, t_k) for s in samples])
        avg_psi = np.mean(psi_vals)
        theta_new = theta_prev + dt * avg_psi
        theta_prev = theta_new
        Theta.append(lambda x, val=theta_new: val)
    return Theta

Theta = compute_theta_offline(num_steps, dt, nu, N_samples=1000)

# ---------------------------
# Definition of φ'' (phi_dd)
# ---------------------------
def phi_dd(r):
    """
    φ''(r) = 324*(r - 0.5)  for r in [1/3, 2/3], 0 otherwise.
    """
    if 1/3 <= r <= 2/3:
        return 324 * (r - 0.5)
    else:
        return 0.0

# ---------------------------
# Symmetric Combined Grid Generation
# ---------------------------
def generate_symmetric_grids():
    """
    1. Create lower grids:
         - Coarse: [x1,x2] x [y1,y2)
         - Fine:   [x1,x2] x [y2,0)
    2. Mirror them about y=0:
         - Upper coarse grid = mirror(lower coarse)
         - Upper fine grid  = mirror(lower fine)
    3. Create a boundary mesh at y=0.
    4. Combine:
         - D0 = lower coarse ∪ upper coarse
         - Db = lower fine ∪ boundary ∪ upper fine
    Returns D0, Db, A_coarse, A_fine, N_coarse, N_fine.
    """
    x1, x2 = region_x
    y1 = region_y[0]  # -6
    y4 = region_y[1]  # 6
    y2 = -0.4         # splitting point (boundary layer thickness)

    # --- Coarse Grid ---
    # Lower coarse grid on [x1,x2] x [y1,y2)
    N_x_coarse = int((x2 - x1) / h0) + 1
    N_y_coarse = int((y2 - y1) / h0)
    x_coarse = np.linspace(x1, x2, N_x_coarse)
    y_coarse_lower = np.linspace(y1, y2, N_y_coarse, endpoint=False)
    XX_coarse, YY_coarse_lower = np.meshgrid(x_coarse, y_coarse_lower, indexing='ij')
    D02 = np.stack((XX_coarse, YY_coarse_lower), axis=-1)
    # Mirror to get upper coarse grid
    D01 = D02.copy()
    D01[:,:,1] = -D01[:,:,1]
    D01 = D01[:, ::-1, :]
    # Combine coarse grids
    D0 = np.concatenate((D02, D01), axis=1)
    N_coarse = N_y_coarse  # vertical points in one half

    # --- Fine Grid ---
    # Lower fine grid on [x1,x2] x [y2, 0)
    N_x_fine = int((x2 - x1) / h1) + 1
    N_y_fine = int((0 - y2) / h2)
    x_fine = np.linspace(x1, x2, N_x_fine)
    y_fine_lower = np.linspace(y2, 0, N_y_fine, endpoint=False)
    XX_fine, YY_fine_lower = np.meshgrid(x_fine, y_fine_lower, indexing='ij')
    Db2 = np.stack((XX_fine, YY_fine_lower), axis=-1)
    # Mirror to get upper fine grid
    Db1 = Db2.copy()
    Db1[:,:,1] = -Db1[:,:,1]
    Db1 = Db1[:, ::-1, :]
    # Boundary mesh at y = 0
    Dbd = np.stack((x_fine, np.zeros_like(x_fine)), axis=-1)
    Dbd = Dbd[:, None, :]
    # Combine fine grids
    Db = np.concatenate((Db2, Dbd, Db1), axis=1)
    A_coarse = h0 * h0
    A_fine = h1 * h2

    return D0, Db, A_coarse, A_fine, N_coarse, N_y_fine

# Use the new symmetric grid generation
D0, Db, A_coarse, A_fine, N_coarse, N_fine = generate_symmetric_grids()

# ---------------------------
# Vortex Initialization on Combined Meshes
# ---------------------------
def initialize_vortices(grid):
    M, N, _ = grid.shape
    w = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x, y = grid[i, j]
            w[i, j] = vorticity(x, y)
    return w

w0_D0 = initialize_vortices(D0)
w0_Db = initialize_vortices(Db)

# ---------------------------
# Helper: Indicator Function
# ---------------------------
def indicator_D(point):
    return 1 if point[1] > 0 else 0

# ---------------------------
# Mollified Biot-Savart Kernel
# ---------------------------
def K_delta(x, y, delta=0.01):
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    r2 = (x1 - y1)**2 + (x2 - y2)**2
    r2_bar = (x1 - y1)**2 + (-x2 - y2)**2
    if r2 < 1e-10 or r2_bar < 1e-10:
        return np.zeros(2)
    k1 = 0.5/np.pi * ((y2 - x2)/r2 - (y2 + x2)/r2_bar)
    k2 = 0.5/np.pi * ((y1 - x1)/r2_bar - (y1 - x1)/r2)
    factor = 1 - np.exp(- (r2/delta)**2)
    return np.array([k1, k2]) * factor

# ---------------------------
# u_func Factory with Viscosity Term
# ---------------------------
def make_u_func(curr_D0, curr_Db, w0_D0, w0_Db, A_coarse, A_fine,
                N_coarse, N_fine, current_step, traj_Db_history, gamma_Db_current, Theta, epsilon):
    def u_func(x):
        u_val = np.zeros(2)
        # Vortex contribution from coarse grid
        M, total = curr_D0.shape[0], curr_D0.shape[1]
        for i in range(M):
            for j in range(N_coarse, total):  # use upper (physical) vortices only
                pos_upper = curr_D0[i, j]
                mirror_index = 2 * N_coarse - 1 - j
                pos_lower = curr_D0[i, mirror_index]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                u_val += A_coarse * w0_D0[i, j] * (term_upper - term_lower)
        # Vortex contribution from fine grid
        M_f, total_f = curr_Db.shape[0], curr_Db.shape[1]  # total_f = 2*N_fine + 1
        for i in range(M_f):
            # Loop over upper fine grid: skip boundary at index N_fine
            for j in range(N_fine+1, total_f):
                pos_upper = curr_Db[i, j]
                mirror_index = 2 * N_fine - j  # correct mirror index for fine grid
                pos_lower = curr_Db[i, mirror_index]
                term_upper = indicator_D(pos_upper) * K_delta(pos_upper, x, delta)
                term_lower = indicator_D(pos_lower) * K_delta(pos_lower, x, delta)
                diffK = term_upper - term_lower
                u_val += A_fine * w0_Db[i, j] * diffK

        # --- Viscosity contribution from fine grid ---
        u_vis = np.zeros(2)
        # for i in range(M_f):
        #     for j in range(N_fine+1, total_f):
        #         pos_current = curr_Db[i, j]
        #         K_val = K_delta(pos_current, x, delta)
        #         vis_sum = 0.0
        #         for l in range(current_step + 1):
        #             t_l = l * dt
        #             if t_l > gamma_Db_current[i, j]:
        #                 X_l = traj_Db_history[l][i, j]
        #                 theta_val = Theta[l](X_l[0])
        #                 phi_val = phi_dd(X_l[1] / epsilon)
        #                 vis_sum += theta_val * phi_val
        #         u_vis += K_val * vis_sum
        # u_vis *= (nu / epsilon**2) * (h1 * h2 * h_vis)
        return u_val + u_vis
    return u_func

# ---------------------------
# Vortex Trajectory Simulation (Euler-Maruyama) with γ Update
# ---------------------------
def simulate_vortex_trajectories():
    traj_D0 = np.zeros((num_steps+1,) + D0.shape)
    traj_Db = np.zeros((num_steps+1,) + Db.shape)
    traj_D0[0] = D0
    traj_Db[0] = Db

    M_f, total_f = Db.shape[0], Db.shape[1]
    gamma_Db = np.full((num_steps+1, M_f, total_f), -np.inf)

    uFuncs = []
    traj_Db_history = [traj_Db[0].copy()]
    
    for step in range(num_steps):
        # Update coarse grid D0
        curr_D0 = traj_D0[step].copy()
        new_D0 = np.zeros_like(curr_D0)
        M, total = curr_D0.shape[0], curr_D0.shape[1]
        for i in range(M):
            for j in range(total):
                dW = np.sqrt(2*nu*dt) * np.random.randn(2)
                new_D0[i, j] = curr_D0[i, j] + dW
        traj_D0[step+1] = new_D0

        # Update fine grid Db
        curr_Db = traj_Db[step].copy()
        new_Db = np.zeros_like(curr_Db)
        for i in range(M_f):
            for j in range(total_f):
                dW = np.sqrt(2*nu*dt) * np.random.randn(2)
                new_Db[i, j] = curr_Db[i, j] + dW
        traj_Db[step+1] = new_Db
        traj_Db_history.append(new_Db.copy())
        
        # Update gamma for fine grid
        for i in range(M_f):
            for j in range(total_f):
                if new_Db[i, j, 1] <= 0:
                    gamma_Db[step+1, i, j] = (step+1)*dt
                else:
                    gamma_Db[step+1, i, j] = gamma_Db[step, i, j]
                    
        u_func = make_u_func(traj_D0[step+1], traj_Db[step+1],
                             w0_D0, w0_Db, A_coarse, A_fine,
                             N_coarse, N_fine, step+1,
                             traj_Db_history, gamma_Db[step+1], Theta, epsilon)
        uFuncs.append(u_func)
    return (traj_D0, traj_Db, gamma_Db), uFuncs

print("Simulating vortex trajectories...")
(trajectories_D0, trajectories_Db, gamma_Db), uFuncs = simulate_vortex_trajectories()

# ---------------------------
# Boat Simulation (Using Physical/Upper Half of Combined Meshes)
# ---------------------------
def generate_boat_grid():
    boat_coarse = trajectories_D0[0][:, N_coarse:, :].reshape(-1, 2)
    boat_fine = trajectories_Db[0][:, N_fine+1:, :].reshape(-1, 2)  # skip the boundary row
    boat_grid = np.concatenate((boat_coarse, boat_fine), axis=0)
    return boat_grid

def simulate_boats(uFuncs):
    boat_grid = generate_boat_grid()
    num_boats = boat_grid.shape[0]
    boat_positions = np.zeros((num_steps+1, num_boats, 2))
    boat_positions[0] = boat_grid
    boat_displacements = np.zeros((num_steps, num_boats, 2))
    for step in range(num_steps):
        u_func = uFuncs[step]
        for b in range(num_boats):
            vel = u_func(boat_positions[step, b])
            boat_displacements[step, b] = dt * vel
            boat_positions[step+1, b] = boat_positions[step, b] + dt * vel
    return boat_positions, boat_displacements

print("Simulating boat trajectories...")
boat_positions, boat_displacements = simulate_boats(uFuncs)

def compute_velocity_field(u_func, query_points):
    P = query_points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)
    for p in range(P):
        vel = u_func(query_points[p])
        U[p] = vel[0]
        V[p] = vel[1]
    return U, V

def generate_query_grid():
    boat_coarse = trajectories_D0[0][:, N_coarse:, :].reshape(-1, 2)
    boat_fine = trajectories_Db[0][:, N_fine+1:, :].reshape(-1, 2)
    query_grid = np.concatenate((boat_coarse, boat_fine), axis=0)
    return query_grid

query_grid = generate_query_grid()

# ---------------------------
# Animation: Velocity Field and Boat Trajectories
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(window_x[0], window_x[1])
ax.set_ylim(window_y[0], window_y[1])
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Vortex and Boat Animation (t=0.00)")

U, V = compute_velocity_field(uFuncs[0], query_grid)
vel_quiver = ax.quiver(query_grid[:, 0], query_grid[:, 1], U, V,
                       color='black', alpha=0.9, pivot='mid',
                       scale=None, angles='xy', scale_units='xy')

boat_scatter = ax.scatter(query_grid[:, 0], query_grid[:, 1],
                          s=10, color='blue', zorder=3)

def update(frame):
    t_current = frame * dt
    u_func = uFuncs[frame] if frame < len(uFuncs) else uFuncs[-1]
    U, V = compute_velocity_field(u_func, query_grid)
    vel_quiver.set_UVC(U, V)
    boat_scatter.set_offsets(boat_positions[frame])
    ax.set_title(f"Vortex and Boat Animation (t={t_current:.2f})")
    return vel_quiver, boat_scatter

anim = FuncAnimation(fig, update, frames=num_steps+1, interval=40, blit=False)
os.makedirs("animation", exist_ok=True)
save_path = os.path.join("animation", "combined_viscous_corrReflec.mp4")
writer = FFMpegWriter(fps=25)
anim.save(save_path, writer=writer)
print(f"Animation saved at: {save_path}")

plt.close(fig)
