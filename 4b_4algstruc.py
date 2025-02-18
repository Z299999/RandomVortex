import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ALGORITHM 1: Inputs and Outputs
# ----------------------------------------------------------
def define_inputs_and_outputs():
    """
    Returns a dictionary of all parameters (inputs), plus space
    for future 'outputs' as needed. 
    """
    params = {}

    # Basic simulation parameters
    params['nu']    = 0.001   # Viscosity
    params['T']     = 10.0    # Final time
    params['dt']    = 0.1     # Time step
    params['h']     = 0.1     # Spatial mesh size
    params['delta'] = 0.1     # Mollification parameter
    params['N']     = 10      # Number of sample paths per vortex
    np.random.seed(42)        # For reproducibility; remove for fresh randomness

    # Derived values
    params['num_steps'] = int(params['T'] / params['dt'])

    # Create a grid on [-1,1]^2
    x_values = np.linspace(-1, 1, 21)
    y_values = np.linspace(-1, 1, 21)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))

    params['xx']          = xx
    params['yy']          = yy
    params['grid_points'] = grid_points
    params['num_particles'] = grid_points.shape[0]

    # Define a discrete vorticity array (w0)
    w0 = np.zeros(params['num_particles'])

    # Just as an example, pick two "active vortex" positions
    def find_closest_index(target, points):
        dists = np.linalg.norm(points - target, axis=1)
        return np.argmin(dists)

    index1 = find_closest_index(np.array([-0.5, 0]), grid_points)
    index2 = find_closest_index(np.array([ 0.5, 0]), grid_points)
    active_indices = [index1, index2]

    # Opposite-signed vorticities at those two points
    w0[index1] =  1.0 / (params['h']**2)
    w0[index2] = -1.0 / (params['h']**2)

    params['active_indices'] = active_indices
    params['w0'] = w0

    print("Active vortex indices:", active_indices)
    print("Active vortex positions:", grid_points[active_indices])

    # Return everything in a dictionary
    return params


# ----------------------------------------------------------
# ALGORITHM 2: Subroutine for K_delta
# ----------------------------------------------------------
def K_delta(x, delta):
    """
    Mollified Biot–Savart kernel:
      K(x) = (1/(2*pi)) * (-x2/|x|^2, x1/|x|^2)
      K_delta(x) = K(x) * [1 - exp(-(|x|/delta)^2)]
    """
    r = np.linalg.norm(x)
    if r < 1e-10:
        return np.array([0.0, 0.0])
    factor = 1 - np.exp(- (r / delta)**2)
    return (1.0 / (2.0 * np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor


# ----------------------------------------------------------
# ALGORITHM 3: Compute Velocity Field
# ----------------------------------------------------------
def compute_velocity_field(positions, w0, h, delta, N, points):
    """
    Given:
      - positions: array of shape (num_particles, N, 2)
                   current positions X_\delta^(rho) for each particle
      - w0:        vorticity array of shape (num_particles,)
      - h:         mesh size
      - delta:     mollification param
      - N:         number of sample paths
      - points:    array of shape (P,2) where we want the velocity

    Returns:
      U, V: velocity components at each point in 'points',
            each shape (P,)
    """
    P = points.shape[0]
    U = np.zeros(P)
    V = np.zeros(P)

    for p_idx in range(P):
        pos = points[p_idx]  # The point where we compute velocity
        vel = np.zeros(2)

        # Sum the contribution from each particle
        for i in range(positions.shape[0]):
            # Average over N paths
            tmp = np.zeros(2)
            for rho in range(N):
                diff = positions[i, rho] - pos
                tmp += K_delta(diff, delta)
            # Multiply by w0[i]*h^2 / N
            vel += (tmp / N) * (w0[i] * (h**2))

        U[p_idx] = vel[0]
        V[p_idx] = vel[1]

    return U, V


# ----------------------------------------------------------
# ALGORITHM 4: Main Vortex Simulation
# ----------------------------------------------------------
def simulate_vortex_trajectories(params):
    """
    Runs the main time-stepping loop to update vortex positions.
    Returns the full 'trajectories' array of shape:
      (num_steps+1, num_particles, N, 2)
    where N = params['N'] is the number of sample paths per particle.
    """
    # Unpack parameters
    num_steps     = params['num_steps']
    num_particles = params['num_particles']
    N            = params['N']
    dt           = params['dt']
    nu           = params['nu']
    h            = params['h']
    delta        = params['delta']
    w0           = params['w0']
    active_indices = params['active_indices']
    grid_points  = params['grid_points']

    # Initialize the trajectory array
    # shape: (num_steps+1, num_particles, N, 2)
    trajectories = np.zeros((num_steps + 1, num_particles, N, 2))
    for i in range(num_particles):
        trajectories[0, i, :, :] = grid_points[i]  # all sample paths start at same location

    # Time stepping
    for step in range(num_steps):
        current_positions = trajectories[step]  # shape: (num_particles, N, 2)
        new_positions = np.zeros_like(current_positions)

        for i in range(num_particles):
            for rho in range(N):
                # Compute drift from active vortices only (like your original code)
                drift = np.zeros(2)
                for z in active_indices:
                    temp = np.zeros(2)
                    for sigma in range(N):
                        diff = current_positions[z, sigma] - current_positions[i, rho]
                        temp += K_delta(diff, delta)
                    drift += (temp / N) * (w0[z] * h**2)

                # Brownian increment
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                new_positions[i, rho] = current_positions[i, rho] + dt * drift + dW

        trajectories[step + 1] = new_positions

    return trajectories


# ----------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------
if __name__ == "__main__":
    # (1) ALGORITHM 1: Get inputs & outputs
    params = define_inputs_and_outputs()

    # (2) ALGORITHM 4: Run the simulation (we already have K_delta from Algo 2)
    #    Note: The velocity field routine (Algo 3) we can call whenever needed.
    trajectories = simulate_vortex_trajectories(params)

    # Now do the same plotting as in your original code, but we can call
    # 'compute_velocity_field' if we want to compute velocity at a set of points.

    # EXAMPLE: Plot the velocity field at times t=0, t=T/4, t=T/2, ...
    time_indices = np.linspace(0, params['num_steps'], 5, dtype=int)

    xx = params['xx']
    yy = params['yy']
    h  = params['h']
    w0 = params['w0']
    N  = params['N']
    delta = params['delta']

    for t_idx in time_indices:
        # shape: (num_particles, N, 2)
        current_positions = trajectories[t_idx]

        # Flatten the grid
        grid_list = np.column_stack((xx.flatten(), yy.flatten()))
        # Compute velocity
        U, V = compute_velocity_field(current_positions, w0, h, delta, N, grid_list)

        # Reshape for plotting
        U_plot = U.reshape(xx.shape)
        V_plot = V.reshape(yy.shape)

        plt.figure(figsize=(6,6))
        plt.quiver(xx, yy, U_plot, V_plot, pivot='mid', color='purple')
        plt.title(f'Velocity Field at t = {t_idx * params["dt"]:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # Plot final velocity field
    current_positions = trajectories[-1]
    grid_list = np.column_stack((xx.flatten(), yy.flatten()))
    U_final, V_final = compute_velocity_field(current_positions, w0, h, delta, N, grid_list)
    U_final = U_final.reshape(xx.shape)
    V_final = V_final.reshape(yy.shape)

    plt.figure(figsize=(6,6))
    plt.quiver(xx, yy, U_final, V_final, pivot='mid', color='green')
    plt.title('Final Velocity Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Plot trajectories
    plt.figure(figsize=(8,8))
    num_steps = params['num_steps']
    num_particles = params['num_particles']
    for i in range(num_particles):
        for rho in range(N):
            plt.plot(trajectories[:, i, rho, 0],
                     trajectories[:, i, rho, 1],
                     color='lightblue', lw=0.5, alpha=0.5)
    # Highlight the active vortex trajectories
    active_indices = params['active_indices']
    colors = ['red', 'blue']
    for idx, vortex in enumerate(active_indices):
        for rho in range(N):
            plt.plot(trajectories[:, vortex, rho, 0],
                     trajectories[:, vortex, rho, 1],
                     color=colors[idx], lw=2,
                     label=f'Vortex {idx+1}' if rho == 0 else "")
    plt.title('Trajectories of All Particles (Active Vortices Highlighted)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()