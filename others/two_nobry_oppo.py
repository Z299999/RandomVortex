import numpy as np

np.random.seed(42)


'''parameters'''
nu = 0.001
T = 10
dt = 0.1
num_steps = int(T/dt)
h = 0.1
delta = 0.1
N = 10

'''mesh'''
x_values = np.linspace(-1,1,51)
y_values = np.linspace(-1,1,51)
xx, yy = np.meshgrid(x_values, y_values)
grid_points = np.column_stack( (xx.flatten(), yy.flatten()) )
num_particles = grid_points.shape[0]


'''locate points'''
def find_closest_index(target, points):
    dists = np.linalg.norm(points - target, axis=1)
    return np.argmin(dists)

index1 = find_closest_index(np.array([-0.5,0]), grid_points)
index2 = find_closest_index(np.array([0.5, 0]), grid_points)

active_indices = [index1, index2]
print('active vortex indices', active_indices)
print('active vortex positions', grid_points[active_indices])


'''put vortices strength'''
w0 = np.zeros(num_particles)
w0[index1] = 1.0 / h**2
w0[index2] = -1.0 / h**2

'''mollification'''
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
    return (1 / (2 * np.pi)) * np.array([-x[1], x[0]]) / (r**2) * factor

'''initialize trajectory'''
trajectories = np.zeros( (num_steps+1, num_particles, N, 2) )
for i in range(num_particles):
    trajectories[0, i, :, :] = grid_points[i]

'''time stepping -- update every particle'''
for step in range(num_steps):
    current_positions = trajectories[step]
    new_positions = np.zeros_like(current_positions)
    for i in range(num_particles): # update each particle
        for n in range(N): # update each sample path
            drift = np.zeros(2)
            for z in active indices





