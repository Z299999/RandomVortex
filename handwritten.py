import os
import numpy as np

nu = 0.001
T = 10
dt = 0.1
num_steps = int(T/dt)
delta = 0.1
N = 5
num_vortices = 2

np.random.seed(42)

vortex_positions = np.zeros((2, num_vortices))
vortex_positions[0, :] = np.array([-0.5, -0.5])
vortex_positions[1, :] = np.array([0.5, -0.5])
w0 = np.array([-1, 1])

def K(x, y, delta=0.01):
    x1, x2, y1, y2 = x[0], x[1], y[0], y[1]
    r2     = (x1-y1)**2 + (x2-y2)**2
    r2_bar = (x1-y1)**2 + (-x2-y2)**2
    k1 = 0.5 / np.pi * ((y2-x2)/r2 - (y2+x2)/r2_bar)
    k2 = 0.5 / np.pi * ((y1-x1)/r2_bar - (y1-x1)/r2)
    factor = 1 - np.exp(-(r2/delta)**2)
    return np.array([k1, k2]) * factor

def indi_D(x):
    if x[1] <= 0:
        return 1
    return 0

def reflec(x):
    return np.array([x[0], -x[1]])

# x = np.array([-0.5, -0.5])
# y = np.array([0.5, -0.5])
# print(K(x,y, 0.1))
# print(K(x,y, 0.0001))

def simulate_vortex_traj():
    dim = (num_steps+1, num_vortices, N, 2)
    traj = np.zeros(dim)
    for i in range(num_vortices):
        for r in range(N):
            traj[0, i, r, :]        = vortex_positions[i]

    # print(traj)

    for step in range(num_steps):
        currentPoz = traj[step]
        newPoz = np.zeros_like(currentPoz)
        for i in range(num_vortices):
            for r in range(N): # update each vortex of each sample
                drift = np.zeros(2)
                for j in range(num_vortices):
                    tmp = np.zeros(2)
                    for sigma in range(N): # average each sample
                        pos_j = currentPoz[j, sigma]
                        contrib1 = indi_D(pos_j        )*K(currentPoz[i,r] - pos_j        )
                        contrib2 = indi_D(reflec(pos_j))*K(currentPoz[i,r] - reflec(pos_j))
                        tmp += contrib1 - contrib2
                    drift += 1/N * tmp * w0[j]
                dW = np.sqrt(2 * nu * dt) * np.random.randn(2)
                newPoz[i, r] = currentPoz[i, r] + drift*dt + dW

simulate_vortex_traj()

