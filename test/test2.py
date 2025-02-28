import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
nu = 0.015          # viscosity
T = 15              # final time (s)
dt = 0.1            # time step
num_steps = int(T/dt)
delta = 0.1         # kernel mollification
epsilon = 0.1       # viscosity smoothing
h_vis = dt          # viscosity time step

# Mesh parameters
h0 = 1.0            # coarse grid spacing
h1 = 0.5            # fine grid spacing in x
h2 = 0.1            # fine grid spacing in y

region_x = [-6, 6]
region_y = [-6, 6]  # y-domain: lower y1, upper y4
y1 = region_y[0]    # -6
y4 = region_y[1]    # 6
y2 = -0.4           # splitting point (boundary layer thickness)

def generate_symmetric_grids():
    """
    1. Create lower grids:
       - Coarse: [x1,x2] x [y1,y2)
       - Fine:   [x1,x2] x [y2,0)
    2. Mirror them to get the upper half.
    3. Add a boundary mesh at y=0.
    4. Combine lower, boundary, and upper parts.
    """
    x1, x2 = region_x

    # Coarse grid: lower half [y1,y2)
    N_x_coarse = int((x2 - x1) / h0) + 1
    N_y_coarse = int((y2 - y1) / h0)
    x_coarse = np.linspace(x1, x2, N_x_coarse)
    y_coarse_lower = np.linspace(y1, y2, N_y_coarse, endpoint=False)
    XX_coarse, YY_coarse_lower = np.meshgrid(x_coarse, y_coarse_lower, indexing='ij')
    D02 = np.stack((XX_coarse, YY_coarse_lower), axis=-1)
    
    # Mirror to get the upper coarse grid
    D01 = D02.copy()
    D01[:,:,1] = -D01[:,:,1]
    D01 = D01[:, ::-1, :]
    
    # Combine coarse grids (lower + upper)
    D0 = np.concatenate((D02, D01), axis=1)
    
    # Fine grid: lower half [y2, 0)
    N_x_fine = int((x2 - x1) / h1) + 1
    N_y_fine_lower = int((0 - y2) / h2)
    x_fine = np.linspace(x1, x2, N_x_fine)
    y_fine_lower = np.linspace(y2, 0, N_y_fine_lower, endpoint=False)
    XX_fine, YY_fine_lower = np.meshgrid(x_fine, y_fine_lower, indexing='ij')
    Db2 = np.stack((XX_fine, YY_fine_lower), axis=-1)
    
    # Mirror to get the upper fine grid
    Db1 = Db2.copy()
    Db1[:,:,1] = -Db1[:,:,1]
    Db1 = Db1[:, ::-1, :]
    
    # Create boundary mesh at y=0
    Dbd = np.stack((x_fine, np.zeros_like(x_fine)), axis=-1)
    Dbd = Dbd[:, None, :]
    
    # Combine fine grids: lower + boundary + upper
    Db = np.concatenate((Db2, Dbd, Db1), axis=1)
    
    return D0, Db

# Generate the grids
D0, Db = generate_symmetric_grids()

# Debug printouts
print("Coarse grid D0 shape:", D0.shape)
print("Fine grid Db shape:", Db.shape)
print("\nCoarse grid sample:")
print(D0[:5, :5, :])
print("\nFine grid sample:")
print(Db[:5, :5, :])

# Plot the grids
plt.figure(figsize=(10, 8))
plt.scatter(D0[:,:,0].flatten(), D0[:,:,1].flatten(), color='red', s=10, label='Coarse Grid (D0)')
plt.scatter(Db[:,:,0].flatten(), Db[:,:,1].flatten(), color='blue', s=10, label='Fine Grid (Db)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Symmetric Combined Grids')
plt.legend()
plt.grid(True)
plt.xlim(region_x[0]-1, region_x[1]+1)
plt.ylim(region_y[0]-1, region_y[1]+1)
plt.show()
