import numpy as np

nu = 0.015          # Viscosity
T = 15              # Final time (seconds)
dt = 0.1            # Time step
num_steps = int(T / dt)
delta = 0.1         # Mollification parameter (choose δ < 0.15 for the boundary layer)

# Mesh parameters:
# h0: partition size in the coarse mesh (uniform in x and y)
# h1: partition size along the x-axis for the fine mesh
# h2: partition size along the y-axis for the fine mesh
h0 = 1.0    # Coarse mesh grid spacing (both x and y)
h1 = 0.5    # Fine mesh grid spacing in the x direction
h2 = 0.1    # Fine mesh grid spacing in the y direction

Re = 0.0001 / nu
layer_thickness = 1 * np.sqrt(Re)  # used as the dividing y-level between fine and coarse grids

region_x = [-6, 6]
region_y = [-6, 6]
window_x = [region_x[0], region_x[1]]
window_y = [region_y[0], region_y[1]]

def generate_nonuniform_grid_D():
    """
    Generates a nonuniform grid in D with a coarse grid covering most of the domain
    and a finer grid near the boundary (y=0). The coarse grid uses a uniform partition
    of size h0 in both x and y directions, while the fine grid uses h1 for the x-axis and
    h2 for the y-axis.
    
    The fine grid covers y in [y1, y3] and the coarse grid covers y in [y3, y2],
    where y3 is set to layer_thickness.
    
    Returns:
      A dictionary with keys 'coarse' and 'fine'. Each value is a tuple (grid, A)
      where grid is a numpy array of shape (n_x, n_y, 2) containing the grid points and
      A is the corresponding area array.
    """
    x1, x2 = region_x
    y1, y2 = region_y
    y3 = layer_thickness  # dividing line between fine and coarse grid in y direction

    # Coarse grid: y in [y3, y2] with spacing h0 in both x and y
    num_x_coarse = int((x2 - x1) / h0) + 1
    num_y_coarse = int((y2 - y3) / h0) + 1
    x_coarse = np.linspace(x1, x2, num_x_coarse)
    y_coarse = np.linspace(y3, y2, num_y_coarse)
    xx_coarse, yy_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    grid_coarse = np.stack((xx_coarse, yy_coarse), axis=-1)
    A_coarse = h0 * h0 * np.ones((num_x_coarse, num_y_coarse))
    
    # Fine grid: y in [y1, y3] with spacing h1 (x) and h2 (y)
    num_x_fine = int((x2 - x1) / h1) + 1
    num_y_fine = int((y3 - y1) / h2) + 1
    x_fine = np.linspace(x1, x2, num_x_fine)
    y_fine = np.linspace(y1, y3, num_y_fine, endpoint=False)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    grid_fine = np.stack((xx_fine, yy_fine), axis=-1)
    A_fine = h1 * h2 * np.ones((num_x_fine, num_y_fine))
    
    print(f"Coarse grid shape: {grid_coarse.shape}, number of points: {grid_coarse.shape[0]*grid_coarse.shape[1]}")
    print(f"Fine grid shape: {grid_fine.shape}, number of points: {grid_fine.shape[0]*grid_fine.shape[1]}")
    
    return {'coarse': (grid_coarse, A_coarse), 'fine': (grid_fine, A_fine)}

def generate_nonuniform_grid_D_reflected():
    '''
    D01 and D02 are coarser mesh.
    Db1 and Db2 are the finer mesh.
    D01 and Db1 are in lower half plane.
    D02 and Db2 are in upper half plane.
    '''
    x1, x2 = region_x
    y1, y4 = region_y
    N0 = int(x2 / h0) + 1
    N2 = int(y4 / h0) + 1
    N1 = int(y4 / h1) + 1
    y3 = N2 * h2
    y2 = - y3
    N3 = int((y2 - y1) / h0) + 1

    D01x = np.linspace(x1, x2, 2*N0)
    D01y = np.linspace(y1, y2, N3, endpoint=False)
    xx_D01, yy_D01 = np.meshgrid(D01x, D01y, indexing='ij')
    grid_D01 = np.stack((xx_D01, yy_D01), axis=-1)

    Db1x  = np.linspace(x1, x2, 2*N1)
    Db1y  = np.linspace(y2, 0, N2, endpoint=False)
    xx_Db1, yy_Db1   = np.meshgrid(Db1x, Db1y, indexing='ij')
    grid_Db1 = np.stack((xx_Db1, yy_Db1), axis=-1)


    Db2x  = np.linspace(x1, x2, 2*N1)
    Db2y  = np.linspace(0, y3, N2, endpoint=False)
    xx_Db2, yy_Db2   = np.meshgrid(Db2x, Db2y, indexing='ij')
    grid_Db2 = np.stack((xx_Db2, yy_Db2), axis=-1)

    D02x = np.linspace(x1, x2, 2*N0)
    D02y = np.linspace(y3, y4, N3)
    xx_D02, yy_D02 = np.meshgrid(D02x, D02y, indexing='ij')
    grid_D02 = np.stack((xx_D02, yy_D02), axis=-1)

    A_coarse = h0 * h0
    A_fine   = h1 * h2

    return grid_D01, grid_D02, grid_Db1, grid_Db2, A_coarse, A_fine

grid_D01, grid_D02, grid_Db1, grid_Db2, A_coarse, A_fine = generate_nonuniform_grid_D_reflected()