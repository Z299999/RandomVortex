import numpy as np
import matplotlib.pyplot as plt

# --- Baby Case Parameters ---
region_x = [0, 2]         # x in [0, 2] → 3 points if spacing = 1.0
region_y = [-4, 4]        # overall vertical extent (not all used)
h0 = 1.0                  # coarse grid spacing
h1 = 1.0                  # fine grid spacing in x (baby case)
h2 = 1.0                  # fine grid spacing in y (baby case)

# For the coarse grid lower half, we choose:
y1 = region_y[0]          # -4
y2 = -2                   # splitting point for coarse lower half
# Then the coarse lower grid spans y in [y1, y2) = [-4, -2)
# and will have N_y_coarse = int((y2-y1)/h0) = int(2) = 2 rows.
# The fine grid lower half covers [y2, 0) = [-2, 0) with 2 rows (since 0 - (-2) = 2 and h2=1).

# --- Generate Split Meshes ---
def generate_split_meshes():
    """
    Generates five separate meshes:
      - D02: Coarse lower grid over [x1,x2] x [y1, y2) with spacing h0.
      - D01: Coarse upper grid, mirror of D02 (with vertical reversal for matching indices).
      - D0:  Coarse boundary grid along y = 0.
      - Db2: Fine lower grid over [x1,x2] x [y2, 0) with spacing h1 (in x) and h2 (in y).
      - Db1: Fine upper grid, mirror of Db2 (with vertical reversal for matching indices).
    """
    x1, x2 = region_x

    ## Coarse Meshes
    N_x_coarse = int((x2 - x1) / h0) + 1  # e.g., (2-0)/1 +1 = 3
    N_y_coarse = int((y2 - y1) / h0)       # ( -2 - (-4) )/1 = 2
    x_coarse = np.linspace(x1, x2, N_x_coarse)
    y_coarse_lower = np.linspace(y1, y2, N_y_coarse, endpoint=False)
    XX_coarse, YY_coarse_lower = np.meshgrid(x_coarse, y_coarse_lower, indexing='ij')
    D02 = np.stack((XX_coarse, YY_coarse_lower), axis=-1)  # Shape: (3,2,2)

    # Mirror to get coarse upper grid (D01):
    D01 = D02.copy()
    D01[:, :, 1] = -D01[:, :, 1]   # reflect y-values
    D01 = D01[:, ::-1, :]           # reverse vertical order so (i,j) pair with D02

    # Coarse boundary grid along y = 0 (D0)
    D0 = np.stack((x_coarse, np.zeros_like(x_coarse)), axis=-1)  # Shape: (3,2) but one row needed
    D0 = D0[:, None, :]  # Shape: (3,1,2)

    ## Fine Meshes
    N_x_fine = int((x2 - x1) / h1) + 1    # (2-0)/1 +1 = 3
    N_y_fine = int((0 - y2) / h2)           # (0 - (-2))/1 = 2
    x_fine = np.linspace(x1, x2, N_x_fine)
    y_fine_lower = np.linspace(y2, 0, N_y_fine, endpoint=False)
    XX_fine, YY_fine_lower = np.meshgrid(x_fine, y_fine_lower, indexing='ij')
    Db2 = np.stack((XX_fine, YY_fine_lower), axis=-1)  # Shape: (3,2,2)

    # Mirror to get fine upper grid (Db1):
    Db1 = Db2.copy()
    Db1[:, :, 1] = -Db1[:, :, 1]
    Db1 = Db1[:, ::-1, :]

    return D01, D02, D0, Db1, Db2

# Generate the five meshes
D01, D02, D0, Db1, Db2 = generate_split_meshes()

# --- Print the Meshes ---
print("D01 (Coarse Upper Grid):")
print(D01)
print("\nD02 (Coarse Lower Grid):")
print(D02)
print("\nD0 (Coarse Boundary Grid):")
print(D0)
print("\nDb1 (Fine Upper Grid):")
print(Db1)
print("\nDb2 (Fine Lower Grid):")
print(Db2)

# --- Plot the Meshes on One Plot ---
plt.figure(figsize=(8, 6))

# Flatten each mesh to get a list of points (each with x,y coordinates)
pts_D01 = D01.reshape(-1, 2)
pts_D02 = D02.reshape(-1, 2)
pts_D0  = D0.reshape(-1, 2)
pts_Db1 = Db1.reshape(-1, 2)
pts_Db2 = Db2.reshape(-1, 2)

plt.scatter(pts_D01[:,0], pts_D01[:,1], marker='o', color='red', label='D01 (Coarse Upper)')
plt.scatter(pts_D02[:,0], pts_D02[:,1], marker='s', color='blue', label='D02 (Coarse Lower)')
plt.scatter(pts_D0[:,0],  pts_D0[:,1],  marker='^', color='green', label='D0 (Coarse Boundary)')
plt.scatter(pts_Db1[:,0], pts_Db1[:,1], marker='D', color='magenta', label='Db1 (Fine Upper)')
plt.scatter(pts_Db2[:,0], pts_Db2[:,1], marker='v', color='orange', label='Db2 (Fine Lower)')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Baby Case: Split Mesh Grids")
plt.legend()
plt.grid(True)
plt.show()
