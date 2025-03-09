import numpy as np
import matplotlib.pyplot as plt

def K_belt_component_1(x1, x2, y1, y2, j):
    """
    Contribution from the j-th image pair to the first component of the belt kernel,
    i.e. the 'K1' part of:
       K_belt(x,y) = (K1(x,y), K2(x,y)).
    """
    # (y_2 + 12j)
    dist1_sq = (x1 - y1)**2 + (x2 - (y2 + 12*j))**2
    term1 = (x2 - (y2 + 12*j)) / dist1_sq
    
    # (-y_2 + 12j)
    dist2_sq = (x1 - y1)**2 + (x2 - (-y2 + 12*j))**2
    term2 = (x2 - (-y2 + 12*j)) / dist2_sq
    
    return - (1/(2*np.pi)) * (term1 - term2)


def K_belt_component_2(x1, x2, y1, y2, j):
    """
    Contribution from the j-th image pair to the second component of the belt kernel,
    i.e. the 'K2' part of:
       K_belt(x,y) = (K1(x,y), K2(x,y)).
    """
    dist1_sq = (x1 - y1)**2 + (x2 - (y2 + 12*j))**2
    dist2_sq = (x1 - y1)**2 + (x2 - (-y2 + 12*j))**2
    
    term1 = (x1 - y1) / dist1_sq
    term2 = (x1 - y1) / dist2_sq
    
    return (1/(2*np.pi)) * (term1 - term2)


def K_belt(x1, x2, y1, y2, N=2):
    """
    Returns both components (K1, K2) of the belt kernel at the point x=(x1,x2),
    truncated to sums over j from -N to N.
    """
    K1_total = 0.0
    K2_total = 0.0
    for j in range(-N, N+1):
        # Add the j-th contribution
        K1_total += K_belt_component_1(x1, x2, y1, y2, j)
        K2_total += K_belt_component_2(x1, x2, y1, y2, j)
    return K1_total, K2_total


# ----------------------------------------------------------------------
# MAIN SCRIPT: Example of how to evaluate and plot K(x) on a 2D grid.
# ----------------------------------------------------------------------

# 1) Choose a point y in the belt domain, e.g. the midpoint (0, 3).
y_fixed = (0.0, 3.0)

# 2) Define a rectangular grid of x-points inside (or around) the belt domain.
#    For demonstration, we pick x1 in [-2,2], x2 in [1,5].
num_points = 50
x1_vals = np.linspace(-2, 2, num_points)
x2_vals = np.linspace(1, 5, num_points)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# 3) Truncate the infinite sum from j = -N to j = +N (increase N if needed).
N_trunc = 2

# 4) Evaluate K1 and K2 on the grid.
K1 = np.zeros_like(X1)
K2 = np.zeros_like(X1)

y1, y2 = y_fixed
for i in range(num_points):
    for j in range(num_points):
        xx1 = X1[i, j]
        xx2 = X2[i, j]
        k1_val, k2_val = K_belt(xx1, xx2, y1, y2, N=N_trunc)
        K1[i, j] = k1_val
        K2[i, j] = k2_val

# 5) Make a single 3D plot with both surfaces (K1 and K2).
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

surf1 = ax.plot_surface(X1, X2, K1, label='K1')
surf2 = ax.plot_surface(X1, X2, K2, label='K2')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Kernel value')
ax.set_title('Belt Kernel Components K1 & K2 (Truncated Sum)')

# Show the plot
plt.show()
