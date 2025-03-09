import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# Create a grid in the parameter domain
u = np.linspace(-2, 2, 50)
v = np.linspace(0, 6, 50)
U, V = np.meshgrid(u, v)

# Map u to the angle theta
theta = (np.pi/2) * (U + 2)  # when u=-2 -> theta=0, when u=2 -> theta=2pi

# Parameterize the cylinder
X = np.cos(theta)
Y = np.sin(theta)
Z = V

# Create the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cylinder obtained by warping the grid [-2,2]x[0,6]')
plt.show()
