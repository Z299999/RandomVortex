import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x), color='blue')

# Update function that changes the sine wave over time
def update(frame):
    line.set_ydata(np.sin(x + frame / 10))  # Shift the sine wave
    return line,

# Create an animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Display the animation
plt.show()