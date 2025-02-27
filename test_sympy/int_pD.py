import numpy as np
import scipy.integrate as spi
import scipy.stats as stats

# Define parameters
x = 0  # Starting position
s = 0  # Start time
t = 10  # End time
nu = 1  # Diffusion coefficient

# Define the 1D Brownian motion transition density function
def p_density(y, x, s, t, nu):
    return stats.norm.pdf(y, loc=x, scale=np.sqrt(2 * nu * (t - s)))

# Define the killed transition density function p^D
def p_D(y, x, s, t, nu):
    return p_density(y, x, s, t, nu) - p_density(-y, x, s, t, nu)

# Perform numerical integration over D (y from 0 to infinity)
integral_result, error = spi.quad(p_D, 0, np.inf, args=(x, s, t, nu))

# Compare with theoretical survival probability erf(x / sqrt(4nu(t-s)))
theoretical_value = stats.norm.cdf(x / np.sqrt(2 * nu * (t - s))) * 2 - 1

# Print results
print("Numerical Integral of p^D over D:", integral_result)
print("Expected erf value:", theoretical_value)
