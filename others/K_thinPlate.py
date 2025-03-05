import numpy as np

def K_delta(x, y, delta=1e-10):
    """
    Computes the mollified Biot–Savart kernel K₍δ₎(x,y) for the thin plate problem.
    
    The function is self-contained and defines inner functions to compute:
      - The conformal map T,
      - Its derivatives T_y1 and T_y2,
      - The auxiliary functions k⁻ and k⁺,
      - And finally, the kernel components K₁ and K₂.
    
    Each inner function applies a mollifier (cutoff) factor in the last step
    to smooth potential singularities.
    
    Parameters:
      x : tuple or array-like of shape (2,)
          The source point in ℝ².
      y : tuple or array-like of shape (2,)
          The evaluation point in ℝ².
      delta : float, optional
          The smoothing parameter (default is 1e-2).
    
    Returns:
      np.array of shape (2,)
          The kernel vector [K₁(x,y), K₂(x,y)].
    """
    
    def mollifier(r, delta):
        # If the distance r is too small, return 0 to avoid singularities.
        if r < 1e-10:
            return 0.0
        return 1 - np.exp(- (r / delta)**2)
    
    def T(point, delta=delta):
        """Conformal map T from the thin plate domain to the upper half-plane."""
        x1, x2 = point
        r = np.sqrt(x1**2 + x2**2)
        if r < 1e-10:
            return np.array([0.0, 0.0])
        T1 = np.sign(x2) * np.sqrt(0.5 * (r + x1))
        T2 = np.sqrt(0.5 * (r - x1))
        factor = mollifier(r, delta)
        return factor * np.array([T1, T2])
    
    def T_y1(point, delta=delta):
        """
        Computes the partial derivative of T with respect to y₁.
        
        Formulas:
          ∂T^1/∂y₁ = sgn(y₂) * sqrt(0.5*(|y| + y₁))/(2|y|)
          ∂T^2/∂y₁ = - sqrt(0.5*(|y| - y₁))/(2|y|)
        """
        x1, x2 = point
        r = np.sqrt(x1**2 + x2**2)
        if r < 1e-10:
            return np.array([0.0, 0.0])
        dT1_dy1 = np.sign(x2) * np.sqrt(0.5*(r + x1)) / (2 * r)
        dT2_dy1 = - np.sqrt(0.5*(r - x1)) / (2 * r)
        factor = mollifier(r, delta)
        return factor * np.array([dT1_dy1, dT2_dy1])
    
    def T_y2(point, delta=delta):
        """
        Computes the partial derivative of T with respect to y₂.
        
        Formulas:
          ∂T^1/∂y₂ = sgn(y₂)* y₂/(4|y|*sqrt(0.5*(|y| + y₁)))
          ∂T^2/∂y₂ = y₂/(4|y|*sqrt(0.5*(|y| - y₁)))
        """
        x1, x2 = point
        r = np.sqrt(x1**2 + x2**2)
        if r < 1e-10:
            return np.array([0.0, 0.0])
        sqrt1 = np.sqrt(0.5*(r + x1))
        sqrt2 = np.sqrt(0.5*(r - x1))
        if sqrt1 < 1e-10 or sqrt2 < 1e-10:
            return np.array([0.0, 0.0])
        dT1_dy2 = np.sign(x2) * x2 / (4 * r * sqrt1)
        dT2_dy2 = x2 / (4 * r * sqrt2)
        factor = mollifier(r, delta)
        return factor * np.array([dT1_dy2, dT2_dy2])
    
    def k_minus(x, y, delta=delta):
        """
        Computes the auxiliary function k⁻(x,y):
          k⁻(x,y) = (1/(2π)) * (T(y) - T(x)) / ((T¹(y)-T¹(x))² + (T²(y)-T²(x))²)
        """
        T_x = T(x, delta)
        T_y_val = T(y, delta)
        diff = T_y_val - T_x
        denom = diff[0]**2 + diff[1]**2
        r_denom = np.sqrt(denom)
        if r_denom < 1e-10:
            return np.array([0.0, 0.0])
        factor = mollifier(r_denom, delta)
        return factor * (1/(2*np.pi)) * diff / denom
    
    def k_plus(x, y, delta=delta):
        """
        Computes the auxiliary function k⁺(x,y):
          k⁺(x,y) = (1/(2π)) * ([T¹(y)-T¹(x), T²(y)+T²(x)]) / ((T¹(y)-T¹(x))² + (T²(y)-T²(x))²)
        """
        T_x = T(x, delta)
        T_y_val = T(y, delta)
        diff1 = T_y_val[0] - T_x[0]
        num2 = T_y_val[1] + T_x[1]
        # Denom is the same as for k_minus.
        diff_dummy = T_y_val[1] - T_x[1]
        denom = diff1**2 + diff_dummy**2
        r_denom = np.sqrt(denom)
        if r_denom < 1e-10:
            return np.array([0.0, 0.0])
        factor = mollifier(r_denom, delta)
        num = np.array([diff1, num2])
        return factor * (1/(2*np.pi)) * num / denom
    
    def K_1(x, y, delta=delta):
        """
        Computes the first component of the kernel:
          K¹(x,y) = k⁻(x,y) · T_y2(y) - k⁺(x,y) · T_y2(y)
        """
        km = k_minus(x, y, delta)
        kp = k_plus(x, y, delta)
        Ty2 = T_y2(y, delta)
        return np.dot(km, Ty2) - np.dot(kp, Ty2)
    
    def K_2(x, y, delta=delta):
        """
        Computes the second component of the kernel:
          K²(x,y) = - k⁻(x,y) · T_y1(y) + k⁺(x,y) · T_y1(y)
        """
        km = k_minus(x, y, delta)
        kp = k_plus(x, y, delta)
        Ty1 = T_y1(y, delta)
        return -np.dot(km, Ty1) + np.dot(kp, Ty1)
    
    return np.array([K_1(x, y, delta), K_2(x, y, delta)])

# ---------------- Example Usage ----------------
if __name__ == '__main__':
    x = (1.0, 0.5)
    y = (2.0, 1.0)
    
    print("K_delta(x,y) =", K_delta(x, y))
