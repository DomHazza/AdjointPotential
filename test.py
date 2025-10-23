import numpy as np
import matplotlib.pyplot as plt

# Parameters
U_inf = 1.0        # Free stream velocity
R = 1.0            # Cylinder radius
Gamma = 4.0        # Circulation
b = 1.0            # Joukowski parameter (affects thickness)
x_c, y_c = -0.1, 0.1   # Cylinder offset → camber
sigma_c = x_c + 1j*y_c

# Define grid in sigma-plane
theta = np.linspace(0, 2*np.pi, 400)
sigma_surface = sigma_c + R * np.exp(1j*theta)

# Compute potential in sigma-plane
Phi_sigma = U_inf * (sigma_surface + (R**2)/(sigma_surface - sigma_c)) + \
            1j * Gamma/(2*np.pi) * np.log(sigma_surface - sigma_c)

# Map to z-plane (airfoil plane)
z_surface = sigma_surface + (b**2)/(sigma_surface)

# Extract coordinates
x_airfoil = np.real(z_surface)
y_airfoil = np.imag(z_surface)

# Plot both planes
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(np.real(sigma_surface), np.imag(sigma_surface), 'b')
plt.axis('equal')
plt.title("Cylinder in σ-plane")
plt.xlabel("xσ")
plt.ylabel("yσ")

plt.subplot(1,2,2)
plt.plot(x_airfoil, y_airfoil, 'r')
plt.axis('equal')
plt.title("Mapped Joukowski Airfoil (z-plane)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()
