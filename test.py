import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------
U_inf = 1.0        # free stream velocity
R = 1.0            # cylinder radius (in sigma-plane)
Gamma = 4.0        # circulation (positive = CCW)
b = 1.0            # Joukowski mapping parameter (z = sigma + b^2/sigma)
x_c, y_c = -0.12, 0.08   # cylinder center offset (sigma_c)
sigma_c = x_c + 1j*y_c

# --------------------------
# Cylinder surface (for airfoil contour)
# --------------------------
theta = np.linspace(0, 2*np.pi, 1000)
sigma_surface = sigma_c + R * np.exp(1j*theta)          # points on cylinder
sigma_rel_surface = sigma_surface - sigma_c            # relative coordinate

# --------------------------
# Complex potential & velocity in sigma-plane (cylinder centered at sigma_c)
# Phi = U * ( (sigma - c) + R^2/(sigma - c) ) + i * Gamma/(2pi) * log(sigma - c)
# w_sigma = dPhi/dsigma = U * ( 1 - R^2/(sigma_rel^2) ) + i*Gamma/(2pi) / sigma_rel
# --------------------------
Phi_surface = U_inf * (sigma_rel_surface + (R**2) / sigma_rel_surface) \
              + 1j * (Gamma / (2*np.pi)) * np.log(sigma_rel_surface)
w_sigma_surface = U_inf * (1.0 - (R**2) / (sigma_rel_surface**2)) \
                  + 1j * (Gamma / (2*np.pi)) / (sigma_rel_surface)

# --------------------------
# Joukowski mapping (applied to the absolute sigma)
# z = sigma + b^2 / sigma
# dz/dsigma = 1 - b^2 / sigma^2
# --------------------------
z_surface = sigma_surface + (b**2) / sigma_surface
dz_dsigma_surface = 1.0 - (b**2) / (sigma_surface**2)

# velocity in z-plane on surface
w_z_surface = w_sigma_surface / dz_dsigma_surface

# pressure coefficient on surface
Cp_surface = 1.0 - (np.abs(w_z_surface) / U_inf)**2

# --------------------------
# Build a grid in sigma-plane for streamlines (avoid inside cylinder)
# --------------------------
nx, ny = 300, 300
x_sigma = np.linspace(-3.0, 3.0, nx)
y_sigma = np.linspace(-2.0, 2.0, ny)
X, Y = np.meshgrid(x_sigma, y_sigma)
sigma_grid = X + 1j*Y
sigma_rel_grid = sigma_grid - sigma_c

# mask inside cylinder to avoid singularities
mask_inside = np.abs(sigma_rel_grid) <= R + 1e-8

# compute complex potential (just need streamfunction: psi = imag(Phi))
# careful to avoid log singularity inside cylinder (we'll mask)
Phi_grid = U_inf * (sigma_rel_grid + (R**2) / np.where(mask_inside, np.nan, sigma_rel_grid)) \
           + 1j * (Gamma / (2*np.pi)) * np.log(np.where(mask_inside, np.nan, sigma_rel_grid))
Psi_grid = np.imag(Phi_grid)   # streamfunction in sigma-plane

# map grid points to z-plane
# avoid mapping points too close to sigma=0 (division by zero)
safe_sigma_grid = np.where(np.abs(sigma_grid) == 0, 1e-12 + 0j, sigma_grid)
z_grid = safe_sigma_grid + (b**2) / safe_sigma_grid

# map streamfunction values to z-plane coordinates for contouring
# we'll plot contours by using z_grid.real, z_grid.imag and Psi_grid (masked)
Zx = np.real(z_grid)
Zy = np.imag(z_grid)
Psi_plot = np.ma.array(Psi_grid, mask=mask_inside)  # mask inside cylinder

# --------------------------
# Plotting
# --------------------------
fig = plt.figure(figsize=(14,6))

# sigma-plane: cylinder + some streamlines (psi contours)
ax1 = fig.add_subplot(1,2,1)
ax1.set_title("Cylinder in σ-plane")
# plot cylinder surface
ax1.plot(np.real(sigma_surface), np.imag(sigma_surface), 'b', lw=1.5)
# streamlines (a few levels)
levels = np.linspace(-2.0, 2.0, 40)
cs1 = ax1.contour(X, Y, Psi_plot, levels=levels, linewidths=0.6)
ax1.set_aspect('equal', 'box')
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-1.8, 1.8)
ax1.set_xlabel("xσ")
ax1.set_ylabel("yσ")

# z-plane: mapped airfoil + mapped streamlines
ax2 = fig.add_subplot(1,2,2)
ax2.set_title("Mapped Joukowski Airfoil (z-plane)")
# plot mapped surface
ax2.plot(np.real(z_surface), np.imag(z_surface), 'r', lw=1.5)
# plot mapped streamlines: contour using axes of mapped grid
# Note: contour requires grid in Cartesian coords; use tricontour if irregular,
# but for many sigma-grid points the mapping is smooth; we'll plot contour on Ny x Nx arrays
# To keep simple, use contour with the transformed arrays (works visually)
cs2 = ax2.tricontour(np.ravel(Zx), np.ravel(Zy), np.ravel(Psi_plot), levels=levels, linewidths=0.6)
ax2.set_aspect('equal', 'box')
ax2.set_xlim(-3.0, 3.0)
ax2.set_ylim(-1.8, 1.8)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()

# --------------------------
# Plot Cp on airfoil surface vs chordwise coordinate (x along airfoil)
# --------------------------
# Sort by x for plotting
idx = np.argsort(np.real(z_surface))
x_surf_sorted = np.real(z_surface)[idx]
Cp_sorted = Cp_surface[idx]

plt.figure(figsize=(8,4))
plt.plot(x_surf_sorted, Cp_sorted, '-k')
plt.gca().invert_yaxis()   # Cp typical plot inverted (higher negative up)
plt.title("Pressure coefficient on Joukowski airfoil surface")
plt.xlabel("x (airfoil chordwise)")
plt.ylabel("Cp")
plt.grid(True)
plt.show()
