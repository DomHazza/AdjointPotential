import matplotlib.pyplot as plt
import src.visualiser
import numpy as np
import src.solver
import src.mesher
import ufl


def conformal_map(x, y):
    sigma = x + 1j * y
    z = sigma + 1 / (sigma)
    return z.real, z.imag

def conformal_map_derivative(x, y):
    """Compute the derivative dz/dσ of the conformal map z = σ + 1/σ"""
    sigma = x + 1j * y
    dz_dsigma = 1 - 1 / (sigma**2)
    return dz_dsigma


mesh, cell_tags, facet_tags = src.mesher.build_sigma_mesh(-0.08, 0.08, 1.08)
velocity, phi, rho = src.solver.potential_compressible(mesh, facet_tags, M_inf=0.1, gamma=1.4)
fig = src.visualiser.show_flow(mesh, facet_tags, velocity, phi, rho)




# Build the new mesh
coords = mesh.geometry.x
z_real, z_imag = conformal_map(coords[:, 0], coords[:, 1])
new_coords = np.column_stack([z_real, z_imag])
mesh.geometry.x[:, :2] = new_coords

fig = src.visualiser.show_flow(mesh, facet_tags, velocity, phi, rho)
plt.show()