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


mesh, cell_tags, facet_tags = src.mesher.build_sigma_mesh(-0.08, 0.0, 1.08)

coords = mesh.geometry.x
z_real, z_imag = conformal_map(coords[:, 0], coords[:, 1])
new_coords = np.column_stack([z_real, z_imag])
mesh.geometry.x[:, :2] = new_coords

velocity, phi, rho = src.solver.potential_compressible(mesh, facet_tags, M_inf=0.1, gamma=1.4)
fig = src.visualiser.show_flow(mesh, facet_tags, velocity, phi, rho)
plt.show()