import matplotlib.pyplot as plt
import jax.numpy as jnp
import src.visualiser
import numpy as np
import src.solver
import src.mesher
import ufl


def conformal_map(x, y):
    # sigma - a = (x - a_real) + i(y - a_imag)
    a_real=0.0
    a_imag=0.0
    sigma_minus_a_real = x - a_real
    sigma_minus_a_imag = y - a_imag
    
    # 1/(sigma - a) = (sigma* - a*) / |sigma - a|^2
    denom = sigma_minus_a_real**2 + sigma_minus_a_imag**2
    inv_real = sigma_minus_a_real / denom
    inv_imag = -sigma_minus_a_imag / denom
    
    # z = (sigma - a) + 1/(sigma - a)
    z_real = sigma_minus_a_real + inv_real
    z_imag = sigma_minus_a_imag + inv_imag
    
    return z_real, z_imag





mesh, cell_tags, facet_tags = src.mesher.build_sigma_mesh()
velocity, phi_sol, rho = src.solver.potential_compressible(mesh, facet_tags, M_inf=0.7, gamma=1.4)

# Build the new mesh
coords = mesh.geometry.x
z_real, z_imag = conformal_map(coords[:, 0], coords[:, 1])
new_coords = np.column_stack([z_real, z_imag])
mesh.geometry.x[:, :2] = new_coords

# fig = src.visualiser.show_mesh(mesh, facet_tags)
fig = src.visualiser.show_flow(mesh, velocity, phi_sol)
fig = src.visualiser.show_flow(mesh, velocity, rho)
plt.show()