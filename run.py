import matplotlib.pyplot as plt
import src.visualiser
import numpy as np
import src.solver2
import src.mesher
import ufl






def to_sigma(mesh):
    coords = mesh.geometry.x
    z = coords[:, 0] + 1j * coords[:, 1]
    sigma = 0.5 * (z + np.sqrt(z**2 - 4))
    coords[:, 0] = sigma.real
    coords[:, 1] = sigma.imag


mesh, cell_tags, facet_tags = src.mesher.build_sigma_mesh()
velocity, phi, rho = src.solver2.potential_compressible(mesh, facet_tags, M_inf=0.1, gamma=1.4)


phi_values = phi.x.array
to_sigma(mesh)

coords = mesh.geometry.x
plt.tricontourf(coords[:, 0], coords[:, 1], phi_values)
plt.colorbar(label='phi')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potential Field phi')
plt.axis('equal')
plt.show()


# plt.plot(q_on_circle[0], q_on_circle[1])
# plt.xlabel('Dof index on r=1 circle')
# plt.ylabel('q value')
# plt.title('Velocity Magnitude q on r=1 Circle')
# plt.show()

# src.visualiser.show_flow(mesh, facet_tags, velocity, phi, rho)
# plt.show()