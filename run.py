import matplotlib.pyplot as plt
import src.visualiser
import src.solver
import src.mesher

mesh, cell_tags, facet_tags = src.mesher.build_sigma_mesh()
#fig = src.visualiser.show_mesh(mesh, facet_tags)
velocity, phi_sol, rho = src.solver.potential_compressible(mesh, facet_tags, M_inf=0.7, gamma=1.4)
# fig = src.visualiser.show_flow(mesh, velocity, phi_sol)
fig = src.visualiser.show_flow(mesh, velocity, rho)
plt.show()