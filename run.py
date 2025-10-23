import matplotlib.pyplot as plt
import src.visualiser
import src.solver
import src.mesher

mesh, cell_tags, facet_tags = src.mesher.build_sigma_mesh()
#fig = src.visualiser.show_mesh(mesh, facet_tags)
velocity, phi_sol = src.solver.potential(mesh, facet_tags)
fig = src.visualiser.show_flow(mesh, velocity, phi_sol)
plt.show()