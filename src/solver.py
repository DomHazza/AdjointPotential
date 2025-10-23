from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem
import src.mesher
import ufl



def potential(mesh, facet_tags):
    # Create scalar function space for potential
    V = fem.functionspace(mesh, ("CG", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Set boundary condition for potential (uniform flow in x-direction)
    phi_inf_expr = fem.Expression(
        ufl.SpatialCoordinate(mesh)[0],
        V.element.interpolation_points()
    )
    phi_inf = fem.Function(V)
    phi_inf.interpolate(phi_inf_expr)

    boundary = fem.locate_dofs_topological(V, 1, facet_tags.find(src.mesher.MARKERS['border']))
    bc_outer = fem.dirichletbc(phi_inf, boundary)

    # Define variational problem (Laplace equation)
    a = ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = fem.Constant(mesh, 0.0) * v * ufl.dx

    # Solve for potential
    problem = LinearProblem(
        a, L, bcs=[bc_outer],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre"}
    )
    phi_sol = problem.solve()

    # Create vector function space for velocity
    V_vec = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))

    # Compute velocity as gradient of potential: u = grad(phi)
    velocity_expr = fem.Expression(ufl.grad(phi_sol), V_vec.element.interpolation_points())
    velocity = fem.Function(V_vec)
    velocity.interpolate(velocity_expr)

    return velocity, phi_sol