from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem
import src.mesher
import ufl



def potential_compressible(mesh, facet_tags, M_inf=0.1, gamma=1.4):
    V = fem.functionspace(mesh, ("CG", 1))
    phi = fem.Function(V)

    # Set up the linear problem
    u_lin = ufl.TrialFunction(V)
    v_lin = ufl.TestFunction(V)
    a_lin = ufl.dot(ufl.grad(u_lin), ufl.grad(v_lin)) * ufl.dx
    L_lin = fem.Constant(mesh, 0.0) * v_lin * ufl.dx

    # Boundary condition
    phi_inf_expr = fem.Expression(
        ufl.SpatialCoordinate(mesh)[0],
        V.element.interpolation_points()
    )
    phi_inf = fem.Function(V)
    phi_inf.interpolate(phi_inf_expr)
    boundary = fem.locate_dofs_topological(V, 1, facet_tags.find(src.mesher.MARKERS['border']))
    bc_outer = fem.dirichletbc(phi_inf, boundary)

    # Solve for incompressible initial guess
    linear_problem = LinearProblem(a_lin, L_lin, bcs=[bc_outer], u=phi)
    linear_problem.solve()
    print("Initial guess computed.")

    # Set up the nonlinear problem
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)

    velocity = ufl.grad(phi)
    u_vel, v_vel = velocity[0], velocity[1]
    rho = (1 + 0.5 * (gamma - 1) * M_inf**2 * (1 - (u_vel**2 + v_vel**2)))**(1 / (gamma - 1))
    
    # Residual F(phi; v)
    F = rho * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    J = ufl.derivative(F, phi)

    # --- 6. Setup and Solve the Nonlinear Problem ---
    problem = NonlinearProblem(F, phi, bcs=[bc_outer], J=J)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.max_it = 20

    n, converged = solver.solve(phi)
    print('Nonlinear solver converged')

    # Project velocity and density to function spaces for plotting
    V_vec = fem.functionspace(mesh, ("CG", 1, (2,)))  # Vector function space for velocity
    V_scalar = fem.functionspace(mesh, ("CG", 1))     # Scalar function space for density

    velocity_func = fem.Function(V_vec)
    rho_func = fem.Function(V_scalar)

    velocity_func.interpolate(fem.Expression(velocity, V_vec.element.interpolation_points()))
    rho_func.interpolate(fem.Expression(rho, V_scalar.element.interpolation_points()))

    return velocity_func, phi, rho_func