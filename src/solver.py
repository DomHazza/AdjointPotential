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
    xi, eta = x[0], x[1]
    R_sq = xi**2 + eta**2
    R = ufl.sqrt(R_sq)
    h = ufl.sqrt((1 - (xi**2 - eta**2)/((xi**2 + eta**2)**2))**2 + (2*xi*eta/((xi**2 + eta**2)**2))**2)
    
    # Compute polar derivatives
    phi_x, phi_y = ufl.grad(phi)[0], ufl.grad(phi)[1]
    phi_r = (xi * phi_x + eta * phi_y) / R
    phi_theta = (-eta * phi_x + xi * phi_y) / R

    u_vel = phi_theta / (R*h)
    v_vel = phi_r / (R_sq*h)
    
    rho=1.
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

    # Compute velocity components from potential
    phi_x, phi_y = ufl.grad(phi)[0], ufl.grad(phi)[1]
    phi_r = (xi * phi_x + eta * phi_y) / R
    phi_theta = (-eta * phi_x + xi * phi_y) / R

    u_vel = phi_theta / (R*h)
    v_vel = phi_r / (R_sq*h)

    # Create vector function space for velocity
    V_vec = fem.functionspace(mesh, ("CG", 1, (2,)))
    velocity = fem.Function(V_vec)

    # Project velocity components
    velocity_expr = fem.Expression(
        ufl.as_vector([u_vel, v_vel]),
        V_vec.element.interpolation_points()
    )
    velocity.interpolate(velocity_expr)

    # Compute final density (placeholder for actual compressible calculation)
    rho = (1+0.5*(gamma - 1)*M_inf**2 * (1 - (u_vel**2 + v_vel**2)))**(1/(gamma - 1))
    return velocity, phi, rho