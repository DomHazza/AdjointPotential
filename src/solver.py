from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
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





def potential_compressible(mesh, facet_tags, M_inf=0.1, gamma=1.4):
    V = fem.functionspace(mesh, ("CG", 1))
    phi_h = fem.Function(V)
    u_lin = ufl.TrialFunction(V)
    v_lin = ufl.TestFunction(V)
    a_lin = ufl.dot(ufl.grad(u_lin), ufl.grad(v_lin)) * ufl.dx
    L_lin = fem.Constant(mesh, 0.0) * v_lin * ufl.dx

    # Set boundary condition for potential (uniform flow in x-direction)
    phi_inf_expr = fem.Expression(
        ufl.SpatialCoordinate(mesh)[0],
        V.element.interpolation_points()
    )
    phi_inf = fem.Function(V)
    phi_inf.interpolate(phi_inf_expr)
    boundary = fem.locate_dofs_topological(V, 1, facet_tags.find(src.mesher.MARKERS['border']))
    bc_outer = fem.dirichletbc(phi_inf, boundary)

    print("Solving for initial guess (incompressible flow)...")
    linear_problem = LinearProblem(a_lin, L_lin, bcs=[bc_outer], u=phi_h)
    linear_problem.solve()
    print("Initial guess computed.")

    # --- 5. Define Nonlinear Variational Problem ---
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    xi, eta = x[0], x[1]
    r_sq = xi**2 + eta**2
    r_four = r_sq**2

    # # Add a small epsilon to h_sq to prevent division by zero if r_inner=1.0
    # # and we hit the trailing edge, though r_inner=1.05 avoids this.
    epsilon = 1e-12
    
    # Compute h_sq = |1 - 1/(x + i*epsilon)|^2
    # where x = xi + i*eta is the complex coordinate
    # h_sq = |1 - 1/(xi + i*eta + i*epsilon)|^2 = |1 - 1/(xi + i*(eta + epsilon))|^2
    denominator_real = xi
    denominator_imag = eta + epsilon
    denominator_mag_sq = denominator_real**2 + denominator_imag**2

    # 1/(xi + i*(eta + epsilon)) = (xi - i*(eta + epsilon)) / |xi + i*(eta + epsilon)|^2
    inverse_real = denominator_real / denominator_mag_sq
    inverse_imag = -denominator_imag / denominator_mag_sq

    # 1 - 1/(xi + i*(eta + epsilon))
    diff_real = 1.0 - inverse_real
    diff_imag = -inverse_imag

    # |1 - 1/(xi + i*(eta + epsilon))|^2
    h_sq = diff_real**2 + diff_imag**2

    # Compressible density correction: rho = (1 - (gamma-1)/2 * M_inf^2 * h_sq)^(1/(gamma-1))
    rho = (1.0 - (gamma - 1.0) / 2.0 * M_inf**2 * h_sq)**(1.0 / (gamma - 1.0))
    # Residual F(phi; v)
    F = rho * ufl.dot(ufl.grad(phi_h), ufl.grad(v)) * ufl.dx

    # Jacobian J = dF/d(phi)
    J = ufl.derivative(F, phi_h)

    # --- 6. Setup and Solve the Nonlinear Problem ---
    problem = NonlinearProblem(F, phi_h, bcs=[bc_outer], J=J)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.max_it = 20

    n, converged = solver.solve(phi_h)
    
    # Compute velocity as gradient of potential: u = grad(phi)
    V_vec = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))
    velocity_expr = fem.Expression(ufl.grad(phi_h), V_vec.element.interpolation_points())
    velocity = fem.Function(V_vec)
    velocity.interpolate(velocity_expr)

    rho_expr = fem.Expression(
        (1.0 - (gamma - 1.0) / 2.0 * M_inf**2 * h_sq)**(1.0 / (gamma - 1.0)),
        V.element.interpolation_points()
    )
    rho_final = fem.Function(V)
    rho_final.interpolate(rho_expr)

    return velocity, phi_h, rho_final