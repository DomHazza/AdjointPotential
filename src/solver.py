from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem
import src.mesher
import ufl



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

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    xi, eta = x[0], x[1]
    r_sq = xi**2 + eta**2
    
    # Compute h_sq = |dz/d_sigma|^2 where z = (sigma - a) + 1/(sigma - a)
    # and sigma = xi + i*eta, a = a_real + i*a_img
    # Need to define a_real and a_img (assuming they're parameters)
    epsilon = 1e-12
    a_real = fem.Constant(mesh, -0.0)
    a_img = fem.Constant(mesh, 0.0)
    
    # sigma - a = (xi - a_real) + i*(eta - a_img)
    sigma_minus_a_real = xi - a_real
    sigma_minus_a_imag = eta - a_img + epsilon
    sigma_minus_a_mag_sq = sigma_minus_a_real**2 + sigma_minus_a_imag**2
    
    # dz/d_sigma = 1 - 1/(sigma - a)^2
    # First compute 1/(sigma - a)^2
    # 1/(sigma - a) = (sigma_minus_a_real - i*sigma_minus_a_imag) / |sigma - a|^2
    inv_real = sigma_minus_a_real / sigma_minus_a_mag_sq
    inv_imag = -sigma_minus_a_imag / sigma_minus_a_mag_sq
    
    # 1/(sigma - a)^2 = (inv_real + i*inv_imag)^2
    inv_sq_real = inv_real**2 - inv_imag**2
    inv_sq_imag = 2.0 * inv_real * inv_imag
    
    # dz/d_sigma = 1 - 1/(sigma - a)^2
    dz_dsigma_real = 1.0 - inv_sq_real
    dz_dsigma_imag = -inv_sq_imag
    
    # |dz/d_sigma|^2
    h_sq = dz_dsigma_real**2 + dz_dsigma_imag**2
    q_sq = ufl.dot(ufl.grad(phi_h), ufl.grad(phi_h))

    # Compressible density correction: rho = (1 - (gamma-1)/2 * M_inf^2 * h_sq)^(1/(gamma-1))
    rho = (1.0 +0.5*(gamma-1.0) * M_inf**2 * q_sq/(r_sq* h_sq))**(1.0 / (gamma - 1.0))
    rho=1.
    
    # Residual F(phi; v)
    F = rho * ufl.dot(ufl.grad(phi_h), ufl.grad(v)) * ufl.dx


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
    #rho_expr = fem.Expression(1.0, V.element.interpolation_points())
    rho_final = fem.Function(V)
    rho_final.interpolate(rho_expr)

    return velocity, phi_h, rho_final