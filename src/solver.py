from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem
import src.mesher
import ufl
import numpy as np
from dolfinx import plot



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








def potential_compressible2(mesh, facet_tags, M_inf=0.1, gamma=1.4, phi_target=None):
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
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    theta = ufl.atan2(x[1], x[0])

    dphi_dr = ufl.grad(phi)[0] * ufl.cos(theta) + ufl.grad(phi)[1] * ufl.sin(theta)
    dphi_dtheta = -ufl.grad(phi)[0] * ufl.sin(theta) + ufl.grad(phi)[1] * ufl.cos(theta)

    u_theta_vel = dphi_dtheta/r
    u_r_vel = dphi_dr/r**2
    q_sq = u_r_vel**2 + u_theta_vel**2

    rho = (1 + 0.5 * (gamma - 1) * M_inf**2 * (1 - q_sq))**(1 / (gamma - 1))
    F = rho * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    J = ufl.derivative(F, phi)
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

    # velocity = ufl.as_vector([
    #     u_r_vel * ufl.cos(theta) - u_theta_vel * ufl.sin(theta),
    #     u_r_vel * ufl.sin(theta) + u_theta_vel * ufl.cos(theta)
    # ])
    velocity = ufl.grad(phi)
    velocity_func.interpolate(fem.Expression(velocity, V_vec.element.interpolation_points()))
    rho_func.interpolate(fem.Expression(rho, V_scalar.element.interpolation_points()))


    # Adjoint problem



    return velocity_func, phi, rho_func




def potential_compressible3(mesh, facet_tags, M_inf=0.1, gamma=1.4):
    # --- Function Spaces ---
    V = fem.functionspace(mesh, ("CG", 2))  # Space for potential phi
    V_scal = fem.functionspace(mesh, ("CG", 1))  # Space for rho, q

    # --- Boundary Conditions ---
    # Outer boundary (r = R_outer, tag 3) -> Dirichlet BC
    phi_D = fem.Function(V)

    # Incompressible potential flow past a cylinder
    def phi_D_expr(x):
        return x[0] * (1.0 + 1.0**2 / (x[0] ** 2 + x[1] ** 2))

    phi_D.interpolate(phi_D_expr)

    phi_inf_expr = fem.Expression(
        ufl.SpatialCoordinate(mesh)[0],
        V.element.interpolation_points()
    )
    phi_inf = fem.Function(V)
    phi_inf.interpolate(phi_inf_expr)
    boundary = fem.locate_dofs_topological(V, 1, facet_tags.find(src.mesher.MARKERS['border']))
    bc_outer = fem.dirichletbc(phi_inf, boundary)

    # Inner boundary (r = R_inner, tag 2) -> Neumann BC (v=0 -> d(phi)/dn = 0)
    # This is a natural BC, so no integral term is added for this boundary.

    # --- Variational Problem ---
    phi = fem.Function(V)
    phi.name = "Potential"
    w = ufl.TestFunction(V)

    # --- Define Non-linear Terms ---
    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    r2 = x[0] ** 2 + x[1] ** 2

    # Gradient of potential
    grad_phi = ufl.grad(phi)

    # 1. h^2 (Joukowsky map h = |1 - 1/sigma^2|)
    # h^2 = 1 - 2(x^2-y^2)/r^4 + 1/r^4
    h2 = 1.0 - (2.0 * (x[0] ** 2 - x[1] ** 2) / (r2**2)) + (1.0 / (r2**2))
    h = ufl.sqrt(h2)

    # 2. Polar derivatives
    # phi_r = d(phi)/dr = (d(phi)/dx * x/r) + (d(phi)/dy * y/r)
    phi_r = (grad_phi[0] * x[0] / r) + (grad_phi[1] * x[1] / r)
    # phi_theta = d(phi)/d(theta) = x * d(phi)/dy - y * d(phi)/dx
    phi_theta = (grad_phi[1] * x[0]) - (grad_phi[0] * x[1])

    # 3. u and v (from eq 2.5)
    u_circ = (r * phi_theta) / h
    v_rad = (r**2 * phi_r) / h

    # 4. q^2 (from eq 2.6)
    q2 = u_circ**2 + v_rad**2

    # 5. rho (from eq 2.2)
    # Add stabilization to prevent sqrt of negative numbers (supersonic pocket)
    rho_arg = 1.0 + 0.5 * (gamma - 1.0) * M_inf**2 * (1.0 - q2)
    rho = (ufl.max_value(rho_arg, 1e-6)) ** (1.0 / (gamma - 1.0))

    # --- Weak Form (Residual) ---
    # This is the weak form of the original Cartesian equation (2.1)
    # F = integral(rho * grad(w) . grad(phi) * dV) = 0
    F = rho * ufl.inner(ufl.grad(w), grad_phi) * ufl.dx

    # --- Initial Guess ---
    # Use the far-field solution as an initial guess
    phi.interpolate(phi_D_expr)

    # --- Solve Non-linear Problem ---
    problem = NonlinearProblem(F, phi, bcs=[bc_outer])
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 25
    solver.report = True
    n, converged = solver.solve(phi)
    print('Nonlinear solver converged')

    # --- Post-Processing ---
    # Project u, v, rho, q2 onto scalar function spaces for output
    rho_out = fem.Function(V_scal)
    rho_out.name = "rho"
    rho_expr = fem.Expression(rho, V_scal.element.interpolation_points())
    rho_out.interpolate(rho_expr)

    q2_out = fem.Function(V_scal)
    q2_out.name = "q_squared"
    q2_expr = fem.Expression(q2, V_scal.element.interpolation_points())
    q2_out.interpolate(q2_expr)

    # Project u, v
    u_out = fem.Function(V_scal)
    u_out.name = "u_circumferential"
    u_expr = fem.Expression(u_circ, V_scal.element.interpolation_points())
    u_out.interpolate(u_expr)

    v_out = fem.Function(V_scal)
    v_out.name = "v_radial"
    v_expr = fem.Expression(v_rad, V_scal.element.interpolation_points())
    v_out.interpolate(v_expr)

    import matplotlib.pyplot as plt

    # Create contour plot of potential phi
    fig, ax = plt.subplots(figsize=(10, 8))
    topology, cell_types, geometry = plot.vtk_mesh(V)
    values = phi.x.array.real
    im = ax.tricontour(geometry[:, 0], geometry[:, 1], values, levels=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Potential φ Contours')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    plt.show()

    return phi, u_out, v_out, rho_out, q2_out
        