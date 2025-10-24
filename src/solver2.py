from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem
import numpy as np
import src.mesher
import ufl


def to_z(mesh):
    coords = mesh.geometry.x
    sigma = coords[:, 0] + 1j * coords[:, 1]
    z = (sigma + 1/sigma)
    coords[:, 0] = z.real
    coords[:, 1] = z.imag


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
    u_x = velocity[0]
    u_y = velocity[1]

    rho = (1 + 0.5 * (gamma - 1) * M_inf**2 * (1 - (u_x**2 + u_y**2)))**(1 / (gamma - 1))
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



    return velocity_func, phi, rho_func

    # # Calculate q on r=1 circle
    # V_scalar_q = fem.functionspace(mesh, ("CG", 1))
    # q_func = fem.Function(V_scalar_q)
    # q = ufl.sqrt(u_x**2 + u_y**2)
    # q_func.interpolate(fem.Expression(q, V_scalar_q.element.interpolation_points()))

    # # Find DOFs on r=1 circle
    # x_dofs = V_scalar_q.tabulate_dof_coordinates()
    # r_dofs = np.sqrt(x_dofs[:, 0]**2 + x_dofs[:, 1]**2)
    # circle_dofs = np.where(np.abs(r_dofs - 1.0) < 1e-10)[0]

    # # Calculate theta for points on the circle
    # theta_on_circle = np.arctan2(x_dofs[circle_dofs, 1], x_dofs[circle_dofs, 0])
    # q_on_circle = q_func.x.array[circle_dofs]
    
    # # Sort by theta for ordered output
    # sorted_indices = np.argsort(theta_on_circle)
    # theta_sorted = theta_on_circle[sorted_indices]
    # q_sorted = q_on_circle[sorted_indices]
    
    # q_on_circle = (theta_sorted, q_sorted)


    # # Adjoint problem
    # w = ufl.TestFunction(V)
    # psi = ufl.TrialFunction(V)

    # pressure = rho**gamma/(gamma*M_inf**2)
    # c_sq = gamma * pressure / rho
    
    # dpsi_dr = psi.dx(0)
    # dpsi_dtheta = psi.dx(1)
    # dw_dr = w.dx(0)
    # dw_dtheta = w.dx(1)

    # A = 1.0 - (u_theta_vel**2) / c_sq
    # B = (rho * u_theta_vel * u_r_vel) / c_sq
    # C = 1.0 - (u_r_vel**2) / c_sq
    # F1 = rho * (A * dpsi_dtheta - B * r * dpsi_dr)
    # F2 = rho * (C * r * dpsi_dr - B * dpsi_dtheta)

    # # 7. Define the bilinear form a(psi, w)
    # # a = ( (dw/dtheta)*F1 + (r*dw/dr)*F2 + w*F2 ) * dr * dtheta
    # a = (dw_dtheta * F1 + r * dw_dr * F2 + w * F2) * ufl.dx
    # L = fem.Constant(mesh, 0.0) * w * ufl.dx


    # return u_r_velocity_func, phi, rho_func, q_on_circle
