import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem import Function, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from petsc4py import PETSc

# ============================================================================
# Parameters
# ============================================================================
gamma = 1.4  # Ratio of specific heats (air)
M_inf = 0.1  # Freestream Mach number
alpha = 0.0  # Angle of attack (radians)

# Mesh parameters
n_r = 50  # Radial resolution
n_theta = 100  # Azimuthal resolution
r_inner = 1.0  # Inner radius (body)
r_outer = 5.0  # Outer radius (far field)

# ============================================================================
# Create mesh in transformed (r, theta) plane
# ============================================================================
# Create a rectangular mesh in (r, theta) coordinates
# r in [r_inner, r_outer], theta in [0, 2*pi]
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[r_inner, 0.0], [r_outer, 2.0 * np.pi]],
    [n_r, n_theta],
    cell_type=mesh.CellType.quadrilateral
)

# Function space (use higher order for better accuracy)
V = fem.functionspace(domain, ("CG", 1))

# ============================================================================
# Define conformal mapping metric h = |dz/dsigma| = |1 - 1/sigma^2|
# ============================================================================
def compute_metric(r, theta):
    """
    Compute h = |dz/dsigma| where z = sigma + 1/sigma
    and sigma = r * exp(i*theta)
    
    dz/dsigma = 1 - 1/sigma^2 = 1 - exp(-2*i*theta)/r^2
    h = |1 - exp(-2*i*theta)/r^2|
    """
    # Real and imaginary parts of (1 - 1/sigma^2)
    real_part = 1.0 - ufl.cos(2.0 * theta) / (r * r)
    imag_part = ufl.sin(2.0 * theta) / (r * r)
    
    # Magnitude
    h = ufl.sqrt(real_part**2 + imag_part**2)
    return h

# ============================================================================
# Define velocity components and density
# ============================================================================
def compute_velocities(phi, r, theta, h):
    """
    Compute velocity components:
    u = (r/h) * dphi/dtheta
    v = (r^2/h) * dphi/dr
    """
    dphi_dtheta = phi.dx(1)  # Derivative w.r.t. theta (second coordinate)
    dphi_dr = phi.dx(0)      # Derivative w.r.t. r (first coordinate)
    
    u = (r / h) * dphi_dtheta
    v = (r * r / h) * dphi_dr
    
    return u, v

def compute_density(u, v, M_inf, gamma):
    """
    Compute density from isentropic relation:
    rho = [1 + 0.5*(gamma-1)*M_inf^2 * (1 - q^2)]^(1/(gamma-1))
    where q^2 = u^2 + v^2
    """
    q_squared = u**2 + v**2
    
    # Density ratio (ensure positivity)
    density_base = 1.0 + 0.5 * (gamma - 1.0) * M_inf**2 * (1.0 - q_squared)
    
    # Add small epsilon to avoid division by zero or negative values
    eps = 1e-10
    density_base = ufl.conditional(ufl.gt(density_base, eps), density_base, eps)
    
    # Power for isentropic relation
    power = 1.0 / (gamma - 1.0)
    rho = density_base**power
    
    return rho

# ============================================================================
# Setup variational problem
# ============================================================================
# Trial and test functions
phi = Function(V, name="potential")
phi_test = ufl.TestFunction(V)

# Get spatial coordinates
x = ufl.SpatialCoordinate(domain)
r = x[0]
theta = x[1]

# Compute metric
h = compute_metric(r, theta)

# Compute velocities
u, v = compute_velocities(phi, r, theta, h)

# Compute density
rho = compute_density(u, v, M_inf, gamma)

# Weak form: integral(rho * grad(phi) . grad(v) dx) = 0
# In cylindrical coordinates with Jacobian r:
# The measure includes r (Jacobian of polar coordinates)
F = rho * ufl.inner(ufl.grad(phi), ufl.grad(phi_test)) * r * ufl.dx

# ============================================================================
# Boundary conditions
# ============================================================================
# We need to identify boundaries
# For a rectangular domain in (r, theta):
# - r = r_inner: inner boundary (body surface)
# - r = r_outer: outer boundary (far field)
# - theta = 0 and theta = 2*pi: periodic boundaries

def on_inner_boundary(x):
    return np.isclose(x[0], r_inner)

def on_outer_boundary(x):
    return np.isclose(x[0], r_outer)

# Find boundary DOFs
fdim = domain.topology.dim - 1
inner_facets = mesh.locate_entities_boundary(domain, fdim, on_inner_boundary)
outer_facets = mesh.locate_entities_boundary(domain, fdim, on_outer_boundary)

# Inner boundary: tangency condition (Neumann: dphi/dr = 0)
# This is naturally satisfied in weak form, no explicit BC needed

# Outer boundary: uniform flow condition
# phi = U_inf * (r * cos(theta - alpha) + 1/r * cos(theta - alpha))
# For simplicity, we can set phi = r * cos(theta - alpha) at far field

def phi_outer(x):
    """Potential at outer boundary for uniform flow at angle alpha"""
    r_val = x[0]
    theta_val = x[1]
    # Uniform flow: phi = U_inf * r * cos(theta - alpha)
    # Normalized by U_inf = 1
    return r_val * np.cos(theta_val - alpha)

# Create boundary condition
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)
bc_outer = fem.dirichletbc(
    fem.Function(V),
    outer_dofs
)

# Set values for outer BC
phi_bc = fem.Function(V)
phi_bc.interpolate(phi_outer)
bc_outer = fem.dirichletbc(phi_bc, outer_dofs)

bcs = [bc_outer]

# ============================================================================
# Initial guess
# ============================================================================
# Use potential flow solution as initial guess
phi.interpolate(phi_outer)

# ============================================================================
# Setup nonlinear solver
# ============================================================================
problem = NonlinearProblem(F, phi, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# Solver parameters
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.convergence_criterion = "incremental"

# Configure PETSc options for the solver
ksp = solver.krylov_solver
opts = PETSc.Options()
opts["ksp_type"] = "gmres"
opts["pc_type"] = "ilu"
opts["ksp_rtol"] = 1e-8
opts["ksp_max_it"] = 1000
ksp.setFromOptions()

# ============================================================================
# Solve the nonlinear problem
# ============================================================================
print("Solving nonlinear compressible potential flow problem...")
print(f"Mach number: M_inf = {M_inf}")
print(f"Gamma: {gamma}")
print(f"Angle of attack: alpha = {alpha} rad = {np.degrees(alpha)} deg")
print(f"Mesh: {n_r} x {n_theta} elements")

n_iter, converged = solver.solve(phi)

if converged:
    print(f"Converged in {n_iter} iterations")
else:
    print(f"Did not converge after {n_iter} iterations")

# ============================================================================
# Post-processing: Compute derived quantities
# ============================================================================
# Create function space for output variables
V_out = fem.functionspace(mesh, ("DG", 1))  # Discontinuous Galerkin for derivatives

# Compute and project velocity components
u_expr = (r / h) * phi.dx(1)
v_expr = (r * r / h) * phi.dx(0)
q_squared_expr = u_expr**2 + v_expr**2

u_func = fem.Function(V_out, name="u_velocity")
v_func = fem.Function(V_out, name="v_velocity")
q_func = fem.Function(V_out, name="velocity_magnitude")
rho_func = fem.Function(V_out, name="density")

# Project expressions onto function space
u_func.interpolate(fem.Expression(u_expr, V_out.element.interpolation_points()))
v_func.interpolate(fem.Expression(v_expr, V_out.element.interpolation_points()))
q_func.interpolate(fem.Expression(ufl.sqrt(q_squared_expr), V_out.element.interpolation_points()))

# Compute density field
rho_expr = compute_density(u_expr, v_expr, M_inf, gamma)
rho_func.interpolate(fem.Expression(rho_expr, V_out.element.interpolation_points()))

# ============================================================================
# Output results
# ============================================================================
# Save to XDMF format for visualization in ParaView
with io.XDMFFile(MPI.COMM_WORLD, "potential_flow.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(phi)
    xdmf.write_function(u_func)
    xdmf.write_function(v_func)
    xdmf.write_function(q_func)
    xdmf.write_function(rho_func)

print("\nResults saved to 'potential_flow.xdmf'")
print("Visualize with ParaView or similar software")

# Print some diagnostic information
print("\n=== Solution Statistics ===")
print(f"Potential (phi) - min: {phi.x.array.min():.6f}, max: {phi.x.array.max():.6f}")
print(f"Density (rho) - min: {rho_func.x.array.min():.6f}, max: {rho_func.x.array.max():.6f}")
print(f"Velocity magnitude (q) - min: {q_func.x.array.min():.6f}, max: {q_func.x.array.max():.6f}")