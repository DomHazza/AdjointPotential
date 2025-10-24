from dolfinx import mesh, fem
import dolfinx.fem.petsc
import dolfin_adjoint as da
from mpi4py import MPI
from petsc4py import PETSc

# --- 1. Mesh ---
domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0, 0), (1, 1)], [50, 50])
V = fem.FunctionSpace(domain, ("CG", 1))

# --- 2. Boundary conditions ---
phi = fem.Function(V)
v = fem.TestFunction(V)
phi_in = fem.Constant(domain, 1.0)
bc = fem.dirichletbc(phi_in, fem.locate_dofs_geometrical(V, lambda x: x[0] < 1e-8))

# --- 3. Weak form for Laplace's equation ---
a = fem.inner(fem.grad(phi), fem.grad(v)) * fem.dx
L = fem.Constant(domain, 0.0) * v * fem.dx
fem.petsc.LinearProblem(a, L, bcs=[bc]).solve(phi)

# --- 4. Objective: minimize phi variance at outlet ---
J = da.assemble((phi - 1.0)**2 * da.dx)

# --- 5. Compute gradient wrt shape or boundary param ---
da.adj_start_tape()
da.set_working_tape(da.get_working_tape())
dJdparam = da.compute_gradient(J, [some_parameter])