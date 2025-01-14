from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# Define the corner points of the box
point_1 = Point(0.0, 0.0, 0.0)  # Bottom-left-front corner
point_2 = Point(0.25, 0.5, 1.0)  # Top-right-back corner

# Define the number of divisions along each axis
nx, ny, nz = 5, 10, 20  # Number of divisions along x, y, and z

# Create the box mesh
mesh = BoxMesh(point_1, point_2, nx, ny, nz)

# Define function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 1)  # Vector-valued function space
V2 = TensorFunctionSpace(mesh, "Lagrange", degree=1, shape=(3, 3))  # Tensor-valued function space

# Define boundary subdomains
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Front(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Back(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)

# Initialize sub-domain instances
left = Left()
right = Right()
front = Front()
back = Back()
bottom = Bottom()
top = Top()

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)  # Set all boundaries to 0 (unmarked)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)
front.mark(boundaries, 5)
back.mark(boundaries, 6)

# Define measure for subdomain integration
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Define Dirichlet boundary conditions
c = Expression(("0.0", "0.0", "0.0"), degree=1)  # Zero displacement
bcbo = DirichletBC(V, c, bottom)  # Fix bottom boundary

# Collect boundary conditions
bcs = [bcbo]

# Define functions
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function
u = Function(V)  # Displacement from the previous iteration
B = Constant((0.0, 0.0, 0.0))  # Body force per unit volume

# Define tractions
T = Constant((0.0, 0.0, 0.0))  # Placeholder for traction forces
Tc = Function(V)  # Variable traction force

# Define kinematics
d = u.geometric_dimension()
I = Identity(d)  # Identity tensor
F = I + grad(u)  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Define invariants of the deformation tensor
Ic = tr(C)  # First invariant
J = det(F)  # Determinant of deformation gradient

# Elasticity parameters
E, nu = 10.0, 0.40  # Young's modulus and Poisson's ratio
mu = Constant(E / (2 * (1 + nu)))  # Shear modulus
lmbda = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))  # Lamé's first parameter

# Define stored strain energy density (compressible Neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J))**2

# Define total potential energy
Pi = psi * dx - dot(B, u) * dx - dot(Tc, u) * ds(2)  # Includes body forces and tractions

# Define constraint to enforce F = I on the top surface
scaling_factor = 100.0  # Scaling factor for the constraint
constraint = inner(F - I, F - I) * ds(2)  # Constraint applied on the top surface
Pi += scaling_factor * constraint  # Add constraint to the potential energy

# Define variational problem
F = derivative(Pi, u, v)  # First variation of the potential energy
J = derivative(F, u, du)  # Jacobian of the first variation

# Simulation parameters
chunks = 10  # Number of simulation steps
Tmax = 10.0  # Maximum traction force

# Create a PVD file to store the displacements
file = File("displacement_T_driven.pvd")

# Time-stepping loop
for i in range(0, chunks + 1):
    print(f'chunk number = {i}')

    # Assign current traction value
    Tc.assign(Constant((0.0, 0.0, (i / chunks) * Tmax)))

    # Solve the variational problem
    solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)

    # Save the solution with the current time as a parameter
    u.rename("Displacement", "label")  # Rename the solution
    file << (u, i / chunks)  # Append to the file with a "time" parameter
