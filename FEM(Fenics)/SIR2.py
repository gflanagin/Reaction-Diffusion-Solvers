############################
# imports
############################
from mpi4py import MPI
import numpy as np
import basix.ufl
import pyvista

from dolfinx import mesh, fem, plot, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.fem import Function, functionspace, FunctionSpace, dirichletbc, locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from dolfinx.io.gmsh import read_from_msh

from petsc4py import PETSc

import ufl
from ufl import (
	TrialFunction, TestFunction,
	split, grad, div, dot, cross, sqrt, exp,
	dx, min_value, CellNormal, as_vector, Identity, outer,
)

from LC_to_capacity import land_cover_to_carrying_capacity
from dolfinx.fem import Function, functionspace

############################
# model and time parameters
############################
T=5 #years for simulation
dt=.01 
isotropy = .02  # value between 0 and 1, needs to lineup with parameter in mesh generation
DS=100
DI=100
DR=100
kappa=200
alpha=365/5
sigma=365/28
beta=80
K=2
r=1.5
############################
# domain, function space, and functions
############################
mesh_data = read_from_msh("terrain4.msh", MPI.COMM_WORLD, gdim=3)
domain = mesh_data.mesh


# single linear element
P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)

# triple element (S, I, R)
mixed = basix.ufl.mixed_element([P1, P1, P1])

# triple function space
P = fem.functionspace(domain, mixed)
Psingle = fem.functionspace(domain, P1) #single function

#trial functions and test functions
U = TrialFunction(P)
uS,uI,uR = split(U) #symbolic splitting for ufl
V = TestFunction(P)
vS,vI,vR = split(V)

#intial condition
U0 = fem.Function(P)
uS0, uI0, uR0 = split(U0)  # symbolic, stays on mixed space

# Load land cover
V_scal = fem.functionspace(domain, ("CG", 1))
V_lc = fem.functionspace(domain, ("Lagrange", 1))

land_cover_classes = np.load("land_cover_classes.npy").astype(np.int32)
land_cover_func = fem.Function(V_lc)
land_cover_func.name = "Land_Cover_Class"
land_cover_func.x.array[:] = land_cover_classes

lcD_values = np.load("land_cover_diffusivity.npy")
lcD_scale = Function(V_scal)
lcD_scale.x.array[:] = lcD_values
lcD_scale.x.scatter_forward()

K_values = land_cover_to_carrying_capacity(land_cover_classes, K_base=K)
K_func = Function(V_scal)
K_func.x.array[:] = K_values
K_func.x.scatter_forward()

with io.XDMFFile(domain.comm, "land_cover_classes.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(land_cover_func)

V_S, dof_map = P.sub(0).collapse()
K_S = Function(V_S)
K_S.interpolate(fem.Expression(K_func, V_S.element.interpolation_points))
U0.sub(0).x.array[dof_map] = K_S.x.array
U0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0)) # I
U0.sub(2).interpolate(lambda x: .01*np.exp(-((x[0]-162)**2+(x[1]-139)**2)/500000)) #R

"""x0, y0 = -162.71, 139.94
radius = 100.0
U0.sub(2).interpolate(lambda x: np.where(             # R
    (x[0] - x0)**2 + (x[1] - y0)**2 < radius**2,
    1.0, 0.0
))"""

U0.x.scatter_forward()

#solution function
U1 = fem.Function(P)

################
# plotter
################
xdmf = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/SIR2_stretched_mesh2.xdmf", "w")
xdmf.write_mesh(domain)


###################
#Construct tensor
###################
n=CellNormal(domain)
g= as_vector((0,0,-1))
cos_theta=dot(g,n)

es_raw=g-dot(g,n)*n
es_norm = ufl.sqrt(dot(es_raw,es_raw))

xdir = as_vector((1,0,0))
x_proj_raw = xdir - dot(xdir,n)*n
x_proj = x_proj_raw / ufl.sqrt(dot(x_proj_raw,x_proj_raw))

es = ufl.conditional(
    ufl.gt(es_norm, 1E-6),
    es_raw / es_norm,
    x_proj
)

activation = (1+exp(50*(.96-1)))/(1+exp(50*(.96-abs(cos_theta))))

# directional diffusivity
Ds=kappa*(isotropy + (1-isotropy)*activation)
Dc=kappa

# tensor
DTens = lcD_scale * ((Ds-Dc)*outer(es,es) + Dc*(Identity(3)-outer(n,n)))

a = (
    uS*vS + dt*DS*dot(DTens*grad(uS), grad(vS))
    + uI*vI + dt*DI*dot(DTens*grad(uI),grad(vI))
    + uR*vR + dt*DR*dot(DTens*grad(uR), grad(vR))
)*dx

############################
# Stiffness matrix and Solver
############################
solver = PETSc.KSP().create(domain.comm)
solver.setType("preonly")
solver.getPC().setType("lu")

A = assemble_matrix(fem.form(a))
A.assemble()
solver.setOperators(A)


############################
# main loop
############################


uS_out = fem.Function(V_scal, name="Susceptible")
uI_out = fem.Function(V_scal, name="Infected")
uR_out = fem.Function(V_scal, name="Rabid")

uS_out.interpolate(U0.sub(0))
uI_out.interpolate(U0.sub(1))
uR_out.interpolate(U0.sub(2))

#xdmf.write_function(uS_out, 0)
#xdmf.write_function(uI_out, 0)
xdmf.write_function(uR_out, 0)

for i in range(int(T/dt)):
	#compute RHS
	# In the weak form, switch between logistic and exponential decay for S
	l = (
		uS0*vS + dt*vS*(
			-beta*uS0*uR0 
			+ ufl.conditional(
				ufl.lt(K_func, .01),
				-r*uS0,                                    # exponential decay in water
				r*uS0*(1-(uS0+uI0+uR0)/K_func)            # logistic growth on land
			)
		)
		+ uI0*vI + dt*vI*(beta*uS0*uR0 - sigma*uI0)
		+ uR0*vR + dt*vR*(sigma*uI0 - alpha*uR0)
	)*dx

	L=assemble_vector(fem.form(l))
	L.assemble()

	solver.solve(L, U1.x.petsc_vec)

	U0.x.array[:]=U1.x.array
	U0.x.scatter_forward()

	if i%3==0: #update xmdf file every 3 frames
		#uS_out.interpolate(U0.sub(0))
		#uI_out.interpolate(U0.sub(1))
		uR_out.interpolate(U0.sub(2))
        
		timestamp = (i + 1) * dt
		#xdmf.write_function(uS_out, timestamp)
		#xdmf.write_function(uI_out, timestamp)
		xdmf.write_function(uR_out, timestamp)

	if i%25==0: #print progress every 25 frames
		print(i*dt/T) 

xdmf.close()
