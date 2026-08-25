############################
# imports
############################
from mpi4py import MPI
import numpy as np
import basix.ufl

from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.fem import (Function, functionspace, Constant,
                          dirichletbc, locate_dofs_topological)
from dolfinx.io.gmsh import read_from_msh

from petsc4py import PETSc

import ufl
from ufl import (
    TrialFunction, TestFunction,
    split, grad, dot, exp,
    dx, CellNormal, as_vector, Identity, outer,
)

from LC_to_capacity import land_cover_to_carrying_capacity

############################
# model and time parameters
############################
T        = 5      # years
dt       = 0.01

isotropy = 0.02   # anisotropy parameter, must match mesh generation
DS = 100          # susceptible
DI = 100          # infected
DD = 100          # dying

kappa = 200
alpha = 365/5     # D removal rate (death / carcass decomposition)
sigma = 365/28    # I → D progression rate
beta  = 80        # direct D→S transmission rate
K     = 2         # carrying capacity base
r     = 1.5       # intrinsic growth rate
p_env   = 0.5     # prion shedding rate from dying deer (D → E)
delta_e = 0.1     # prion environmental decay rate
rho     = 1    # environmental transmission rate: rho*E*S subtracted from S, added to I

############################
# domain and function spaces
############################
mesh_data = read_from_msh("terrain4.msh", MPI.COMM_WORLD, gdim=3)
domain    = mesh_data.mesh

# Single CG1 element
P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)

# Mixed element for the four PDE compartments: S, I, D, E
# Wolf density W is NOT in the mixed system — it is a derived field from the wolf ODE
mixed = basix.ufl.mixed_element([P1, P1, P1, P1])
P     = fem.functionspace(domain, mixed)

# Scalar CG1 space (for output functions and wolf auxiliary fields)
V_scal = fem.functionspace(domain, ("CG", 1))
V_lc   = fem.functionspace(domain, ("Lagrange", 1))

# Trial / test functions
U  = TrialFunction(P)
uS, uI, uD, uE = split(U)
V  = TestFunction(P)
vS, vI, vD, vE = split(V)

# Solution at previous time step (symbolic split for UFL RHS)
U0 = fem.Function(P)
uS0, uI0, uD0, uE0 = split(U0)

############################
# land cover, diffusivity, carrying capacity
############################
land_cover_classes      = np.load("land_cover_classes.npy").astype(np.int32)
land_cover_func         = fem.Function(V_lc)
land_cover_func.name    = "Land_Cover_Class"
land_cover_func.x.array[:] = land_cover_classes

lcD_values = np.load("land_cover_diffusivity.npy")
lcD_scale  = Function(V_scal)
lcD_scale.x.array[:] = lcD_values
lcD_scale.x.scatter_forward()

K_values = land_cover_to_carrying_capacity(land_cover_classes, K_base=K)
K_func   = Function(V_scal)
K_func.x.array[:] = K_values
K_func.x.scatter_forward()

with io.XDMFFile(domain.comm, "land_cover_classes.xdmf", "w") as xdmf_lc:
    xdmf_lc.write_mesh(domain)
    xdmf_lc.write_function(land_cover_func)

############################
# initial conditions
############################
# Collapse each subspace to get DOF maps into the parent mixed array
V_S, dof_map_S = P.sub(0).collapse()
V_I, dof_map_I = P.sub(1).collapse()
V_D, dof_map_D = P.sub(2).collapse()
V_E, dof_map_E = P.sub(3).collapse()

#construct carrying capacity function space
K_S = Function(V_S)
K_S.interpolate(fem.Expression(K_func, V_S.element.interpolation_points))

# S: carrying capacity
# I: zero
# D: small Gaussian seed of dying deer (CWD outbreak origin)
# E: no prions in environment at t=0
U0.sub(0).x.array[dof_map_S] = K_S.x.array
U0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0))
U0.sub(2).interpolate(lambda x: 0.01 * np.exp(-((x[0] - 162)**2 + (x[1] - 139)**2) / 500_000))
U0.sub(3).interpolate(lambda x: np.full(x.shape[1], 0.0))
U0.x.scatter_forward()

# Forward-solve buffer
U1 = fem.Function(P)

############################
# diffusion tensor (terrain-aware anisotropic, unchanged from original)
############################
n         = CellNormal(domain)
g         = as_vector((0, 0, -1))
cos_theta = dot(g, n)

es_raw  = g - dot(g, n) * n
es_norm = ufl.sqrt(dot(es_raw, es_raw))

xdir       = as_vector((1, 0, 0))
x_proj_raw = xdir - dot(xdir, n) * n
x_proj     = x_proj_raw / ufl.sqrt(dot(x_proj_raw, x_proj_raw))

es = ufl.conditional(
    ufl.gt(es_norm, 1e-6),
    es_raw / es_norm,
    x_proj
)

activation = (1 + exp(50 * (0.96 - 1))) / (1 + exp(50 * (0.96 - abs(cos_theta))))
Ds    = kappa * (isotropy + (1 - isotropy) * activation)
Dc    = kappa
DTens = lcD_scale * ((Ds - Dc) * outer(es, es) + Dc * (Identity(3) - outer(n, n)))

############################
# bilinear form a  (LHS — time-independent, assembled once)
############################
a = (
      uS * vS + dt * DS * dot(DTens * grad(uS), grad(vS))
    + uI * vI + dt * DI * dot(DTens * grad(uI), grad(vI))
    + uD * vD + dt * DD * dot(DTens * grad(uD), grad(vD))
    + uE * vE          # E: mass matrix only — no spatial diffusion (purely local reaction)
) * dx

solver = PETSc.KSP().create(domain.comm)
solver.setType("preonly")
solver.getPC().setType("lu")

A = assemble_matrix(fem.form(a))
A.assemble()
solver.setOperators(A)

############################
# linear form l  (RHS — pre-compiled once; all variable data updated in-place)
#
# W_func is a FEM Function (not a symbolic split) so it can multiply split
# expressions in UFL.  Its buffer is updated each step before assembly.
############################
l = (
    # ---- S: logistic growth − direct transmission − environmental transmission − wolf predation
    uS0 * vS + dt * vS * (
        ufl.conditional(
            ufl.lt(K_func, 0.01),
            -r * uS0,                                          # exponential decay in water
            r * uS0 * (1 - (uS0 + uI0 + uD0) / K_func)       # logistic growth on land
        )
        - beta  * uS0 * uD0          # direct transmission from dying deer
        - rho   * uE0 * uS0          # environmental (prion) transmission
    )

    # ---- I: direct + environmental inflow − clinical progression − wolf predation
    + uI0 * vI + dt * vI * (
          beta  * uS0 * uD0          # direct transmission
        + rho   * uE0 * uS0          # environmental transmission
        - sigma * uI0                # progression to dying stage
    )

    # ---- D: progression from I − natural removal − selective wolf predation
    + uD0 * vD + dt * vD * (
          sigma * uI0                # inflow from infected
        - alpha * uD0                # natural death / removal
    )

    # ---- E: prion shedding from dying deer − environmental decay
    #         (purely reactive; spatial spread via DE diffusion in the bilinear form)
    + uE0 * vE + dt * vE * (
          p_env   * uD0              # prion shedding from dying deer
        - delta_e * uE0              # environmental prion decay
    )
) * dx

l_form = fem.form(l)   # compile once; buffers updated in-place each step

############################
# output setup
############################
xdmf_S = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/CWD_S.xdmf", "w")
xdmf_I = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/CWD_I.xdmf", "w")
xdmf_D = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/CWD_D.xdmf", "w")
xdmf_E = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/CWD_E.xdmf", "w")

xdmf_S.write_mesh(domain)
xdmf_I.write_mesh(domain)
xdmf_D.write_mesh(domain)
xdmf_E.write_mesh(domain)

uS_out = Function(V_scal, name="Susceptible")
uI_out = Function(V_scal, name="Infected")
uD_out = Function(V_scal, name="Dying")
uE_out = Function(V_scal, name="Environment")

def write_outputs(t):
    uS_out.interpolate(U0.sub(0))
    uI_out.interpolate(U0.sub(1))
    uD_out.interpolate(U0.sub(2))
    uE_out.interpolate(U0.sub(3))
    xdmf_S.write_function(uS_out, t)
    xdmf_I.write_function(uI_out, t)
    xdmf_D.write_function(uD_out, t)
    xdmf_E.write_function(uE_out, t)

S_func.x.array[:] = U0.x.array[dof_map_S]
I_func.x.array[:] = U0.x.array[dof_map_I]
D_func.x.array[:] = U0.x.array[dof_map_D]
S_func.x.scatter_forward()
I_func.x.scatter_forward()
D_func.x.scatter_forward()

xmin, xmax = dof_x.min(), dof_x.max()
ymin, ymax = dof_y.min(), dof_y.max()
print(f"x range: [{xmin:.0f},{xmax:.0f}]")
print(f"y range: [{ymin:.0f},{ymax:.0f}]")

############################
# main time loop
############################
for i in range(int(T / dt)):
    S_func.x.array[:] = U0.x.array[dof_map_S]
    I_func.x.array[:] = U0.x.array[dof_map_I]
    D_func.x.array[:] = U0.x.array[dof_map_D]
    S_func.x.scatter_forward()
    I_func.x.scatter_forward()
    D_func.x.scatter_forward()

    L = assemble_vector(l_form)
    L.assemble()

    solver.solve(L, U1.x.petsc_vec)

    U0.x.array[:] = U1.x.array
    U0.x.scatter_forward()

    if i % 3 == 0:
        write_outputs((i + 1) * dt)

    if i % 10 == 0:
        print(f"t = {i*dt:.2f}/{T}")

xdmf_S.close()
xdmf_I.close()
xdmf_D.close()
xdmf_E.close()
xdmf_N.close()
xdmf_W.close()

# Save wolf trajectory for analysis / adjoint post-processing
np.save("wolf_trajectory.npy", np.array(wolf_log))
print("Done. Wolf trajectory saved to wolf_trajectory.npy")