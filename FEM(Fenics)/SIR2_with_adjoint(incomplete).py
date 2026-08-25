"""
SIR2_with_adjoint.py
====================
Your existing SIR2.py forward solve with StateCheckpointer wired in,
plus a skeleton adjoint / gradient-descent loop ready for your weak forms.

Changes from SIR2.py are marked with  # <<< ADDED/CHANGED >>>
Everything else is identical to your original file.
"""

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

from state_checkpointer import StateCheckpointer   # <<< ADDED >>>

############################
# model and time parameters
############################
T        = 15
dt       = .01
isotropy = .02
DS       = 100
DI       = 100
DR       = 100
kappa    = 200
alpha    = 365 / 5
sigma    = 365 / 28
beta     = 80
K        = 2
r        = 1.5

n_steps  = int(T / dt)   # <<< ADDED: used by checkpointer >>>

############################
# domain, function space, and functions
############################
mesh_data = read_from_msh("terrain3.msh", MPI.COMM_WORLD, gdim=3)
domain    = mesh_data.mesh

P1    = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
mixed = basix.ufl.mixed_element([P1, P1, P1])
P     = fem.functionspace(domain, mixed)

U  = TrialFunction(P)
uS, uI, uR = split(U)
V  = TestFunction(P)
vS, vI, vR = split(V)

U0 = fem.Function(P)
uS0, uI0, uR0 = split(U0)

V_scal = fem.functionspace(domain, ("CG", 1))
V_lc   = fem.functionspace(domain, ("Lagrange", 1))

land_cover_classes    = np.load("land_cover_classes.npy").astype(np.int32)
land_cover_func       = fem.Function(V_lc)
land_cover_func.name  = "Land_Cover_Class"
land_cover_func.x.array[:] = land_cover_classes

lcD_values = np.load("land_cover_diffusivity.npy")
lcD_scale  = Function(V_scal)
lcD_scale.x.array[:] = lcD_values
lcD_scale.x.scatter_forward()

K_values = land_cover_to_carrying_capacity(land_cover_classes, K_base=K)
K_func   = Function(V_scal)
K_func.x.array[:] = K_values
K_func.x.scatter_forward()

with io.XDMFFile(domain.comm, "land_cover_classes.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(land_cover_func)

V_S, dof_map = P.sub(0).collapse()
K_S = Function(V_S)
K_S.interpolate(fem.Expression(K_func, V_S.element.interpolation_points))
U0.sub(0).x.array[dof_map] = K_S.x.array
U0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0))
U0.sub(2).interpolate(lambda x: .01 * np.exp(-((x[0] - 162)**2 + (x[1] - 139)**2) / 500000))
U0.x.scatter_forward()

U1 = fem.Function(P)

################
# plotter
################
xdmf = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/SIR2_stretched_mesh2.xdmf", "w")
xdmf.write_mesh(domain)

###################
# Construct tensor
###################
n         = CellNormal(domain)
g         = as_vector((0, 0, -1))
cos_theta = dot(g, n)

es_raw  = g - dot(g, n) * n
es_norm = ufl.sqrt(dot(es_raw, es_raw))

xdir        = as_vector((1, 0, 0))
x_proj_raw  = xdir - dot(xdir, n) * n
x_proj      = x_proj_raw / ufl.sqrt(dot(x_proj_raw, x_proj_raw))

es = ufl.conditional(
    ufl.gt(es_norm, 1E-6),
    es_raw / es_norm,
    x_proj
)

activation = (1 + exp(50 * (.96 - 1))) / (1 + exp(50 * (.96 - abs(cos_theta))))

Ds    = kappa * (isotropy + (1 - isotropy) * activation)
Dc    = kappa
DTens = lcD_scale * ((Ds - Dc) * outer(es, es) + Dc * (Identity(3) - outer(n, n)))

a = (
    uS * vS + dt * DS * dot(DTens * grad(uS), grad(vS))
    + uI * vI + dt * DI * dot(DTens * grad(uI), grad(vI))
    + uR * vR + dt * DR * dot(DTens * grad(uR), grad(vR))
) * dx

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
# Checkpointer setup          <<< ADDED >>>
############################
# Memory estimate before choosing strategy:
#   dof_size * n_steps * 8 bytes  e.g. 30k DOFs * 1500 steps * 8 = ~360 MB
dof_size    = U0.x.array.size
est_gb      = dof_size * n_steps * 8 / 1e9
strategy    = "memory" if est_gb < 2.0 else "hdf5"

checkpointer = StateCheckpointer(
    filepath   = "forward_states.h5",   # ignored when strategy="memory"
    n_steps    = n_steps,
    dof_size   = dof_size,
    strategy   = strategy,
    cache_size = 150,                   # ~150 * dof_size * 8 bytes in RAM
)

print(f"[checkpointer] strategy={strategy}, est_size={est_gb:.2f} GB")

############################
# Outer optimal-control loop  <<< ADDED >>>
############################
# Wrap your existing forward solve in an outer loop over gradient-descent
# iterations.  Each outer iteration does one forward pass + one backward pass.

N_CONTROL_ITER = 20      # gradient descent steps
STEP_SIZE      = 1e-4    # learning rate on the control variable

# --- Initialise your control here ---
# e.g. control = beta (scalar), or a spatial field stored as a Function
control_beta = beta      # placeholder: optimising transmission rate

for ctrl_iter in range(N_CONTROL_ITER):

    ############################
    # FORWARD PASS
    ############################
    # Reset state to initial condition each outer iteration
    U0.sub(0).x.array[dof_map] = K_S.x.array
    U0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0))
    U0.sub(2).interpolate(lambda x: .01 * np.exp(-((x[0] - 162)**2 + (x[1] - 139)**2) / 500000))
    U0.x.scatter_forward()

    uS_out = fem.Function(V_scal, name="Susceptible")
    uI_out = fem.Function(V_scal, name="Infected")
    uR_out = fem.Function(V_scal, name="Rabid")

    uS_out.interpolate(U0.sub(0))
    uI_out.interpolate(U0.sub(1))
    uR_out.interpolate(U0.sub(2))
    xdmf.write_function(uS_out, 0)
    xdmf.write_function(uI_out, 0)
    xdmf.write_function(uR_out, 0)

    # Save step 0 (initial condition)               <<< ADDED >>>
    checkpointer.save(0, U0)

    with checkpointer.forward_context() as cp:     # <<< ADDED: keeps HDF5 open >>>
        for i in range(n_steps):
            l = (
                uS0 * vS + dt * vS * (
                    -control_beta * uS0 * uR0
                    + ufl.conditional(
                        ufl.lt(K_func, .01),
                        -r * uS0,
                        r * uS0 * (1 - (uS0 + uI0 + uR0) / K_func)
                    )
                )
                + uI0 * vI + dt * vI * (control_beta * uS0 * uR0 - sigma * uI0)
                + uR0 * vR + dt * vR * (sigma * uI0 - alpha * uR0)
            ) * dx

            L = assemble_vector(fem.form(l))
            L.assemble()

            solver.solve(L, U1.x.petsc_vec)

            U0.x.array[:] = U1.x.array
            U0.x.scatter_forward()

            cp.save(i + 1, U0)                     # <<< ADDED: checkpoint every step >>>

            if i % 3 == 0:
                uS_out.interpolate(U0.sub(0))
                uI_out.interpolate(U0.sub(1))
                uR_out.interpolate(U0.sub(2))
                timestamp = (i + 1) * dt
                xdmf.write_function(uS_out, timestamp)
                xdmf.write_function(uI_out, timestamp)
                xdmf.write_function(uR_out, timestamp)

            if i % 25 == 0:
                print(f"[fwd iter {ctrl_iter}]  t={i*dt:.2f}/{T}")

    ############################
    # BACKWARD (ADJOINT) PASS    <<< ADDED >>>
    ############################
    # Allocate adjoint functions in the same mixed space
    Lambda  = fem.Function(P)              # adjoint state  [λ_S, λ_I, λ_R]
    Lambda1 = fem.Function(P)             # adjoint at next step (already computed)

    # Zero terminal condition: ∂Φ/∂U(T) = 0 unless your objective has a
    # Mayer term at the terminal time.
    Lambda.x.array[:] = 0.0
    Lambda.x.scatter_forward()

    # Accumulate ∂J/∂control across all steps
    dJ_dbeta = 0.0   # scalar gradient placeholder

    # reverse_iterator yields (step_index, full_mixed_DOF_array) backwards
    U_fwd = fem.Function(P)   # scratch function for loading forward state

    for i, fwd_state in checkpointer.reverse_iterator(prefetch=20):
        # Restore forward state for this timestep
        U_fwd.x.array[:] = fwd_state
        U_fwd.x.scatter_forward()
        uS_fwd, uI_fwd, uR_fwd = split(U_fwd)

        # === FILL IN YOUR ADJOINT WEAK FORMS HERE ===
        #
        # Canonical structure (implicit Euler adjoint, reading forward state):
        #
        #   λ_adj = TrialFunction(P)
        #   μ_adj = TestFunction(P)
        #   λS, λI, λR = split(λ_adj)
        #   μS, μI, μR = split(μ_adj)
        #
        #   a_adj = (
        #       λS*μS + dt*DS*dot(DTens*grad(λS), grad(μS)) + ...
        #       # + linearisation of reaction terms w.r.t. U_fwd
        #   )*dx
        #
        #   l_adj = (
        #       # RHS = terminal cost gradient + running cost gradient ∂L/∂U
        #       # evaluated at U_fwd, plus contribution from Lambda1 (next adjoint step)
        #   )*dx
        #
        # solver_adj.solve(L_adj, Lambda.x.petsc_vec)
        # Lambda.x.scatter_forward()

        # Gradient contribution at this step: ∂H/∂β = -λ_S * uS * uR + λ_I * uS * uR
        # (gradient of the Hamiltonian w.r.t. the control)
        # NOTE: these are pointwise integrals — assemble with fem.assemble_scalar
        #
        # lam_S_arr, lam_I_arr, _ = Lambda1.sub(0), Lambda1.sub(1), Lambda1.sub(2)
        # dJ_dbeta += dt * fem.assemble_scalar(fem.form(
        #     (-lam_S_arr + lam_I_arr) * uS_fwd * uR_fwd * dx
        # ))

        Lambda1.x.array[:] = Lambda.x.array   # shift: current → "next" for next step

    print(f"[ctrl iter {ctrl_iter}]  dJ/dβ = {dJ_dbeta:.6e}")
    print(checkpointer.summary())

    # Gradient descent update on control
    control_beta -= STEP_SIZE * dJ_dbeta
    control_beta  = max(0.0, control_beta)   # keep physically meaningful

xdmf.close()
