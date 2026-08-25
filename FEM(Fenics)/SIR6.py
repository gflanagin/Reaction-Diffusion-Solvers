############################
# imports
############################
from mpi4py import MPI
import numpy as np
import basix.ufl

import dolfinx
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.fem import Function, functionspace, Constant
from dolfinx.io import XDMFFile

from petsc4py import PETSc

import ufl
from ufl import (
    TrialFunction, TestFunction,
    split, grad, dot, exp,
    dx, CellNormal, as_vector, Identity, outer,
)

from LC_to_capacity import land_cover_to_carrying_capacity

import sys

############################
# file paths
############################
MESH_FILE = "/mnt/c/Users/flanagg/Downloads/SIRbase/terrain5.xdmf"
LC_CLASS_FILE = "land_cover_classes.npy"
LC_DIFF_FILE  = "land_cover_diffusivity.npy"
OUTPUT_DIR    = "results"

############################
# time parameters
############################
T  = 1.0
dt = 0.01

############################
# terrain / diffusion
############################
isotropy = 0.02
kappa    = 200

DS = 100    # diffusion coefficient - susceptible
DI = 100    # diffusion coefficient - infected
DD = 100    # diffusion coefficient - dying

############################
# CWD disease parameters
############################
alpha   = 365 / 5
sigma   = 365 / 28
beta    = 80
r       = 1.5
K_base  = 2
p_env   = 0.5
delta_e = 0.1
rho     = 1.0

############################
# wolf parameters
############################
p1        = 365/5    # predation rate on S, I
p2        = 730/5    # predation rate on D (preferential)
sigma_w   = 15.0    # cohesion strength
lambda_w  = 1.0    # movement cost
r_w       = 1.5    # wolf growth rate
W_star = 0.005    # target local wolf density — cohesive below, dispersive above
W_pd = 1/15 

# Helmholtz cohesion parameter.
# Replaces the non-local integral G(x) = int w(x,y) W(y)(y-x)dy with a
# local PDE:  (I - beta_helm DTens grad.grad) phi = W, G = 0.5 DTens grad(phi).
# beta_helm ~ 1 matches the Gaussian kernel w = exp(-|d|^2_M) length scale.
# Increase to broaden cohesion range, decrease to tighten it.
beta_helm = 1.0

############################
# initial conditions
############################
outbreak_xy  = (162.0, 139.0)
outbreak_var = 500_000.0
outbreak_amp = 0.01

wolf_xy  = (-1000,-1000)
wolf_var = 20_000.0
wolf_z0  = 1500.0       # total initial wolf population (integral of W)

############################
# output cadence
############################
output_every = 3
print_every  = 10

# ==================================================================
#                       MESH & FUNCTION SPACES
# ==================================================================
with XDMFFile(MPI.COMM_WORLD, MESH_FILE, "r") as xf:
    domain = xf.read_mesh()


P1    = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
mixed = basix.ufl.mixed_element([P1, P1, P1, P1])
P     = fem.functionspace(domain, mixed)

V_scal = fem.functionspace(domain, ("CG", 1))

# SIDE trial / test functions
U_tr = TrialFunction(P)
uS, uI, uD, uE = split(U_tr)
U_te = TestFunction(P)
vS, vI, vD, vE = split(U_te)

# SIDE solution vectors
U0 = fem.Function(P)
uS0, uI0, uD0, uE0 = split(U0)
U1 = fem.Function(P)

# subspace DOF maps
V_S, dof_map_S = P.sub(0).collapse()
V_I, dof_map_I = P.sub(1).collapse()
V_D, dof_map_D = P.sub(2).collapse()
V_E, dof_map_E = P.sub(3).collapse()

# scalar helpers for extracting compartments each step
S_func = Function(V_scal, name="S_tmp")
I_func = Function(V_scal, name="I_tmp")
D_func = Function(V_scal, name="D_tmp")

# ==================================================================
#               LAND COVER / DIFFUSIVITY / CARRYING CAPACITY
# ==================================================================
land_cover_classes = np.load(LC_CLASS_FILE).astype(np.int32)
V_lc = fem.functionspace(domain, ("Lagrange", 1))

land_cover_func      = fem.Function(V_lc)
land_cover_func.name = "Land_Cover_Class"

lcD_values = np.load(LC_DIFF_FILE)
lcD_scale  = Function(V_scal)
K_values = land_cover_to_carrying_capacity(land_cover_classes, K_base=K_base)
K_func   = Function(V_scal)

local_dofs_lc = V_lc.dofmap.index_map.local_range
n_local = V_lc.dofmap.index_map.size_local
land_cover_func.x.array[:n_local] = land_cover_classes[local_dofs_lc[0]:local_dofs_lc[1]]   # owned DOFs: slice from global array
land_cover_func.x.scatter_forward() 

local_dofs_sc = V_scal.dofmap.index_map.local_range
n_local_sc    = V_scal.dofmap.index_map.size_local

lcD_scale.x.array[:n_local_sc] = lcD_values[local_dofs_sc[0]:local_dofs_sc[1]]
K_func.x.array[:n_local_sc] = K_values[local_dofs_sc[0]:local_dofs_sc[1]]

lcD_scale.x.scatter_forward()
K_func.x.scatter_forward() 

with io.XDMFFile(domain.comm, "land_cover_classes.xdmf", "w") as xdmf_lc:
    xdmf_lc.write_mesh(domain)
    xdmf_lc.write_function(land_cover_func)

# ==================================================================
#                       INITIAL CONDITIONS
# ==================================================================
K_S = Function(V_S)
K_S.interpolate(fem.Expression(K_func, V_S.element.interpolation_points))
U0.sub(0).x.array[dof_map_S] = K_S.x.array
U0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0))
ox, oy = outbreak_xy
U0.sub(2).interpolate(
    lambda x: outbreak_amp * np.exp(-((x[0]-ox)**2 + (x[1]-oy)**2) / outbreak_var))
U0.sub(3).interpolate(lambda x: np.full(x.shape[1], 0.0))
U0.x.scatter_forward()

# ==================================================================
#              DIFFUSION TENSOR (terrain-aware, anisotropic)
# ==================================================================
n_cell    = CellNormal(domain)
g_vec     = as_vector((0, 0, -1))
cos_theta = dot(g_vec, n_cell)

es_raw  = g_vec - dot(g_vec, n_cell) * n_cell
es_norm = ufl.sqrt(dot(es_raw, es_raw))
xdir       = as_vector((1, 0, 0))
x_proj_raw = xdir - dot(xdir, n_cell) * n_cell
x_proj     = x_proj_raw / ufl.sqrt(dot(x_proj_raw, x_proj_raw))
es = ufl.conditional(ufl.gt(es_norm, 1e-6), es_raw / es_norm, x_proj)

activation = (1 + exp(50 * (0.96 - 1))) / (1 + exp(50 * (0.96 - abs(cos_theta))))
Ds_ufl = kappa * (isotropy + (1 - isotropy) * activation)
Dc_ufl = kappa
DTens  = lcD_scale * ((Ds_ufl - Dc_ufl) * outer(es, es)
                      + Dc_ufl * (Identity(3) - outer(n_cell, n_cell)))

# ==================================================================
#                     WOLF FUNCTIONS
# ==================================================================
W_func   = Function(V_scal, name="W_wolf")    # wolf density
K_wolf   = Function(V_scal, name="K_wolf")    # deer reaction kernel p1(S+I)+p2 D
phi_func = Function(V_scal, name="phi_helm")  # Helmholtz auxiliary field for cohesion

# wolf initial condition: Gaussian with integral = wolf_z0
wx, wy = wolf_xy
W_func.interpolate(lambda x: np.exp(-((x[0]-wx)**2 + (x[1]-wy)**2) / wolf_var))
W_int = fem.assemble_scalar(fem.form(W_func * dx))
if abs(W_int) > 1e-14:
    W_func.x.array[:] *= wolf_z0 / W_int
W_func.x.scatter_forward()

# ==================================================================
#   CONSISTENT MASS MATRIX FOR WOLF UPDATE (assembled once)
# ==================================================================
W_trial_mass = TrialFunction(V_scal)
W_test_mass  = TestFunction(V_scal)
a_mass = W_trial_mass * W_test_mass * dx
a_mass_form = fem.form(a_mass)
A_mass = assemble_matrix(a_mass_form)
A_mass.assemble()
 
solver_mass = PETSc.KSP().create(domain.comm)
solver_mass.setType("cg")
solver_mass.getPC().setType("hypre")
solver_mass.getPC().setHYPREType("boomeramg")
solver_mass.setTolerances(rtol=1e-8, max_it=100)
solver_mass.setOperators(A_mass)
 
W_new = Function(V_scal)

# ==================================================================
#   SIDE BILINEAR FORM (assembled once, time-independent)
# ==================================================================
a_side = (
      uS * vS + dt * DS * dot(DTens * grad(uS), grad(vS))
    + uI * vI + dt * DI * dot(DTens * grad(uI), grad(vI))
    + uD * vD + dt * DD * dot(DTens * grad(uD), grad(vD))
    + uE * vE
) * dx

a_side_form = fem.form(a_side)
A_side = assemble_matrix(a_side_form)
A_side.assemble()

solver_side = PETSc.KSP().create(domain.comm)
solver_side.setType("gmres")
solver_side.getPC().setType("hypre")
solver_side.getPC().setHYPREType("boomeramg")
solver_side.setTolerances(rtol=1e-8, max_it=200)
solver_side.setOperators(A_side)

# ==================================================================
#   SIDE LINEAR FORM (RHS updated in-place each step)
#   Wolf predation: -p1 S W, -p1 I W, -p2 D W
# ==================================================================
l_side = (
    uS0 * vS + dt * vS * (
        ufl.conditional(ufl.lt(K_func, 0.01),
                        -r * uS0,
                        r * uS0 * (1 - (uS0 + uI0 + uD0) / K_func))
        - beta  * uS0 * uD0
        - rho   * uE0 * uS0
        - p1    * uS0 * W_func
    )
    + uI0 * vI + dt * vI * (
          beta  * uS0 * uD0
        + rho   * uE0 * uS0
        - sigma * uI0
        - p1    * uI0 * W_func
    )
    + uD0 * vD + dt * vD * (
          sigma * uI0
        - alpha * uD0
        - p2    * uD0 * W_func
    )
    + uE0 * vE + dt * vE * (
          p_env   * uD0
        - delta_e * uE0
    )
) * dx

l_side_form = fem.form(l_side)

# ==================================================================
#   HELMHOLTZ SYSTEM FOR COHESION  (matrix assembled once)
#
#   (I - beta_helm DTens grad.grad) phi = W
#
#   Approximates phi(x) = int exp(-|d_tilde|^2_M) W(y) dy for slowly
#   varying W.  Differentiating phi w.r.t. x gives:
#     grad(phi) = 2 M G  =>  G = 0.5 DTens grad(phi)
#   which is exact for slowly-varying W and costs only one scalar
#   linear solve per step with a constant (time-independent) matrix.
# ==================================================================
phi_trial = TrialFunction(V_scal)
phi_test  = TestFunction(V_scal)

a_helm = (phi_trial * phi_test
          + beta_helm * dot(DTens * grad(phi_trial), grad(phi_test))) * dx
l_helm = W_func * (W_star - W_func) * phi_test * dx

a_helm_form = fem.form(a_helm)
l_helm_form = fem.form(l_helm)

A_helm = assemble_matrix(a_helm_form)
A_helm.assemble()

solver_helm = PETSc.KSP().create(domain.comm)
solver_helm.setType("cg")
solver_helm.getPC().setType("hypre")
solver_helm.getPC().setHYPREType("boomeramg")
solver_helm.setTolerances(rtol=1e-8, max_it=100)
solver_helm.setOperators(A_helm)

# ==================================================================
#   WOLF PDE  (explicit Euler, consistent mass solve)
#
#   dW/dt = -div(W V*) + W(K - d)
#   V* = (1/2 lambda) DTens (grad K + sigma_w G)
#
#   Forward Euler weak form after IBP (no-flux BC):
#     M_mass W_new = b
#   where b = assemble_vector of l_wolf.
#   M_mass is constant so it is assembled and factored once.
#   CFL condition: dt < h / |V*|.
# ==================================================================
G_ufl      = 0.5 * DTens * grad(phi_func)
V_star_ufl = (1.0 / (2.0 * lambda_w)) * DTens * (grad(K_wolf) + sigma_w * G_ufl)

W_test = TestFunction(V_scal)

l_wolf = (W_func * W_test
          + dt * W_func * dot(V_star_ufl, grad(W_test))
          + dt * W_func * r_w* (1-W_func*p1/(W_pd*K_wolf)) * W_test) * dx

l_wolf_form = fem.form(l_wolf)

# ==================================================================
#                         OUTPUT SETUP
# ==================================================================
uS_out = Function(V_scal, name="Susceptible")
uI_out = Function(V_scal, name="Infected")
uD_out = Function(V_scal, name="Dying")
uE_out = Function(V_scal, name="Environment")
uW_out = Function(V_scal, name="WolfDensity")

xdmf_files = {}
for tag in ("S", "I", "D", "E", "W"):
    xf = io.XDMFFile(domain.comm, f"{OUTPUT_DIR}/CWD_{tag}.xdmf", "w")
    xf.write_mesh(domain)
    xdmf_files[tag] = xf


def write_outputs(t):
    uS_out.interpolate(U0.sub(0));  xdmf_files["S"].write_function(uS_out, t)
    uI_out.interpolate(U0.sub(1));  xdmf_files["I"].write_function(uI_out, t)
    uD_out.interpolate(U0.sub(2));  xdmf_files["D"].write_function(uD_out, t)
    uE_out.interpolate(U0.sub(3));  xdmf_files["E"].write_function(uE_out, t)
    uW_out.x.array[:] = W_func.x.array
    uW_out.x.scatter_forward()
    xdmf_files["W"].write_function(uW_out, t)


dof_x = V_scal.tabulate_dof_coordinates()[:, 0]
dof_y = V_scal.tabulate_dof_coordinates()[:, 1]

print(f"x range: [{dof_x.min():.0f}, {dof_x.max():.0f}], domain rank={domain.comm.rank}", flush=True)
print(f"y range: [{dof_y.min():.0f}, {dof_y.max():.0f}], domain rank={domain.comm.rank}", flush=True)

if domain.comm.rank == 0:
    log = open("progress.log", "w", buffering=1)
    
# ==================================================================
#                         MAIN TIME LOOP
# ==================================================================
W_total_form = fem.form(W_func * dx)
write_outputs(0.0)
for i in range(int(T / dt)):
    t = (i + 1) * dt

    # ---- extract S, I, D into scalar helpers ----
    S_func.interpolate(U0.sub(0))
    I_func.interpolate(U0.sub(1))
    D_func.interpolate(U0.sub(2))

    # ---- wolf reaction kernel K = p1(S+I) + p2 D ----
    K_wolf.x.array[:] = p1 * (S_func.x.array + I_func.x.array) + p2 * D_func.x.array
    K_wolf.x.scatter_forward()

    # ---- Helmholtz solve: (I - beta DTens grad.grad) phi = W ----
    b_helm = assemble_vector(l_helm_form)
    b_helm.assemble()
    solver_helm.solve(b_helm, phi_func.x.petsc_vec)
    phi_func.x.scatter_forward()
    b_helm.destroy()

    # ---- SIDE solve ----
    b_side = assemble_vector(l_side_form)
    b_side.assemble()
    solver_side.solve(b_side, U1.x.petsc_vec)
    U0.x.array[:] = U1.x.array
    U0.x.scatter_forward()

    # ---- explicit lumped-mass wolf update (vector assembly only) ----
    b_wolf = assemble_vector(l_wolf_form)
    b_wolf.assemble()
    solver_mass.solve(b_wolf, W_new.x.petsc_vec)
    W_func.x.array[:] = np.maximum(W_new.x.array, 0.0)
    W_func.x.scatter_forward()
    b_wolf.destroy()

    # ---- output ----
    if i % output_every == 0:
        write_outputs(t)
    if i % print_every == 0 and domain.comm.rank == 0:
        W_total = domain.comm.allreduce(fem.assemble_scalar(W_total_form), op=MPI.SUM)
        log.write(f"t = {t:.2f}/{T}  |  int W = {W_total:.4f}\n")
        log.flush()

# ==================================================================
#                            CLEANUP
# ==================================================================
if domain.comm.rank == 0:
    log.close()
for xf in xdmf_files.values():
    xf.close()

print("Done.")