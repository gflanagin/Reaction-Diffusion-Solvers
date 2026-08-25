############################
# imports
############################
from mpi4py import MPI
import numpy as np
import basix.ufl

from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.fem import Function, functionspace
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
# file paths
############################
MESH_FILE     = "terrain6.msh"
LC_CLASS_FILE = "land_cover_classes.npy"
LC_DIFF_FILE  = "land_cover_diffusivity.npy"
OUTPUT_DIR    = "results"

############################
# time parameters
############################
T  = 1
dt = 0.01

############################
# terrain / diffusion
############################
isotropy = 0.02
kappa    = 200

DS = 10    # diffusion coefficient - susceptible
DI = 7.5    # diffusion coefficient - infected
DD = 2.5   # diffusion coefficient - dying
DE = .001    # diffusion coefficient - environmental prions

############################
# CWD disease parameters
############################
sigma   = 365 / 28   # I -> D progression rate
alpha   = 365 / 5    # CWD-induced mortality rate of D
d_d     = 0.1        # background natural deer mortality rate
beta1   = 80         # direct transmission rate D->S
beta2   = 1.0        # environmental transmission rate E->S
r       = 1.5        # logistic growth rate
K_base  = 2          # carrying capacity base (scaled by land cover)

# prion parameters
rho_I   = 0.1        # prion shedding rate from I (living)
rho_R   = 0.5        # prion shedding rate from D (living), rho_R > rho_I
Lambda0 = 100.0      # prion load released per decomposing carcass
eta     = 0.04       # fraction of prions surviving canid digestion (0.03-0.05)
delta_e = 0.01       # environmental prion decay rate

############################
# wolf parameters (two packs)
############################
# Holling Type II functional response
p1      = 1.0        # baseline search rate for S
theta_I = 1.5        # selectivity multiplier for I (> 1)
theta_R = 4.0        # selectivity multiplier for D (>> theta_I)
h_hunt  = 0.1        # handling time per deer

# wolf movement / cohesion
lambda_w  = 0.001      # movement cost (terrain penalty weight)
sigma_w   = 0.0     # cohesion penalty/reward strength
wolf_density_threshold = 0.75   # cohesion-to-dispersal crossover as fraction of carrying capacity
beta_helm = 1.0      # Helmholtz cohesion length scale

# wolf demography
r_w   = 0.5          # wolf intrinsic growth rate
zeta  = 0.1          # wolf per deer carrying capacity

# scent marking
eta_scent = 1.0      # sensitivity to foreign scent-mark gradient
m_mark    = 0.8      # scent-mark enhancement factor in presence of foreign marks
f_decay   = 0.05     # scent-mark first-order decay rate

############################
# landscape of fear (prey taxis)
############################
chi_S = .5          # avoidance sensitivity - healthy deer
chi_I = .3          # avoidance sensitivity - pre-clinical deer
chi_D = 0 # (clinical deer unresponsive)

############################
# initial conditions
############################
outbreak_xy  = (0,0)
outbreak_var = 250_000.0
outbreak_amp = 0.01

# pack 1 and pack 2 initial positions and sizes
wolf1_xy  = (-1000.0, -1000.0)
wolf2_xy  = (1000,1000)
wolf_var  = 50_000.0
wolf1_z0  = 200.0
wolf2_z0  = 200.0

############################
# output cadence
############################
output_every = 3
print_every  = 10

# ==================================================================
#                       MESH & FUNCTION SPACES
# ==================================================================
mesh_data = read_from_msh(MESH_FILE, MPI.COMM_WORLD, gdim=3)
domain    = mesh_data.mesh

P1    = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
mixed = basix.ufl.mixed_element([P1, P1, P1, P1])
P     = fem.functionspace(domain, mixed)

V_scal = fem.functionspace(domain, ("CG", 1))

# SIRE trial / test functions
U_tr = TrialFunction(P)
uS, uI, uD, uE = split(U_tr)
U_te = TestFunction(P)
vS, vI, vD, vE = split(U_te)

# SIRE solution vectors
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
land_cover_func.x.array[:] = land_cover_classes

lcD_values = np.load(LC_DIFF_FILE)
lcD_scale  = Function(V_scal)
lcD_scale.x.array[:] = lcD_values
lcD_scale.x.scatter_forward()

K_values = land_cover_to_carrying_capacity(land_cover_classes, K_base=K_base)
K_func   = Function(V_scal)
K_func.x.array[:] = K_values
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
#   WOLF AND SCENT-MARK FUNCTIONS (updated every step)
# ==================================================================
W1_func  = Function(V_scal, name="W1")         # pack 1 density
W2_func  = Function(V_scal, name="W2")         # pack 2 density
Wtot_func = Function(V_scal, name="Wtot")      # W1 + W2 (for taxis and reaction)
p_func   = Function(V_scal, name="p_scent")    # pack 1 scent marks
q_func   = Function(V_scal, name="q_scent")    # pack 2 scent marks
K_wolf  = Function(V_scal, name="K_wolf")      # Hunting kernel 
phi1_func = Function(V_scal, name="phi1")      # Helmholtz field for W1 cohesion
phi2_func = Function(V_scal, name="phi2")      # Helmholtz field for W2 cohesion

# wolf initial conditions
wx1, wy1 = wolf1_xy
W1_func.interpolate(lambda x: np.exp(-((x[0]-wx1)**2 + (x[1]-wy1)**2) / wolf_var))
W1_int = fem.assemble_scalar(fem.form(W1_func * dx))
if abs(W1_int) > 1e-14:
    W1_func.x.array[:] *= wolf1_z0 / W1_int
W1_func.x.scatter_forward()

wx2, wy2 = wolf2_xy
W2_func.interpolate(lambda x: np.exp(-((x[0]-wx2)**2 + (x[1]-wy2)**2) / wolf_var))
W2_int = fem.assemble_scalar(fem.form(W2_func * dx))
if abs(W2_int) > 1e-14:
    W2_func.x.array[:] *= wolf2_z0 / W2_int
W2_func.x.scatter_forward()

# scent marks initialised to zero
p_func.interpolate(lambda x: np.full(x.shape[1], 0.0))
q_func.interpolate(lambda x: np.full(x.shape[1], 0.0))

W_tr = TrialFunction(V_scal)
W_te = TestFunction(V_scal)

W1_new = Function(V_scal)
W2_new = Function(V_scal)

# ==================================================================
#   HELMHOLTZ SYSTEMS FOR COHESION (matrices assembled once)
#
#   (I - beta_helm DTens grad.grad) phi_i = W_i*(W_star-W)_i
#   G_i = 0.5 DTens grad(phi_i)
#   V*_i = (1/2lambda) DTens (grad K_i - eta_scent grad q_j + sigma_w G_i)
# ==================================================================
phi_tr = TrialFunction(V_scal)
phi_te = TestFunction(V_scal)

a_helm = (phi_tr * phi_te
          + beta_helm * dot(DTens * grad(phi_tr), grad(phi_te))) * dx
a_helm_form = fem.form(a_helm)
A_helm = assemble_matrix(a_helm_form)

l_helm1_form = fem.form(W1_func * (zeta * wolf_density_threshold * K_func - W1_func) * phi_te * dx)
l_helm2_form = fem.form(W2_func * (zeta * wolf_density_threshold * K_func - W2_func) * phi_te * dx)
A_helm.assemble()

solver_helm = PETSc.KSP().create(domain.comm)
solver_helm.setType("cg")
solver_helm.getPC().setType("hypre")
solver_helm.getPC().setHYPREType("boomeramg")
solver_helm.setTolerances(rtol=1e-8, max_it=100)
solver_helm.setOperators(A_helm)

# cohesion fields and optimal velocities in UFL
G1_ufl = 0.5 * DTens * grad(phi1_func)
G2_ufl = 0.5 * DTens * grad(phi2_func)

V1_star = (1.0 / (2.0 * lambda_w)) * DTens * (
    grad(K_wolf) - eta_scent * grad(q_func) + sigma_w * G1_ufl)
V2_star = (1.0 / (2.0 * lambda_w)) * DTens * (
    grad(K_wolf) - eta_scent * grad(p_func) + sigma_w * G2_ufl)

# ==================================================================
#   SIRE BILINEAR FORM
#
#   The landscape-of-fear taxis terms add -chi_S div(S grad Wtot) and
#   -chi_I div(I grad Wtot) to the S and I equations.  After IBP:
#     -chi * div(u grad Wtot) v = chi * u grad(Wtot).grad(v)
#                                - boundary term (zero, no-flux)
#   Wtot changes every step so the SIRE bilinear form must be
#   reassembled each step.
# ==================================================================
# SIRE LHS: compiled once. Wtot_func is a Function whose array is updated
# in-place each step -- UFL picks up the new values automatically on reassembly.
a_sire_form = fem.form((
      uS * vS + dt * DS * dot(DTens * grad(uS), grad(vS))
    + uI * vI + dt * DI * dot(DTens * grad(uI), grad(vI))
    + uD * vD + dt * DD * dot(DTens * grad(uD), grad(vD))
    + uE * vE + dt * DE * dot(DTens * grad(uE), grad(vE))
) * dx)

A_sire = assemble_matrix(a_sire_form)
A_sire.assemble()
solver_sire = PETSc.KSP().create(domain.comm)
solver_sire.setType("preonly")
solver_sire.getPC().setType("lu")
solver_sire.setOperators(A_sire)

# ==================================================================
#   SIRE LINEAR FORM (RHS, reassembled each step)
#
#   Holling Type II predation:
#     Phi_S = p1 S             / (1 + h p1 (S + theta_I I + theta_R D))
#     Phi_I = theta_I p1 I     / (1 + h p1 (S + theta_I I + theta_R D))
#     Phi_D = theta_R p1 D     / (1 + h p1 (S + theta_I I + theta_R D))
#
#   Environmental prion equation:
#     dE/dt = DE div(DTens grad E) + rho_I I + rho_R D
#           + Lambda0 (alpha D + d_d (I + D))    [carcass deposition]
#           - eta Lambda0 Wtot                   [wolf prion sink]
#           - delta_e E
# ==================================================================
# SIRE RHS: compiled once. All Functions (uS0/uI0/uD0/uE0 via U0, Wtot_func,
# K_func) are updated in-place each step.
_denom   = 1.0 + h_hunt * p1 * (uS0 + theta_I * uI0 + theta_R * uD0)
_Phi_S   = p1            * uS0 / _denom
_Phi_I   = theta_I * p1  * uI0 / _denom
_Phi_D   = theta_R * p1  * uD0 / _denom
_N0      = uS0 + uI0 + uD0

l_sire_form = fem.form((
    # S
    uS0 * vS 
    - dt * chi_S * dot(uS0 * DTens * grad(Wtot_func), grad(vS))
    + dt * vS * (
        ufl.conditional(ufl.lt(K_func, 0.01),
                        -r * uS0,
                        r * uS0 * (1.0 - _N0 / K_func))
        - beta1 * uS0 * uD0
        - beta2 * uE0 * uS0
        - _Phi_S * Wtot_func
    )
    # I
    + uI0 * vI 
    - dt * chi_I * dot(uI0 * DTens * grad(Wtot_func), grad(vI))
    + dt * vI * (
          beta1 * uS0 * uD0
        + beta2 * uE0 * uS0
        - sigma * uI0
        - _Phi_I * Wtot_func
    )
    # D
    + uD0 * vD 
    - dt * chi_D * dot(uD0 * DTens * grad(Wtot_func), grad(vD))
    + dt * vD * (
          sigma * uI0
        - alpha * uD0
        - _Phi_D * Wtot_func
    )
    # E: shedding from I and D + carcass deposition - wolf prion sink - decay
    + uE0 * vE 
    + dt * vE * (
          rho_I * uI0
        + rho_R * uD0
        + Lambda0 * (alpha * uD0 + d_d * (uI0 + uD0))
        - eta * Lambda0 * Wtot_func
        - delta_e * uE0
    )
) * dx)

# ==================================================================
#   WOLF TRANSPORT FORMS (reassembled each step since V* changes)
#
#   dW_i/dt = -div(W_i V*_i) + W_i r_w (1 - W_i / (zeta N))
#
#   Forward Euler weak form after IBP:
#     M W_i_new = b_i
# ==================================================================
# Wolf RHS forms: compiled once per pack. W1_func/W2_func, V1_star/V2_star,

h_mesh = 2.0 * ufl.CellDiameter(domain)
tau_W1 = h_mesh / (2.0 * ufl.sqrt(dot(V1_star, V1_star) + 1e-10))
tau_W2 = h_mesh / (2.0 * ufl.sqrt(dot(V2_star, V2_star) + 1e-10))

# bilinear form (LHS) - assembled each step since V* changes
a_wolf1 = fem.form((
    W_tr * W_te
    - dt * W_tr * dot(V1_star, grad(W_te))                                    # implicit advection
    + dt * tau_W1 * dot(V1_star, grad(W_tr)) * dot(V1_star, grad(W_te))       # SUPG
) * dx)

a_wolf2 = fem.form((
    W_tr * W_te
    - dt * W_tr * dot(V2_star, grad(W_te))                                    # implicit advection
    + dt * tau_W1 * dot(V2_star, grad(W_tr)) * dot(V2_star, grad(W_te))       # SUPG
) * dx)

# linear form (RHS) - reaction term stays explicit
l_wolf1 = fem.form((
    W1_func * W_te
    + dt * W1_func * r_w * (1.0 - W1_func / (zeta * K_func + 1e-10)) * W_te
) * dx)

l_wolf2 = fem.form((
    W2_func * W_te
    + dt * W2_func * r_w * (1.0 - W2_func / (zeta * K_func + 1e-10)) * W_te
) * dx)

# ==================================================================
#                         OUTPUT SETUP
# ==================================================================
uS_out  = Function(V_scal, name="Susceptible")
uI_out  = Function(V_scal, name="Infected")
uD_out  = Function(V_scal, name="Dying")
uE_out  = Function(V_scal, name="Environment")
uW1_out = Function(V_scal, name="Wolf1")
uW2_out = Function(V_scal, name="Wolf2")
up_out  = Function(V_scal, name="ScentP")
uq_out  = Function(V_scal, name="ScentQ")

xdmf_files = {}
for tag in ("S", "I", "D", "E", "W1", "W2", "P", "Q"):
    xf = io.XDMFFile(domain.comm, f"{OUTPUT_DIR}/CWD_{tag}.xdmf", "w")
    xf.write_mesh(domain)
    xdmf_files[tag] = xf

W_total_form = fem.form((W1_func + W2_func) * dx)


def write_outputs(t):
    uS_out.interpolate(U0.sub(0));   xdmf_files["S"].write_function(uS_out, t)
    uI_out.interpolate(U0.sub(1));   xdmf_files["I"].write_function(uI_out, t)
    uD_out.interpolate(U0.sub(2));   xdmf_files["D"].write_function(uD_out, t)
    uE_out.interpolate(U0.sub(3));   xdmf_files["E"].write_function(uE_out, t)
    uW1_out.x.array[:] = W1_func.x.array; uW1_out.x.scatter_forward()
    uW2_out.x.array[:] = W2_func.x.array; uW2_out.x.scatter_forward()
    up_out.x.array[:]  = p_func.x.array;  up_out.x.scatter_forward()
    uq_out.x.array[:]  = q_func.x.array;  uq_out.x.scatter_forward()
    xdmf_files["W1"].write_function(uW1_out, t)
    xdmf_files["W2"].write_function(uW2_out, t)
    xdmf_files["P"].write_function(up_out, t)
    xdmf_files["Q"].write_function(uq_out, t)


if domain.comm.rank == 0:
    log = open(f"{OUTPUT_DIR}/progress.log", "w", buffering=1)

dof_x = V_scal.tabulate_dof_coordinates()[:, 0]
dof_y = V_scal.tabulate_dof_coordinates()[:, 1]

if domain.comm.rank == 0:
    log.write(f"x range: [{dof_x.min():.0f}, {dof_x.max():.0f}]\n")
    log.write(f"y range: [{dof_y.min():.0f}, {dof_y.max():.0f}]\n")
    log.flush()

# ==================================================================
#                         MAIN TIME LOOP
# ==================================================================
write_outputs(0.0)

for i in range(int(T / dt)):
    t = (i + 1) * dt

    # ---- extract compartments into scalar helpers ----
    S_func.interpolate(U0.sub(0))
    I_func.interpolate(U0.sub(1))
    D_func.interpolate(U0.sub(2))

    # ---- total deer density and total wolf density ----
    Wtot_func.x.array[:] = W1_func.x.array + W2_func.x.array
    Wtot_func.x.scatter_forward()

    # ---- Holling Type II kernel (same denominator for both packs) ----
    denom_arr = (1.0 + h_hunt * p1
                 * (S_func.x.array + theta_I * I_func.x.array
                    + theta_R * D_func.x.array))
    K_num = p1 * (S_func.x.array + theta_I * I_func.x.array
                  + theta_R * D_func.x.array)
    K_wolf.x.array[:] = K_num / denom_arr
    K_wolf.x.scatter_forward()

    # ---- Helmholtz solves for cohesion fields ----
    for l_h_form, phi_f in ((l_helm1_form, phi1_func), (l_helm2_form, phi2_func)):
        b_h = assemble_vector(l_h_form)
        b_h.assemble()
        solver_helm.solve(b_h, phi_f.x.petsc_vec)
        phi_f.x.scatter_forward()
        b_h.destroy()

    # ---- SIRE solve ----
    b_sire = assemble_vector(l_sire_form)
    b_sire.assemble()
    solver_sire.solve(b_sire, U1.x.petsc_vec)
    U0.x.array[:] = U1.x.array
    U0.x.array[dof_map_S] = np.maximum(U0.x.array[dof_map_S], 0.0)
    U0.x.array[dof_map_I] = np.maximum(U0.x.array[dof_map_I], 0.0)
    U0.x.array[dof_map_D] = np.maximum(U0.x.array[dof_map_D], 0.0)
    U0.x.array[dof_map_E] = np.maximum(U0.x.array[dof_map_E], 0.0)
    U0.x.scatter_forward()
    b_sire.destroy()

    # ---- wolf transport (explicit, mass solve) ----
    for l_form, a_form, W_f, W_new_f in (
            (l_wolf1, a_wolf1, W1_func, W1_new),
            (l_wolf2, a_wolf2, W2_func, W2_new)):
        b_w = assemble_vector(l_form)
        b_w.assemble()
        A_w = assemble_matrix(a_form)
        A_w.assemble()
        solver_W = PETSc.KSP().create(domain.comm)
        solver_W.setType("gmres")
        solver_W.getPC().setType("ilu")
        solver_W.setTolerances(rtol=1e-6, max_it=100)
        solver_W.setOperators(A_w)
        solver_W.solve(b_w, W_new_f.x.petsc_vec)
        W_f.x.array[:] = np.maximum(W_new_f.x.array, 0.0)
        W_f.x.scatter_forward()
        b_w.destroy()
        A_w.destroy()

    # ---- scent mark ODEs (purely local, forward Euler) ----
    # dp/dt = W1 (1 + m q) - f p
    # dq/dt = W2 (1 + m p) - f q
    p_new = p_func.x.array + dt * (
        W1_func.x.array * (1.0 + m_mark * q_func.x.array) - f_decay * p_func.x.array)
    q_new = q_func.x.array + dt * (
        W2_func.x.array * (1.0 + m_mark * p_func.x.array) - f_decay * q_func.x.array)
    p_func.x.array[:] = np.maximum(p_new, 0.0)
    q_func.x.array[:] = np.maximum(q_new, 0.0)
    p_func.x.scatter_forward()
    q_func.x.scatter_forward()

    # ---- output ----
    if i % output_every == 0:
        write_outputs(t)
    if i % print_every == 0 and domain.comm.rank == 0:
        W_total = domain.comm.allreduce(
            fem.assemble_scalar(W_total_form), op=MPI.SUM)
        log.write(f"t = {t:.2f}/{T}  |  int(W1+W2) = {W_total:.4f}\n")
        log.flush()

# ==================================================================
#                            CLEANUP
# ==================================================================
for xf in xdmf_files.values():
    xf.close()
if domain.comm.rank == 0:
    log.write("Done.\n")
    log.close()