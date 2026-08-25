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
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

############################
# model and time parameters
############################
T        = 10      # years
dt       = 0.01
isotropy = 0.02   # anisotropy parameter, must match mesh generation

# Deer PDE diffusion coefficients
DS = 100          # susceptible
DI = 100          # infected
DD = 100          # dying
# E has no diffusion: prion transport in soil would require downslope advection,
# not the animal-movement anisotropic tensor.  E is purely reactive (local ODE).

kappa = 200
alpha = 365/5     # D removal rate (death / carcass decomposition)
sigma = 365/28    # I → D progression rate
beta  = 80        # direct D→S transmission rate
K     = 2         # carrying capacity base
r     = 1.5       # intrinsic growth rate

# Environmental prion parameters
p_env   = 0.5     # prion shedding rate from dying deer (D → E)
delta_e = 0.1     # prion environmental decay rate
rho     = 1    # environmental transmission rate: rho*E*S subtracted from S, added to I

# Wolf ODE parameters
sigma_w = 50.0    # pack territory Gaussian half-width (same coord units as mesh)
p1      = 7     # predation rate on S and I (non-symptomatic deer)
p2      = 15     # predation rate on D (selective — symptomatic deer are easier prey)
gamma_w = 3    # wolf pack mortality / emigration rate
wolf_v   = 1500.0     # scalar for pack center movement speed (c1 in the PDF)
c2      = .001  # distance sensitivity: controls how quickly the directional, weight falls off with distance from pack center (c2 in PDF eq. 3)
C0 = [3000,3000]
z0 = 1

############################
# domain and function spaces
############################
mesh_data = read_from_msh("terrain5.msh", MPI.COMM_WORLD, gdim=3)
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

# S: initialise at carrying capacity
K_S = Function(V_S)
K_S.interpolate(fem.Expression(K_func, V_S.element.interpolation_points))
U0.sub(0).x.array[dof_map_S] = K_S.x.array

# I: zero
U0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.0))

# D: small Gaussian seed of dying deer (CWD outbreak origin)
U0.sub(2).interpolate(
    lambda x: 0.01 * np.exp(-((x[0] - 162)**2 + (x[1] - 139)**2) / 500_000)
)

# E: no prions in environment at t=0
U0.sub(3).interpolate(lambda x: np.full(x.shape[1], 0.0))

U0.x.scatter_forward()

# Forward-solve buffer
U1 = fem.Function(P)

############################
# wolf ODE — initial state
#
# W(x,t) = z(t) * f(d̄(C,x)^2)   where f is a Gaussian and
#   d̄(C,x) = sqrt(y^T M(C) y),  M = D(C)^{-1}
#   y = tangent-projected unit direction from C to x  (PDF sec. 8 approx.)
#
# Pack centre ODE (PDF eq. 3):
#   C_t = (c1 / ||W||_L1) * ∫ y * F / (1 + c2*d(C,x)) dx
#   where y = (x_xy - C) / |x_xy - C|  (unit horizontal direction to x)
#         F = W * [p1*(S+I) + p2*D]     (wolf-deer interaction density)
#
# Pack size ODE (PDF eq. 4):
#   z_t = (z / ||W||_L1) * ∫ F dx  -  z * gamma_w
#
# Note: ||W||_L1 = ∫ f dx = denom (assembled below), which normalises both
# integrals so that z retains its meaning as the total pack-size scalar.
############################
C = np.array(C0)   # pack center
z = z0             # initial pack size scaling factor

# Updatable UFL constants for pack centre (updated before each integral assembly)
Cx_const = Constant(domain, float(C[0]))
Cy_const = Constant(domain, float(C[1]))

# Wolf density as a spatial Function (W = z * f, rebuilt from numpy each step)
W_func = Function(V_scal, name="Wolf_Density")

# Auxiliary scalar Functions for wolf ODE integrands
# (updated from U0 each step; pre-compiled forms hold references to these buffers)
S_func = Function(V_scal)    # S component
I_func = Function(V_scal)    # I component
D_func = Function(V_scal)    # D component
N_func = Function(V_scal)    # total deer N = S + I + D

# DOF coordinate arrays for fast numpy Gaussian evaluation
dof_coords = V_scal.tabulate_dof_coordinates()   # shape (n_dofs, 3)
dof_x = dof_coords[:, 0]
dof_y = dof_coords[:, 1]

# ---- Geometry helper: point evaluation of FEM fields at C ----
# Used each step to extract n_C, es_C, Ds_C so we can compute the
# tangent projection and metric distance d̄ at C (PDF sec. 8).
_mesh_tree = bb_tree(domain, domain.topology.dim)

def _eval_at_C(func, C_xy):
    """Evaluate a scalar or vector FEM Function at the horizontal point C_xy.
    Returns a numpy array of shape (func.function_space.dofmap.bs,).
    Falls back to zero if C lies outside the mesh."""
    dist2 = (dof_x - C_xy[0])**2 + (dof_y - C_xy[1])**2
    idx = np.argmin(dist2)
    pt = dof_coords[idx:idx+1, :]
    collisions = compute_collisions_points(_mesh_tree, pt)
    colliding  = compute_colliding_cells(domain, collisions, pt)
    cells = colliding.array
    if cells.size == 0:
        print("No cell containing C found")
        bs = func.function_space.dofmap.index_map_bs
        return np.zeros(bs)
    return func.eval(pt, cells[:1]).flatten()

# Vector FEM Functions to store the surface normal and downhill direction
# at every DOF — we evaluate these pointwise at C each step.
V_vec = fem.functionspace(domain, ("CG", 1, (3,)))
n_func  = Function(V_vec)   # surface normal field
es_func = Function(V_vec)   # downhill tangent direction field

# Scalar FEM Function for the slope-direction diffusivity Ds at every DOF
Ds_func = Function(V_scal)

# ---- Pre-compiled UFL forms for wolf ODE integrals ----
# All quantities that depend on the pack centre C are encoded as
# UFL Constants updated in-place before each assembly.  This keeps all
# four forms pre-compiled while allowing C to move each timestep.

x_ufl = ufl.SpatialCoordinate(domain)

# Raw horizontal offset vector from C to x (3-D; z-component is zero)
dx_c = x_ufl[0] - Cx_const   # x - Cx
dy_c = x_ufl[1] - Cy_const   # y - Cy

# ------------------------------------------------------------------
# Tangent projection of (dx_c, dy_c, 0) onto T_C Γ  (PDF sec. 8)
#
# The PDF defines y as the unit vector in the direction of the
# projection of (x - C) onto the tangent plane at C.  We approximate
# this by projecting with the normal n_C evaluated at C (constant over
# the domain for the purpose of these integrals).
#
# n_C is stored as a UFL Constant (3-component) and updated each step.
# Projection: v_tan = v - (v · n_C) * n_C   where v = (dx_c, dy_c, 0)
# ------------------------------------------------------------------
nCx = Constant(domain, 0.0)   # x-component of n_C  (updated each step)
nCy = Constant(domain, 0.0)   # y-component of n_C
nCz = Constant(domain, 1.0)   # z-component of n_C  (≈1 on flat ground)

raw_vec_x = dx_c
raw_vec_y = dy_c
raw_vec_z = ufl.as_ufl(0.0)

dot_with_nC = raw_vec_x * nCx + raw_vec_y * nCy + raw_vec_z * nCz

# Tangent-projected components of (x - C)
proj_x = raw_vec_x - dot_with_nC * nCx
proj_y = raw_vec_y - dot_with_nC * nCy
proj_z = raw_vec_z - dot_with_nC * nCz

proj_norm = ufl.sqrt(proj_x**2 + proj_y**2 + proj_z**2 + 1e-10)

# Unit tangent direction y (only x,y components needed for C_dot since
# C lives in the horizontal plane, but we carry the full 3-D vector for
# the metric computation below)
yx = proj_x / proj_norm   # x-component of y
yy = proj_y / proj_norm   # y-component of y
yz = proj_z / proj_norm   # z-component of y

# ------------------------------------------------------------------
# Metric-weighted distance d̄(C, x)  (PDF sec. 8)
#
# d̄² = y^T M(C) y   where M = D^{-1} and
#   D(C) = Ds_C * (es_C ⊗ es_C) + Dc_C * (ec_C ⊗ ec_C)
#   D^{-1}(C) = (1/Ds_C)*(es_C⊗es_C) + (1/Dc_C)*(ec_C⊗ec_C)
#
# es_C and ec_C are stored as UFL Constants and updated each step.
# Dc = kappa (isotropic contour diffusivity, a Python scalar — no need
# to evaluate pointwise since it is spatially constant before lcD_scale).
# Note: lcD_scale is a spatially varying scalar that multiplies the whole
# tensor; its value at C cancels in d̄ because it appears in both Ds and Dc.
# ------------------------------------------------------------------
esCx = Constant(domain, 1.0)   # x-component of es at C
esCy = Constant(domain, 0.0)   # y-component of es at C
esCz = Constant(domain, 0.0)   # z-component of es at C

DsC  = Constant(domain, float(kappa))   # Ds evaluated at C  (updated each step)
DcC  = Constant(domain, float(kappa))   # Dc = kappa (constant, but kept as Constant for symmetry)

# Dot products of unit tangent y with the frame vectors at C
y_dot_es = yx * esCx + yy * esCy + yz * esCz
# ec_C = n_C × es_C  (computed each step in numpy; store as Constants)
eCx = Constant(domain, 0.0)
eCy = Constant(domain, 0.0)
eCz = Constant(domain, 1.0)
y_dot_ec = yx * eCx + yy * eCy + yz * eCz

# d̄²(C, x) = |x-C|_2² * (y · es_C)² / Ds_C + |x-C|_2² * (y · ec_C)² / Dc_C)
#           = |x-C|_2² * y^T M(C) y
eucl_sq = dx_c**2 + dy_c**2
dbar_sq = eucl_sq * (y_dot_es**2 / DsC + y_dot_ec**2 / DcC)
dbar    = ufl.sqrt(dbar_sq + 1e-10)

# ------------------------------------------------------------------
# Gaussian f using metric distance d̄  (PDF sec. 8 approximation)
#   f(d̄²) = exp(-d̄² / (2 * sigma_w²))
# ------------------------------------------------------------------
f_gauss = ufl.exp(-dbar_sq / (2.0 * sigma_w**2))

# Wolf-deer interaction density F = W * [p1*(S+I) + p2*D]  (PDF eq. 3/4)
F_interact = W_func * (p1 * (S_func + I_func) + p2 * D_func)

# Distance-sensitivity weight: 1 / (1 + c2 * d̄(C,x))  (PDF eq. 3)
#dist_weight = 1.0 / (1.0 + c2 * dbar)

# Compiled once; Constants and *_func buffers are updated in-place each step.
#
# denom  = ∫ f dx
# num_x  = ∫ yx * F / (1 + c2*d̄) dx   → x-component of C_t numerator
# num_y  = ∫ yy * F / (1 + c2*d̄) dx   → y-component of C_t numerator
# z_grow = ∫ F dx                       → growth numerator for z_t
form_denom  = fem.form(f_gauss * dx)
#form_num_x  = fem.form(F_interact * yx * dist_weight * dx)
#form_num_y  = fem.form(F_interact * yy * dist_weight * dx)
form_num_x  = fem.form(F_interact * yx * dx)
form_num_y  = fem.form(F_interact * yy * dx)
form_z_grow = fem.form(F_interact * dx)

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
# Interpolate surface geometry fields into CG1 vector functions
# so they can be point-evaluated at C each time step.
############################
# Surface normal n — interpolate the UFL CellNormal expression
n_expr  = fem.Expression(n,  V_vec.element.interpolation_points)
n_func.interpolate(n_expr)
n_func.x.scatter_forward()

# Downhill tangent direction es — same UFL expression used in DTens
es_expr = fem.Expression(es, V_vec.element.interpolation_points)
es_func.interpolate(es_expr)
es_func.x.scatter_forward()

# Slope-direction scalar diffusivity Ds (before lcD_scale, since that
# cancels in the metric; keep as the pure terrain-dependent value)
Ds_ufl_expr = fem.Expression(
    kappa * (isotropy + (1 - isotropy) * activation),
    V_scal.element.interpolation_points
)
Ds_func.interpolate(Ds_ufl_expr)
Ds_func.x.scatter_forward()

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
        - p1    * W_func * uS0       # wolf predation on susceptible
    )

    # ---- I: direct + environmental inflow − clinical progression − wolf predation
    + uI0 * vI + dt * vI * (
          beta  * uS0 * uD0          # direct transmission
        + rho   * uE0 * uS0          # environmental transmission
        - sigma * uI0                # progression to dying stage
        - p1    * W_func * uI0       # wolf predation on sub-clinical infected
    )

    # ---- D: progression from I − natural removal − selective wolf predation
    + uD0 * vD + dt * vD * (
          sigma * uI0                # inflow from infected
        - alpha * uD0                # natural death / removal
        - p2    * W_func * uD0       # wolf predation (higher rate — symptomatic deer easier prey)
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
xdmf_N = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/CWD_N.xdmf", "w")
xdmf_W = io.XDMFFile(domain.comm, "/home/flanagg/SIRbase/CWD_W.xdmf", "w")

xdmf_S.write_mesh(domain)
xdmf_I.write_mesh(domain)
xdmf_D.write_mesh(domain)
xdmf_E.write_mesh(domain)
xdmf_N.write_mesh(domain)
xdmf_W.write_mesh(domain)

uS_out = Function(V_scal, name="Susceptible")
uI_out = Function(V_scal, name="Infected")
uD_out = Function(V_scal, name="Dying")
uE_out = Function(V_scal, name="Environment")
uN_out = Function(V_scal, name="Total Population")
uW_out = Function(V_scal, name="WolfDensity")

def write_outputs(t):
    uS_out.interpolate(U0.sub(0))
    uI_out.interpolate(U0.sub(1))
    uD_out.interpolate(U0.sub(2))
    uE_out.interpolate(U0.sub(3))
    uN_out.interpolate(N_func)
    uW_out.x.array[:] = W_func.x.array
    uW_out.x.scatter_forward()
    xdmf_S.write_function(uS_out, t)
    xdmf_I.write_function(uI_out, t)
    xdmf_D.write_function(uD_out, t)
    xdmf_E.write_function(uE_out, t)
    xdmf_N.write_function(uN_out, t)
    xdmf_W.write_function(uW_out, t)

S_func.x.array[:] = U0.x.array[dof_map_S]
I_func.x.array[:] = U0.x.array[dof_map_I]
D_func.x.array[:] = U0.x.array[dof_map_D]
N_func.x.array[:] = S_func.x.array + I_func.x.array + D_func.x.array
S_func.x.scatter_forward()
I_func.x.scatter_forward()
D_func.x.scatter_forward()
N_func.x.scatter_forward()

def _compute_dbar2(C_xy):
    """Compute d̄²(C, x) at every DOF using the metric M(C) = D(C)^{-1}.
    Returns a numpy array of shape (n_dofs,)."""
    n_C  = _eval_at_C(n_func,  C_xy)
    es_C = _eval_at_C(es_func, C_xy)
    Ds_C = float(_eval_at_C(Ds_func, C_xy)[0])
    n_C  = n_C  / (np.linalg.norm(n_C)  + 1e-14)
    es_C = es_C / (np.linalg.norm(es_C) + 1e-14)
    ec_C = np.cross(n_C, es_C)
    ec_C = ec_C / (np.linalg.norm(ec_C) + 1e-14)
    Dc_C = float(kappa)

    # Raw offset in 3-D (z-component zero — C lives in horizontal plane)
    raw = np.stack([dof_x - C_xy[0], dof_y - C_xy[1],
                    np.zeros(len(dof_x))], axis=1)   # (n, 3)

    # Tangent projection onto T_C Γ
    dot_n = raw @ n_C                                # (n,)
    tan   = raw - np.outer(dot_n, n_C)              # (n, 3)
    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True) + 1e-10
    y = tan / tan_norm                               # unit tangent direction (n, 3)

    # Euclidean distance squared |x - C|_2^2
    dist2_eucl = (dof_x - C_xy[0])**2 + (dof_y - C_xy[1])**2   # (n,)

    # Metric components: d̄² = |x-C|_2² * y^T M(C) y
    y_es = y @ es_C   # (n,)
    y_ec = y @ ec_C   # (n,)
    return dist2_eucl * (y_es**2 / max(Ds_C, 1e-6) + y_ec**2 / Dc_C)   # d̄²  (n,)

# t = 0 snapshot
dbar2_0 = _compute_dbar2(C)
W_func.x.array[:] = z * np.exp(-dbar2_0 / (2.0 * sigma_w**2))
W_func.x.scatter_forward()
write_outputs(0.0)

# Wolf state log for post-processing / adjoint
wolf_log = []   # list of (t, Cx, Cy, z)

xmin, xmax = dof_x.min(), dof_x.max()
ymin, ymax = dof_y.min(), dof_y.max()
print(f"Domain x bounds: [{xmin:.0f},{xmax:.0f}]")
print(f"Domain y bounds: [{ymin:.0f},{ymax:.0f}]")

############################
# main time loop
############################
for i in range(int(T / dt)):

    # ----------------------------------------
    # 0. Sync S,I,D,N from U0 = timestep n
    # ----------------------------------------
    S_func.x.array[:] = U0.x.array[dof_map_S]
    I_func.x.array[:] = U0.x.array[dof_map_I]
    D_func.x.array[:] = U0.x.array[dof_map_D]
    N_func.x.array[:] = S_func.x.array + I_func.x.array + D_func.x.array
    S_func.x.scatter_forward()
    I_func.x.scatter_forward()
    D_func.x.scatter_forward()
    N_func.x.scatter_forward()

    # ----------------------------------------
    # 1. Build Wolf Function  W = z * f(d̄(C,x)^2)
    #    using metric distance d̄ with M(C) = D(C)^{-1}
    # ----------------------------------------
    dbar2 = _compute_dbar2(C)
    W_func.x.array[:] = z * np.exp(-dbar2 / (2.0 * sigma_w**2))
    W_func.x.scatter_forward()

    # ----------------------------------------
    # 2. Update C and z  (PDF equations 3 & 4)
    #
    # Before assembling the UFL forms we update all Constants that
    # encode geometry at the current pack centre C:
    #   n_C    — surface normal at C  (for tangent projection of y)
    #   es_C   — downhill direction at C  (for metric d̄)
    #   ec_C   — contour direction at C = n_C × es_C
    #   Ds_C   — slope diffusivity at C  (for metric d̄)
    #
    # denom = ||W||_L1 = ∫ f(d̄²) dx  (shared normalisation)
    #
    # C_t = (wolf_v / (z* denom)) * ∫ y * F / (1 + c2*d̄) dx   (PDF eq. 3)
    # z_t = (1 / denom)      * ∫ F dx  -  z * gamma_w     (PDF eq. 4)
    # ----------------------------------------
    Cx_const.value = C[0]
    Cy_const.value = C[1]

    # --- Evaluate geometry at C ---
    n_C_val  = _eval_at_C(n_func,  C)   # shape (3,)
    es_C_val = _eval_at_C(es_func, C)   # shape (3,)
    Ds_C_val = float(_eval_at_C(Ds_func, C)[0])

    # Renormalise in case interpolation introduced small errors
    n_C_val  = n_C_val  / (np.linalg.norm(n_C_val)  + 1e-14)
    es_C_val = es_C_val / (np.linalg.norm(es_C_val) + 1e-14)

    # Contour direction ec_C = n_C × es_C
    ec_C_val = np.cross(n_C_val, es_C_val)
    ec_C_val = ec_C_val / (np.linalg.norm(ec_C_val) + 1e-14)

    # Push geometry into UFL Constants
    nCx.value  = n_C_val[0];  nCy.value  = n_C_val[1];  nCz.value  = n_C_val[2]
    esCx.value = es_C_val[0]; esCy.value = es_C_val[1]; esCz.value = es_C_val[2]
    eCx.value  = ec_C_val[0]; eCy.value  = ec_C_val[1]; eCz.value  = ec_C_val[2]
    DsC.value  = max(Ds_C_val, 1e-6)
    DcC.value  = float(kappa)   # Dc is spatially constant

    denom = max(fem.assemble_scalar(form_denom), 1e-10)

    # Pack centre velocity: wolf_v/||W||_L1 * ∫ y * F/(1+c2*d̄) dx  (PDF eq. 3)
    C_dot = np.array([
        fem.assemble_scalar(form_num_x) / denom,
        fem.assemble_scalar(form_num_y) / denom,
    ])
    C = C + dt * wolf_v * C_dot / z
    C[0] = np.clip(C[0], xmin, xmax)
    C[1] = np.clip(C[1], ymin, ymax)

    # Pack size ODE: z_t = (1/ ∫ f dx) * ∫ F dx  -  z * gamma_w  (PDF eq. 4)
    z_growth = (1 / denom) * fem.assemble_scalar(form_z_grow) - z * gamma_w
    z = z + dt * z_growth
    z = max(z, 0.0)

    # ----------------------------------------
    # 3. Solve PDE
    # ----------------------------------------
    L = assemble_vector(l_form)
    L.assemble()

    solver.solve(L, U1.x.petsc_vec)

    U0.x.array[:] = U1.x.array
    U0.x.scatter_forward()

    # ------------------------------------------------------------------
    # 6. Periodic output and progress
    # -----------------------------------------------------------------
    if i==0:
        print(fem.assemble_scalar(form_z_grow))
        print(W_func.x.array.sum())
        print(n_C_val)
        print(es_C_val)

    if i % 10 == 0:
        write_outputs((i + 1) * dt)
        wolf_log.append(((i+1) * dt, C[0], C[1], z))
        print(f"t = {(i+1)*dt:.2f}/{T}   C = ({C[0]:.1f}, {C[1]:.1f})   z = {z:.4f}")

xdmf_S.close()
xdmf_I.close()
xdmf_D.close()
xdmf_E.close()
xdmf_N.close()
xdmf_W.close()

# Save wolf trajectory for analysis / adjoint post-processing
np.save("wolf_trajectory.npy", np.array(wolf_log))
print("Done. Wolf trajectory saved to wolf_trajectory.npy")