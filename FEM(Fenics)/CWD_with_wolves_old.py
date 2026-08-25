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
T        = 15      # years
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

# Wolf ODE parameters  (PDF sec. 5, eqs. 1–2)
# Wolf density: W(x) = z * p(d_{M̃}(C,x))  where p is a Gaussian
# Pack centre ODE:  C_dot = Phi(max{0, A(h*)}) * h*   (eq. 1)
# Pack size ODE:    z_dot = F(C) - z*d_w              (eq. 2)
sigma_w = 50.0    # Gaussian half-width for p(t) = exp(-t^2/(2*sigma_w^2))
r_avg   = 150.0   # radius used to average M over B_r(C) to get M̃_C  (same units as mesh)
p1      = 7.0     # predation rate on S and I (non-symptomatic deer)
p2      = 50.0    # predation rate on D/R (symptomatic; easier prey)
d_w     = 3.0     # wolf pack mortality / emigration rate  (d in eq. 2)
lam     = 0.5     # lambda: weight on movement cost J̃ in A(h)  (PDF sec. 5.1)
phi_c   = 10  # saturation speed scalar in Phi(t) = phi_c * t / (1 + phi_k*t)
phi_k   = 0.0005   # saturation parameter in Phi
n_hgrid = 36      # number of directions on the half unit circle to evaluate A(h)
C0 = [3000, 3000]
z0 = 1.0

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
# wolf ODE — initial state  (PDF sec. 5, eqs. 1–2)
#
# W̃(x) = z * p(d_{M̃_C}(C, x))   where p(t) = exp(-t²/(2*sigma_w²))
#
#   d_{M̃_C}(C, x) = sqrt((x-C)^T M̃_C (x-C))   (approx. geodesic distance,
#   working in 2D horizontal coordinates via M = N^T D^{-1} N, PDF sec. 5.1)
#
#   M̃_C = average of M over B_{r_avg}(C)  (PDF sec. 5.4)
#
# The gradient in 2D horizontal coords is:
#   ∇W̃_C(x) = z * p'(d) / d * M̃_C * (x - C)   where d = d_{M̃_C}(C, x)
#
# Pack centre ODE  (PDF eq. 1):
#   C_dot = Phi(max{0, A(h*)}) * h*
#   where h* = argmax_{|h|=1} A_e(h)
#         A_e(h) = D_h F̃ - (λ / ||W̃||_L1) * J̃(C, h)
#
# Pack size ODE  (PDF eq. 2):
#   z_dot = F̃(C) - z * d_w
#   F̃(C) = ∫ W̃ * K dx,   K = p1*(S+I) - p2*R  (reward kernel)
#
# h* search strategy (user spec.):
#   - Evaluate D_h F̃ on a grid of the half unit circle
#   - If D_h F̃ < 0, flip to opposite direction
#   - Evaluate J̃ at the resulting directions, pick best A_e(h)
############################
C = np.array(C0, dtype=float)   # pack centre in horizontal (x,y) coords
z = float(z0)                   # pack size scalar

# Wolf density FEM Function (rebuilt from numpy each step)
W_func = Function(V_scal, name="Wolf_Density")

# Auxiliary scalar Functions — updated each step, referenced in the UFL l-form
S_func = Function(V_scal)
I_func = Function(V_scal)
D_func = Function(V_scal)
N_func = Function(V_scal)

# DOF coordinate arrays for fast numpy computation of W̃ and its gradient
dof_coords = V_scal.tabulate_dof_coordinates()   # (n_dofs, 3)
dof_x = dof_coords[:, 0]
dof_y = dof_coords[:, 1]

# ---- Geometry: FEM Functions holding surface normal and downhill direction ----
_mesh_tree = bb_tree(domain, domain.topology.dim)

def _eval_at_point(func, pt_xy):
    """Evaluate a scalar or vector FEM Function at a horizontal point (x,y).
    Returns a 1-D numpy array; falls back to zero if outside the mesh."""
    dist2 = (dof_x - pt_xy[0])**2 + (dof_y - pt_xy[1])**2
    idx   = np.argmin(dist2)
    pt3   = dof_coords[idx:idx+1, :]
    cols  = compute_collisions_points(_mesh_tree, pt3)
    cells = compute_colliding_cells(domain, cols, pt3).array
    if cells.size == 0:
        bs = func.function_space.dofmap.index_map_bs
        return np.zeros(bs)
    return func.eval(pt3, cells[:1]).flatten()

V_vec   = fem.functionspace(domain, ("CG", 1, (3,)))
n_func  = Function(V_vec)    # surface normal field n(x)
es_func = Function(V_vec)    # downhill tangent direction es(x)
Ds_func = Function(V_scal)   # slope diffusivity Ds(x) (terrain-only, before lcD_scale)

# ---- Numpy helpers for the new wolf model ----

def _build_M_at_dofs():
    """Compute the 2-D metric tensor M(x) = N(x)^T D^{-1}(x) N(x) at every DOF.

    N = [[1, 0], [0, 1], [h_x, h_y]]  maps 2D vectors to the embedded surface.
    D^{-1} = (1/Ds)*es⊗es + (1/Dc)*ec⊗ec  (the 3×3 anisotropic inverse diffusion tensor).

    Returns M_dofs of shape (n_dofs, 2, 2).
    """
    n_vals  = n_func.x.array.reshape(-1, 3)    # (n, 3) surface normals
    es_vals = es_func.x.array.reshape(-1, 3)   # (n, 3) downhill directions
    Ds_vals = Ds_func.x.array                  # (n,)   slope diffusivities
    Dc_val  = float(kappa)

    # ec = n × es (contour direction)
    ec_vals = np.cross(n_vals, es_vals)
    # Normalise (small errors from interpolation)
    for arr in (n_vals, es_vals, ec_vals):
        nrm = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(nrm, 1e-14)

    # D^{-1} in 3D: inv_D_ij = (1/Ds)*es_i*es_j + (1/Dc)*ec_i*ec_j
    Ds_safe = np.maximum(Ds_vals, 1e-6)        # (n,)
    # We need N^T (inv_D) N where N = [[1,0],[0,1],[hx,hy]]
    # Equivalent: for each DOF extract the 2×2 block of inv_D corresponding
    # to the (x,y) rows/cols, plus cross terms with the z row weighted by hx,hy.
    # Since N is defined implicitly through the surface parametrisation, and we
    # only need M acting on 2D horizontal vectors, we project:
    #   (M v)_i = v^T M v = (Nv)^T D^{-1} (Nv)   for v in R^2
    # We compute the 2×2 matrix M = N^T D^{-1} N directly by:
    #   M_ij = e_i^T (N^T D^{-1} N) e_j   where e_1=(1,0,0), e_2=(0,1,0) in 3D restricted to xy
    # More simply: for a 2D vector v=(v1,v2), Nv = (v1, v2, hx*v1+hy*v2).
    # But we don't have hx,hy directly; instead we use the fact that the tangent
    # projection of (1,0,0) and (0,1,0) onto TΓ gives us the rows of N.
    # Tangent component of e1=(1,0,0): e1 - (e1·n)*n
    e1 = np.zeros_like(n_vals); e1[:, 0] = 1.0
    e2 = np.zeros_like(n_vals); e2[:, 1] = 1.0
    Ne1 = e1 - (np.sum(e1 * n_vals, axis=1, keepdims=True)) * n_vals  # (n,3)
    Ne2 = e2 - (np.sum(e2 * n_vals, axis=1, keepdims=True)) * n_vals  # (n,3)
    # Apply D^{-1} to Ne1 and Ne2:
    #   D^{-1} v = (1/Ds)*(es·v)*es + (1/Dc)*(ec·v)*ec
    def apply_Dinv(v):
        a = np.sum(es_vals * v, axis=1)   # (n,)  es·v
        b = np.sum(ec_vals * v, axis=1)   # (n,)  ec·v
        return (a / Ds_safe)[:, None] * es_vals + (b / Dc_val)[:, None] * ec_vals   # (n,3)
    DinvNe1 = apply_Dinv(Ne1)   # (n,3)
    DinvNe2 = apply_Dinv(Ne2)   # (n,3)
    # M_ij = Ne_i · D^{-1} Ne_j
    M11 = np.sum(Ne1 * DinvNe1, axis=1)   # (n,)
    M12 = np.sum(Ne1 * DinvNe2, axis=1)   # (n,)
    M22 = np.sum(Ne2 * DinvNe2, axis=1)   # (n,)
    M_dofs = np.stack([np.stack([M11, M12], axis=1),
                       np.stack([M12, M22], axis=1)], axis=1)   # (n,2,2)
    return M_dofs


def _compute_Mtilde(C_xy, M_dofs):
    """Average M over DOFs within B_{r_avg}(C) to get M̃_C (2×2 numpy array)."""
    dist2 = (dof_x - C_xy[0])**2 + (dof_y - C_xy[1])**2
    mask  = dist2 < r_avg**2
    if mask.sum() == 0:
        mask = np.array([np.argmin(dist2)])   # fallback: nearest DOF only
    return M_dofs[mask].mean(axis=0)   # (2,2)


def _Mtilde_directional_derivative(C_xy, h2d, M_dofs):
    """Approximate D^C_h M̃_C = ∫_{B_r(C)} D_h M(y) dy  (PDF sec. 5.4).

    We use a finite-difference approximation:
        D^C_h M̃_C ≈ (M̃_{C+eps*h} - M̃_{C-eps*h}) / (2*eps)
    where eps is a small step in the horizontal plane.
    """
    eps = r_avg * 0.1
    h_norm = h2d / (np.linalg.norm(h2d) + 1e-14)
    Mfwd = _compute_Mtilde(C_xy + eps * h_norm, M_dofs)
    Mbwd = _compute_Mtilde(C_xy - eps * h_norm, M_dofs)
    return (Mfwd - Mbwd) / (2.0 * eps)


def _compute_wolf_distribution(C_xy, z_val, Mf):
    """Compute W̃(x) = z * p(d_{M̃}(C,x)) at every DOF.

    d_{M̃}(C,x) = sqrt((x-C)^T M̃ (x-C))
    p(t) = exp(-t²/(2*sigma_w²))

    Returns (W_arr, d_arr) each of shape (n_dofs,).
    """
    r = np.stack([dof_x - C_xy[0], dof_y - C_xy[1]], axis=1)   # (n,2)
    # d² = r^T M̃ r
    Mfr   = r @ Mf.T      # (n,2)
    d2    = np.sum(r * Mfr, axis=1)   # (n,)
    d2    = np.maximum(d2, 0.0)
    d_arr = np.sqrt(d2)
    W_arr = z_val * np.exp(-d2 / (2.0 * sigma_w**2))
    return W_arr, d_arr


def _compute_gradW(C_xy, z_val, Mf, d_arr):
    """Compute ∇W̃_C(x) in 2D at every DOF.

    ∇W̃_C = z * p'(d) / d * M̃ * (x - C)
    p'(d) = -d/sigma_w² * p(d),  so:
    ∇W̃_C = -z / sigma_w² * exp(-d²/(2*sigma_w²)) * M̃ * (x - C)

    Returns gradW of shape (n_dofs, 2).
    """
    r   = np.stack([dof_x - C_xy[0], dof_y - C_xy[1]], axis=1)   # (n,2)
    d2  = np.maximum(d_arr**2, 0.0)
    p   = np.exp(-d2 / (2.0 * sigma_w**2))                         # (n,)
    Mfr = r @ Mf.T                                                  # (n,2)
    gradW = -(z_val / sigma_w**2) * p[:, None] * Mfr               # (n,2)
    return gradW


def _compute_Ph(h2d, C_xy, Mf, M_dofs):
    """Compute the metric-curvature correction P_h(x) = ½ M̃⁻¹ D^C_h M̃ M̃⁻¹(C-x).

    This is the first-order correction from the fact that M̃ depends on C.
    (PDF sec. 5.4, formula below eq. for W̃_{C+ht})

    Returns Ph of shape (n_dofs, 2).
    """
    Mfinv = np.linalg.inv(Mf)                                      # (2,2)
    DhMf  = _Mtilde_directional_derivative(C_xy, h2d, M_dofs)      # (2,2)
    # ½ M̃⁻¹ D^C_h M̃   (PDF sec. 5.4)
    half_A = 0.5 * Mfinv @ DhMf                                    # (2,2)
    r_C    = np.stack([C_xy[0] - dof_x, C_xy[1] - dof_y], axis=1) # (n,2) = C - x
    Ph     = r_C @ half_A.T                                         # (n,2)
    return Ph


def _eval_A(h2d, C_xy, z_val, Mf, M_dofs,
            gradW, W_arr, S_arr, I_arr, D_arr, L1_W):
    """Evaluate Ã(h) for a single direction h2d (2D unit vector).

    A_e(h) = D_h F̃ - (λ / ||W̃||_L1) * J̃(C, h)

    D_h F̃ = -∫ [(h + P_h) · ∇W̃_C] * K dx        (needs area_per_dof)
    J̃(C,h) = ∫ |(h + P_h) · ∇W̃_C| / |∇W̃_C|² * W̃_C * sqrt(∇W̃_C^T M̃ ∇W̃_C) dx

    K = p1*(S+I) + p2*R   (wolves benefit from eating all deer; page 7 of PDF)

    Integral approximation:
      DhF  ≈ sum(f_i) * area_per_dof   — area factor needed, does not cancel
      Je   ≈ sum(j_i)                  — area appears in both J and ||W||_L1,
      L1_W ≈ sum(W_i)                    so it cancels in (lam/L1_W)*Je
    """
    Ph  = _compute_Ph(h2d, C_xy, Mf, M_dofs)           # (n,2)
    h_plus_Ph = h2d[None, :] + Ph                       # (n,2) broadcast

    # (h + P_h) · ∇W̃_C at every DOF
    hph_dot_gradW = np.sum(h_plus_Ph * gradW, axis=1)   # (n,)

    # K(x) = p1*(S+I) + p2*R   (correct sign: all predation is rewarding)
    K_arr = p1 * (S_arr + I_arr) + p2 * D_arr           # (n,)

    # DhF needs area_per_dof to approximate the integral correctly
    DhF = -np.sum(hph_dot_gradW * K_arr) * area_per_dof

    # J̃ integrand: |(h+Ph)·∇W| / |∇W|² * W * sqrt(∇W^T M̃ ∇W)
    # area_per_dof cancels between Je and L1_W, so plain sums suffice
    gradW_sq     = np.sum(gradW**2, axis=1) + 1e-30     # |∇W̃|²   (n,)
    MfgradW      = gradW @ Mf.T                          # M̃ ∇W̃   (n,2)
    gradW_M_norm = np.sqrt(np.sum(gradW * MfgradW, axis=1) + 1e-30)  # (n,)
    J_integrand  = (np.abs(hph_dot_gradW) / gradW_sq
                    * W_arr * gradW_M_norm)               # (n,)
    Je = np.sum(J_integrand)

    L1_safe = max(L1_W, 1e-10)
    return DhF - (lam / L1_safe) * Je, DhF, Je


def _find_hstar(C_xy, z_val, Mf, M_dofs,
                S_arr, I_arr, D_arr):
    """Find h* = argmax_{|h|=1} A_e(h) using the half-circle grid search.

    Strategy (user spec.):
      1. Evaluate D_h F̃ on a uniform grid of the HALF unit circle (n_hgrid angles
         in [0, π)).
      2. For each direction where D_h F̃ < 0, flip to the opposite direction.
         This ensures we always consider the better of (h, -h) for the F component.
      3. Evaluate full A_e(h) at the resulting candidate directions.
      4. Return the direction with the highest A_e value.

    Returns (h_star_2d, A_star, DhF_star) where h_star_2d is a 2D unit vector.
    """
    W_arr, d_arr = _compute_wolf_distribution(C_xy, z_val, Mf)
    gradW        = _compute_gradW(C_xy, z_val, Mf, d_arr)
    L1_W         = np.sum(W_arr)   # plain sum; area_per_dof cancels in (lam/L1_W)*Je

    angles_half  = np.linspace(0.0, np.pi, n_hgrid, endpoint=False)
    h_candidates = np.stack([np.cos(angles_half),
                              np.sin(angles_half)], axis=1)   # (n_hgrid, 2)

    # Step 1: evaluate D_h F̃ on half circle
    # (cheap: only the linear term, no J̃ yet)
    DhF_vals = np.empty(n_hgrid)
    for k, h2d in enumerate(h_candidates):
        Ph            = _compute_Ph(h2d, C_xy, Mf, M_dofs)
        hph_dot_gradW = np.sum((h2d[None, :] + Ph) * gradW, axis=1)
        K_arr         = p1 * (S_arr + I_arr) + p2 * D_arr   # correct sign
        DhF_vals[k]   = -np.sum(hph_dot_gradW * K_arr) * area_per_dof

    # Step 2: flip any direction where D_h F̃ < 0
    for k in range(n_hgrid):
        if DhF_vals[k] < 0.0:
            h_candidates[k] = -h_candidates[k]   # use opposite side
            DhF_vals[k]     = -DhF_vals[k]       # D_{-h}F = -D_h F (linearity)

    # Step 3: evaluate full A_e on all candidate directions
    A_vals = np.empty(n_hgrid)
    for k, h2d in enumerate(h_candidates):
        A_vals[k], _, _ = _eval_A(h2d, C_xy, z_val, Mf, M_dofs,
                                  gradW, W_arr, S_arr, I_arr, D_arr, L1_W)

    # Step 4: pick best
    best   = np.argmax(A_vals)
    h_star = h_candidates[best]
    h_star = h_star / (np.linalg.norm(h_star) + 1e-14)   # ensure unit length
    return h_star, A_vals[best], DhF_vals[best], W_arr, d_arr


def _phi(t):
    """Saturation function Phi(t) = phi_c * t / (1 + phi_k * t)."""
    return phi_c * t / (1.0 + phi_k * t)

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

# Pre-compute the 2×2 metric tensor M(x) at every DOF (section 5.1).
# This is done once after geometry fields are interpolated and reused
# every step inside _compute_Mtilde and _build_M_at_dofs.
M_dofs = _build_M_at_dofs()   # (n_dofs, 2, 2)

# t = 0 snapshot — build W̃ using M̃_C averaged over B_{r_avg}(C)
Mf_0       = _compute_Mtilde(C, M_dofs)
W0_arr, _  = _compute_wolf_distribution(C, z, Mf_0)
W_func.x.array[:] = W0_arr
W_func.x.scatter_forward()
write_outputs(0.0)

# Wolf state log for post-processing / adjoint
wolf_log = []   # list of (t, Cx, Cy, z)

xmin, xmax = dof_x.min(), dof_x.max()
ymin, ymax = dof_y.min(), dof_y.max()
print(f"Domain x bounds: [{xmin:.0f},{xmax:.0f}]")
print(f"Domain y bounds: [{ymin:.0f},{ymax:.0f}]")

# Approximate area per DOF: used to turn DOF sums into integral approximations.
# DhF = ∫ f dx ≈ sum(f_i) * area_per_dof.  The J term (lam/L1_W)*Je uses
# sum(j)/sum(W), where area_per_dof cancels, so those stay as plain sums.
area_per_dof = (xmax - xmin) * (ymax - ymin) / len(dof_x)

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
    # 1. Build W̃(x) = z * p(d_{M̃_C}(C, x))  (PDF sec. 5.4 flat-metric approx.)
    #    M̃_C = average of M(x) over B_{r_avg}(C)
    # ----------------------------------------
    Mf       = _compute_Mtilde(C, M_dofs)
    W_arr, _ = _compute_wolf_distribution(C, z, Mf)
    W_func.x.array[:] = W_arr
    W_func.x.scatter_forward()

    # ----------------------------------------
    # 2. Update C and z  (PDF eqs. 1–2, sec. 5.1 and 5.4)
    #
    # Step 2a — find h* = argmax A_e(h) via half-circle grid search
    #   A_e(h) = D_h F̃ - (λ/||W̃||_L1) * J̃(C, h)
    #
    # Step 2b — pack centre ODE (eq. 1):
    #   C_dot = Phi(max{0, A_e(h*)}) * h*
    #   Phi(t) = phi_c * t / (1 + phi_k * t)
    #
    # Step 2c — pack size ODE (eq. 2):
    #   z_dot = F̃(C) - z * d_w
    #   F̃(C) = ∫ W̃ * K dx,  K = p1*(S+I) - p2*R
    # ----------------------------------------

    # Current deer arrays (already synced above)
    S_arr = S_func.x.array
    I_arr = I_func.x.array
    D_arr = D_func.x.array   # "D" compartment == "R" (dying/removed) in PDF notation

    # Step 2a: find h*
    h_star, A_star, DhF_star, W_arr_h, d_arr_h = _find_hstar(
        C, z, Mf, M_dofs, S_arr, I_arr, D_arr
    )

    # Step 2b: pack centre update
    speed  = _phi(max(0.0, A_star))
    C_dot  = speed * h_star          # 2D horizontal velocity
    C_new  = C + dt * C_dot
    C_new[0] = np.clip(C_new[0], xmin, xmax)
    C_new[1] = np.clip(C_new[1], ymin, ymax)
    C = C_new

    # Step 2c: pack size ODE  z_dot = F̃(C) - z * d_w
    K_arr    = p1 * (S_arr + I_arr) + p2 * D_arr      # reward kernel (correct sign)
    F_tilde  = np.sum(W_arr_h * K_arr) * area_per_dof # ≈ ∫ W̃ K dx
    L1_W_z   = max(np.sum(W_arr_h) * area_per_dof, 1e-10)  # ≈ ||W̃||_L1
    z_dot    = F_tilde / L1_W_z - z * d_w
    z        = max(z + dt * z_dot, 0.0)

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
    if i == 0:
        print(f"Step 0 debug — F̃(C)={F_tilde:.4f}  A*={A_star:.4f}  h*={h_star}  z={z:.4f}")
        print(f"W̃ total (proxy L1) = {np.mean(W_arr_h):.6f}")

    if i % 5 == 0:
        write_outputs((i + 1) * dt)
        wolf_log.append(((i + 1) * dt, C[0], C[1], z))
        print(f"t = {(i+1)*dt:.2f}/{T}   C = ({C[0]:.1f}, {C[1]:.1f})   z = {z:.4f}"
              f"   A* = {A_star:.4f}   h* = ({h_star[0]:.3f}, {h_star[1]:.3f})")

xdmf_S.close()
xdmf_I.close()
xdmf_D.close()
xdmf_E.close()
xdmf_N.close()
xdmf_W.close()

# Save wolf trajectory for analysis / adjoint post-processing
np.save("wolf_trajectory.npy", np.array(wolf_log))
print("Done. Wolf trajectory saved to wolf_trajectory.npy")