"""Uncontrolled CWD reaction-diffusion solver on a terrain surface mesh.

Advances the four-compartment state system

    S  susceptible deer
    I  infected, sub-clinical, non-shedding deer
    D  prion-shedding deer, pre-clinical through clinical  (R in the write-up)
    E  environmental prion contamination                   (W in the write-up)

with the lumped-mass semi-implicit (IMEX) Euler step described in the write-up:
diffusion implicit against a factorization built once, reactions explicit at the
previous time level. This is the v = 0 reference trajectory; the controlled
problem lives in CWD_optimal_control.py and shares this module's mesh bundle,
parameter file, disease-free spin-up, and block solver.

Example
-------
    python CWD_solver.py --mesh-folder outputs --output-folder solver_outputs
"""

############################
# imports
############################
import argparse
import sys
from pathlib import Path

import numpy as np
import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.io.gmsh import read_from_msh
from ufl import (
    CellNormal, Identity, TestFunction,
    as_vector, dot, dx, exp, outer, split,
)

WORKFLOW_ROOT = Path(__file__).resolve().parent
UTILITIES_DIR = WORKFLOW_ROOT / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from block_solver import CompartmentBlockSolver  # noqa: E402
from disease_spinup import (  # noqa: E402
    default_disease_paths,
    disease_initial_state,
    disease_signature,
    run_disease_spinup,
)
from mesh_bundle import add_bundle_arguments, resolve_input_files  # noqa: E402
from shared_parameters import (  # noqa: E402
    land_cover_to_carrying_capacity,
    load_parameters,
)
from susceptible_spinup import (  # noqa: E402
    default_equilibrium_paths,
    susceptible_initial_condition,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the uncontrolled CWD reaction-diffusion solver on a mesh "
            "bundle produced by generate_mesh_and_attributes.py."
        )
    )
    add_bundle_arguments(parser)
    parser.add_argument(
        "--output-folder",
        default="solver_outputs",
        help="Folder for solver XDMF/HDF5 results (default: solver_outputs).",
    )
    return parser

args = build_parser().parse_args()
mesh_path, classes_path, diffusivity_path, spatial_parameter_path = resolve_input_files(args)
spatial_parameters = load_parameters(spatial_parameter_path)
output_folder = Path(args.output_folder).expanduser().resolve()
if MPI.COMM_WORLD.rank == 0:
    output_folder.mkdir(parents=True, exist_ok=True)
MPI.COMM_WORLD.barrier()

# Cached disease-free susceptible field. It belongs to the mesh bundle rather
# than to any one run, so it lives beside the mesh by default and is shared by
# every simulation on that bundle.
equilibrium_array_path, equilibrium_signature_path = default_equilibrium_paths(
    mesh_path
)
if args.susceptible_equilibrium is not None:
    equilibrium_array_path = Path(args.susceptible_equilibrium).expanduser().resolve()
    equilibrium_signature_path = equilibrium_array_path.with_suffix(".json")

tensor_parameters = spatial_parameters["diffusion_tensor"]
time_parameters = spatial_parameters["time"]
kinetic_parameters = spatial_parameters["reaction"]
initial_condition_parameters = spatial_parameters["initial_conditions"]

if MPI.COMM_WORLD.rank == 0:
    print(f"Mesh: {mesh_path}")
    print(f"Land-cover classes: {classes_path}")
    print(f"Land-cover diffusivity: {diffusivity_path}")
    print(f"Model parameters: {spatial_parameter_path}")
    print(f"Susceptible equilibrium: {equilibrium_array_path}")
    print(f"Solver outputs: {output_folder}")

############################
# model and time parameters
############################
T = float(time_parameters["final_time_years"])
dt = float(time_parameters["time_step_years"])

isotropy = float(tensor_parameters["isotropy"])
kappa = float(tensor_parameters["kappa"])
DS = float(tensor_parameters["compartment_scales"]["susceptible"])
DI = float(tensor_parameters["compartment_scales"]["infected"])
DD = float(tensor_parameters["compartment_scales"]["dying"])
activation_steepness = float(tensor_parameters["activation_steepness"])
activation_cosine_threshold = float(
    tensor_parameters["activation_cosine_threshold"]
)

alpha = float(kinetic_parameters["clinical_removal_rate"])
sigma = float(kinetic_parameters["infection_to_clinical_rate"])
beta = float(kinetic_parameters["direct_transmission_rate"])
r = float(kinetic_parameters["intrinsic_growth_rate"])
p_env = float(kinetic_parameters["environmental_shedding_rate"])
delta_e = float(kinetic_parameters["environmental_decay_rate"])
rho = float(kinetic_parameters["environmental_transmission_rate"])

############################
# domain and function spaces
############################
mesh_data = read_from_msh(str(mesh_path), MPI.COMM_WORLD, gdim=3)
domain    = mesh_data.mesh

# Single CG1 element
P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)

# Mixed element for the four PDE compartments: S, I, D, E
mixed = basix.ufl.mixed_element([P1, P1, P1, P1])
P     = fem.functionspace(domain, mixed)

# Scalar CG1 space for land-cover data, the initial condition, and output
V_scal = fem.functionspace(domain, ("CG", 1))

# Test functions. There is no TrialFunction here: the bilinear form is built
# and factorized inside CompartmentBlockSolver, and this file assembles only
# the load vector.
V  = TestFunction(P)
vS, vI, vD, vE = split(V)

# Solution at previous time step (symbolic split for UFL RHS)
U0 = fem.Function(P)
uS0, uI0, uD0, uE0 = split(U0)

############################
# land cover, diffusivity, carrying capacity
############################
land_cover_classes      = np.load(classes_path).astype(np.int32)
land_cover_func         = fem.Function(V_scal)
land_cover_func.name    = "Land_Cover_Class"
if land_cover_classes.size != land_cover_func.x.array.size:
    raise ValueError(
        "land_cover_classes.npy has "
        f"{land_cover_classes.size} values, but this mesh/function space expects "
        f"{land_cover_func.x.array.size}. The mesh and arrays are not a matching bundle."
    )
land_cover_func.x.array[:] = land_cover_classes

lcD_values = np.load(diffusivity_path)
lcD_scale  = Function(V_scal)
if lcD_values.size != lcD_scale.x.array.size:
    raise ValueError(
        "land_cover_diffusivity.npy has "
        f"{lcD_values.size} values, but this mesh/function space expects "
        f"{lcD_scale.x.array.size}. The mesh and arrays are not a matching bundle."
    )
lcD_scale.x.array[:] = lcD_values
lcD_scale.x.scatter_forward()

K_values = land_cover_to_carrying_capacity(land_cover_classes, spatial_parameters)
K_func   = Function(V_scal)
K_func.x.array[:] = K_values
K_func.x.scatter_forward()

with io.XDMFFile(domain.comm, str(output_folder / "land_cover_classes.xdmf"), "w") as xdmf_lc:
    xdmf_lc.write_mesh(domain)
    xdmf_lc.write_function(land_cover_func)

############################
# diffusion tensor (terrain-aware anisotropic, unchanged from original)
#
# Built before the initial conditions because the disease-free spin-up that
# produces the initial S field needs it.
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

activation = (
    1 + exp(activation_steepness * (activation_cosine_threshold - 1))
) / (
    1
    + exp(
        activation_steepness
        * (activation_cosine_threshold - abs(cos_theta))
    )
)
Ds    = kappa * (isotropy + (1 - isotropy) * activation)
Dc    = kappa
DTens = lcD_scale * ((Ds - Dc) * outer(es, es) + Dc * (Identity(3) - outer(n, n)))

############################
# initial conditions
############################
# Only the susceptible block is written through its DOF map; I, D, and E are
# set with sub().interpolate(), which needs no collapsed space.
V_S, dof_map_S = P.sub(0).collapse()

# S starts from the DISEASE-FREE EQUILIBRIUM, not from the raw carrying
# capacity: K is a piecewise-constant land-cover lookup, so it is not a steady
# state of the S equation, and starting there lays an order-one demographic
# transient on top of the epidemic. The relaxed field is computed once per mesh
# bundle and cached; see utilities/susceptible_spinup.py.
S_equilibrium_values = susceptible_initial_condition(
    domain, V_scal, DTens, K_func, lcD_values, spatial_parameters,
    equilibrium_array_path, equilibrium_signature_path,
    comm=MPI.COMM_WORLD, recompute=args.recompute_equilibrium,
)
S_equilibrium = Function(V_scal)
S_equilibrium.x.array[:] = S_equilibrium_values
S_equilibrium.x.scatter_forward()

S_initial = Function(V_S)
S_initial.interpolate(
    fem.Expression(S_equilibrium, V_S.element.interpolation_points)
)

# S: disease-free equilibrium
# I: sum of user-configured Gaussian populations
# D: zero
# E: no prions in environment at t=0
U0.sub(0).x.array[dof_map_S] = S_initial.x.array
infected_gaussians = initial_condition_parameters["infected_gaussians"]


def infected_initial_condition(x):
    """Sum configured isotropic Gaussians at the mesh coordinates."""
    values = np.zeros(x.shape[1], dtype=np.float64)
    for gaussian in infected_gaussians:
        center_x, center_y = (float(value) for value in gaussian["center"])
        mean = float(gaussian["mean"])
        standard_deviation = float(gaussian["standard_deviation"])
        squared_distance = (x[0] - center_x) ** 2 + (x[1] - center_y) ** 2
        values += mean * np.exp(
            -squared_distance / (2.0 * standard_deviation**2)
        )
    return values


U0.sub(1).interpolate(infected_initial_condition)
U0.sub(2).interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
U0.sub(3).interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
U0.x.scatter_forward()

# Forward-solve buffer
U1 = fem.Function(P)

############################
# lumped-mass quadrature measure
#
# At the vertices phi_i*phi_j = delta_ij, so integrating the mass terms with a
# vertex rule produces the lumped mass matrix M_L = diag(M*1) directly. This is
# what Section 5.3 of the write-up has always described. It matters because the
# consistent mass matrix has strictly positive off-diagonal entries, which
# destroy the M-matrix structure of (M + dt*A) and let the implicit diffusion
# step undershoot on sharp initial data — producing negative populations, which
# flip the sign of the logistic growth term and blow the solve up. For P1
# elements lumping is an O(h^2) consistent quadrature, so no accuracy order is
# lost. Both sides of the step are lumped, so the scheme really is
# (M_L + dt*A) U^{n+1} = M_L U^n + dt*F(U^n).
############################
dx_lumped = ufl.Measure(
    "dx", domain=domain,
    metadata={"quadrature_rule": "vertex", "quadrature_degree": 1},
)

############################
# bilinear form a  (LHS — time-independent, factorized once)
#
# The form carries no cross-compartment terms:
#
#   a = (uS*vS + uI*vI + uD*vD + uE*vE) * dx_lumped
#     + dt * ( DS*dot(DTens*grad(uS), grad(vS))
#            + DI*dot(DTens*grad(uI), grad(vI))
#            + DD*dot(DTens*grad(uD), grad(vD)) ) * dx
#
# so B is block diagonal by compartment and is never assembled as one matrix.
# E has no diffusion term at all, leaving the lumped mass matrix — diagonal, so
# that block is a multiply rather than a triangular solve. See
# utilities/block_solver.py; the control driver uses the same class, so both
# drivers solve the same system the same way.
############################
solver = CompartmentBlockSolver(
    P, dt, DTens, (DS, DI, DD),
    mass_measure=dx_lumped, stiffness_measure=dx, comm=MPI.COMM_WORLD,
)
if MPI.COMM_WORLD.rank == 0:
    print(
        f"Implicit operator: {solver.distinct_factorizations} factorization(s) "
        f"({solver.factor_package}), W block diagonal"
    )

############################
# linear form l  (RHS — pre-compiled once; all variable data updated in-place)
#
############################
l = (
    # ---- M_L U^n : lumped, matching the mass block of the bilinear form
    uS0 * vS + uI0 * vI + uD0 * vD + uE0 * vE
) * dx_lumped + dt * (
    # ---- S: logistic growth − direct transmission − environmental transmission
    vS * (
        ufl.conditional(
            ufl.lt(K_func, 0.01),
            -r * uS0,                                          # exponential decay in water
            r * uS0 * (1 - (uS0 + uI0 + uD0) / K_func)       # logistic growth on land
        )
        - beta  * uS0 * uD0          # direct transmission from dying deer
        - rho   * uE0 * uS0          # environmental (prion) transmission
    )

    # ---- I: direct + environmental inflow − progression
    + vI * (
          beta  * uS0 * uD0          # direct transmission
        + rho   * uE0 * uS0          # environmental transmission
        - sigma * uI0                # progression to shedding stage
    )

    # ---- D: progression from I − removal
    #         alpha is the only exit from D, so it absorbs disease mortality
    #         and any other removal of shedding animals alike.
    + vD * (
          sigma * uI0                # inflow from infected
        - alpha * uD0                # removal of shedding deer
    )

    # ---- E: prion shedding from dying deer − environmental decay
    #         (purely reactive; spatial spread via DE diffusion in the bilinear form)
    + vE * (
          p_env   * uD0              # prion shedding from dying deer
        - delta_e * uE0              # environmental prion decay
    )
) * dx

l_form = fem.form(l)   # compile once; buffers updated in-place each step

############################
# output setup
############################
xdmf_S = io.XDMFFile(domain.comm, str(output_folder / "CWD_S.xdmf"), "w")
xdmf_I = io.XDMFFile(domain.comm, str(output_folder / "CWD_I.xdmf"), "w")
xdmf_D = io.XDMFFile(domain.comm, str(output_folder / "CWD_D.xdmf"), "w")
xdmf_E = io.XDMFFile(domain.comm, str(output_folder / "CWD_E.xdmf"), "w")

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


def update_progress(completed_steps, total_steps, width=40):
    """Update one in-place progress bar on the root MPI rank."""
    if domain.comm.rank != 0:
        return
    fraction = completed_steps / total_steps if total_steps else 1.0
    filled = min(width, int(width * fraction))
    bar = "=" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\rSimulation progress: [{bar}] {fraction:6.1%} "
        f"({completed_steps}/{total_steps})"
    )
    sys.stdout.flush()

############################
# main time loop
############################
write_outputs(0.0)
total_steps = int(T / dt)
update_progress(0, total_steps)

# Assembled once and refilled in place each step; reallocating a PETSc vector
# per step is pure overhead over a thousand-step run.
L = assemble_vector(l_form)


def advance_one_step():
    """One uncontrolled IMEX step, U0 -> U0. Shared with the spin-up."""
    with L.localForm() as L_local:
        L_local.set(0.0)
    assemble_vector(L, l_form)
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    solver.solve(L, U1)
    U0.x.array[:] = U1.x.array
    U0.x.scatter_forward()

############################
# disease spin-up
#
# Advance the uncontrolled system to an observed prevalence and restart the
# clock there, so that the twenty-year window describes an established
# epizootic rather than an introduction. The control driver uses the same
# cached state, which is what keeps this run usable as its do-nothing
# baseline. See utilities/disease_spinup.py.
############################
if bool(spatial_parameters["disease_spinup"]["enabled"]):
    _uS_p, _uI_p, _uD_p, _uE_p = ufl.split(U0)
    _infected_form = fem.form((_uI_p + _uD_p) * dx)
    _deer_form = fem.form((_uS_p + _uI_p + _uD_p) * dx)

    def _prevalence():
        infected = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(_infected_form), op=MPI.SUM
        )
        total = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(_deer_form), op=MPI.SUM
        )
        return infected / total if total > 0.0 else 0.0

    _disease_array_path, _disease_signature_path = default_disease_paths(
        mesh_path
    )
    if args.disease_state is not None:
        _disease_array_path = Path(args.disease_state).expanduser().resolve()
        _disease_signature_path = _disease_array_path.with_suffix(".json")
    _expected = disease_signature(
        domain, K_func.x.array, lcD_values, S_equilibrium_values,
        spatial_parameters,
    )
    U0.x.array[:] = disease_initial_state(
        _expected, U0.x.array.size,
        lambda: run_disease_spinup(
            advance_one_step, _prevalence, lambda: U0.x.array.copy(),
            spatial_parameters, comm=MPI.COMM_WORLD,
        ),
        _disease_array_path, _disease_signature_path,
        comm=MPI.COMM_WORLD, recompute=args.recompute_disease_state,
    )
    U0.x.scatter_forward()
    write_outputs(0.0)

for i in range(total_steps):
    advance_one_step()

    if i % 3 == 0:
        write_outputs((i + 1) * dt)

    completed_steps = i + 1
    if completed_steps % 10 == 0 or completed_steps == total_steps:
        update_progress(completed_steps, total_steps)

if domain.comm.rank == 0:
    print()

xdmf_S.close()
xdmf_I.close()
xdmf_D.close()
xdmf_E.close()
if domain.comm.rank == 0:
    print(f"Done. Results written to {output_folder}")
