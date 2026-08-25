############################
# imports
############################
from mpi4py import MPI
import numpy as np
import basix.ufl
import argparse
from pathlib import Path
import sys

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

WORKFLOW_ROOT = Path(__file__).resolve().parent
UTILITIES_DIR = WORKFLOW_ROOT / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import (
    DEFAULT_PARAMETERS,
    land_cover_to_carrying_capacity,
    load_parameters,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the CWD reaction-diffusion solver using shared parameter files."
    )
    parser.add_argument(
        "--mesh-folder",
        default=None,
        help=(
            "Folder containing one .msh file, land_cover_classes.npy, "
            "land_cover_diffusivity.npy, and effective_parameters.json."
        ),
    )
    parser.add_argument("--parameters", default=None,
                        help="Combined model parameter JSON; overrides bundle discovery.")
    parser.add_argument("--mesh", default=None,
                        help="Explicit mesh path; overrides --mesh-folder discovery.")
    parser.add_argument("--land-cover-classes", default=None,
                        help="Explicit class-array path; overrides --mesh-folder discovery.")
    parser.add_argument("--land-cover-diffusivity", default=None,
                        help="Explicit diffusivity-array path; overrides --mesh-folder discovery.")
    parser.add_argument(
        "--output-folder",
        default="solver_outputs",
        help="Folder for solver XDMF/HDF5 results (default: solver_outputs).",
    )
    return parser


def _require_file(path, description):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def resolve_input_files(parsed_args):
    """Resolve a mesh bundle, while allowing individual path overrides."""
    mesh_folder = None
    if parsed_args.mesh_folder is not None:
        mesh_folder = Path(parsed_args.mesh_folder).expanduser().resolve()
        if not mesh_folder.is_dir():
            raise NotADirectoryError(f"Mesh folder does not exist: {mesh_folder}")

    if parsed_args.mesh is not None:
        mesh_path = _require_file(parsed_args.mesh, "mesh")
    elif mesh_folder is not None:
        mesh_candidates = sorted(mesh_folder.glob("*.msh"))
        if len(mesh_candidates) != 1:
            names = ", ".join(path.name for path in mesh_candidates) or "none"
            raise ValueError(
                f"Expected exactly one .msh file in {mesh_folder}; found: {names}. "
                "Use --mesh to select one explicitly."
            )
        mesh_path = mesh_candidates[0]
    else:
        mesh_path = _require_file("terrain.msh", "mesh")

    classes_path = _require_file(
        parsed_args.land_cover_classes
        or (mesh_folder / "land_cover_classes.npy" if mesh_folder else "land_cover_classes.npy"),
        "land-cover class array",
    )
    diffusivity_path = _require_file(
        parsed_args.land_cover_diffusivity
        or (
            mesh_folder / "land_cover_diffusivity.npy"
            if mesh_folder
            else "land_cover_diffusivity.npy"
        ),
        "land-cover diffusivity array",
    )

    if parsed_args.parameters is not None:
        spatial_path = _require_file(parsed_args.parameters, "model parameter file")
    elif mesh_folder is not None:
        effective_path = mesh_folder / "effective_parameters.json"
        fallback_path = mesh_folder / "parameters.json"
        if effective_path.is_file():
            spatial_path = effective_path
        elif fallback_path.is_file():
            spatial_path = fallback_path
        else:
            raise FileNotFoundError(
                "The mesh folder contains no effective_parameters.json "
                "or parameters.json."
            )
    else:
        spatial_path = _require_file(DEFAULT_PARAMETERS, "parameter file")

    return mesh_path, classes_path, diffusivity_path, spatial_path


args = build_parser().parse_args()
mesh_path, classes_path, diffusivity_path, spatial_parameter_path = resolve_input_files(args)
spatial_parameters = load_parameters(spatial_parameter_path)
output_folder = Path(args.output_folder).expanduser().resolve()
if MPI.COMM_WORLD.rank == 0:
    output_folder.mkdir(parents=True, exist_ok=True)
MPI.COMM_WORLD.barrier()
mesh_parameters = spatial_parameters["mesh"]
tensor_parameters = spatial_parameters["diffusion_tensor"]
land_cover_parameters = spatial_parameters["land_cover"]
time_parameters = spatial_parameters["time"]
kinetic_parameters = spatial_parameters["reaction"]
initial_condition_parameters = spatial_parameters["initial_conditions"]

if MPI.COMM_WORLD.rank == 0:
    print(f"Mesh: {mesh_path}")
    print(f"Land-cover classes: {classes_path}")
    print(f"Land-cover diffusivity: {diffusivity_path}")
    print(f"Model parameters: {spatial_parameter_path}")
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

# Scalar CG1 space for land-cover data and output functions
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
land_cover_classes      = np.load(classes_path).astype(np.int32)
land_cover_func         = fem.Function(V_lc)
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
# I: sum of user-configured Gaussian populations
# D: zero
# E: no prions in environment at t=0
U0.sub(0).x.array[dof_map_S] = K_S.x.array
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
############################
l = (
    # ---- S: logistic growth − direct transmission − environmental transmission
    uS0 * vS + dt * vS * (
        ufl.conditional(
            ufl.lt(K_func, 0.01),
            -r * uS0,                                          # exponential decay in water
            r * uS0 * (1 - (uS0 + uI0 + uD0) / K_func)       # logistic growth on land
        )
        - beta  * uS0 * uD0          # direct transmission from dying deer
        - rho   * uE0 * uS0          # environmental (prion) transmission
    )

    # ---- I: direct + environmental inflow − clinical progression
    + uI0 * vI + dt * vI * (
          beta  * uS0 * uD0          # direct transmission
        + rho   * uE0 * uS0          # environmental transmission
        - sigma * uI0                # progression to dying stage
    )

    # ---- D: progression from I − natural removal
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

for i in range(total_steps):
    L = assemble_vector(l_form)
    L.assemble()

    solver.solve(L, U1.x.petsc_vec)

    U0.x.array[:] = U1.x.array
    U0.x.scatter_forward()

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
