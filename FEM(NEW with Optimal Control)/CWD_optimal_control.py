"""Optimal culling control for the CWD reaction-diffusion model.

Minimizes

    J(v) = int_0^T int_Gamma [ c1*R + c2*W + c3*v^2 ] dGamma dt

over culling controls v(x, t) in the admissible box [v_min, v_max], subject to
the surface reaction-diffusion state system, where v removes *shedding* deer
through a -v*R term in the R equation (code name D). R is what drives both
transmission routes -- directly through beta_1*S*R and indirectly by generating
the environmental reservoir through rho*R -- so culling it acts on both at
once; culling I would only delay the same animals' arrival in R.

An optional L1 (elastic-net) weight c4 on v may be added to the running cost.
Because v >= 0 is already imposed, int|v| = int v on the admissible set, so the
term is smooth and only shifts stationarity to v* = P[(lambda_R*R - c4)/(2*c3)]
-- an exact-zero threshold rather than the quadratic penalty's whisper of effort
everywhere.

Algorithm (one outer iteration)
-------------------------------
  1. FORWARD SOLVE   advance the state system with the current control,
                     recording the whole trajectory y^0 .. y^N and J(v).
  2. BACKWARD SOLVE  sweep the discrete adjoint from lambda^N down to
                     lambda^1, accumulating the reduced gradient
                     grad J(v)|_b = 2*c3*v_b + c4 - mean_{n in b} R^n
                     lambda_R^(n+1), summed over the time steps of each
                     control block.
  3. OPTIMIZER STEP  L-BFGS-B on the reduced problem by default, or the
                     reference projected-gradient method with an Armijo
                     backtracking line search; each backtrack costs one more
                     forward solve.

Both sweeps reuse the block factorizations of the (symmetric) implicit
diffusion operator, built once at start-up.

Examples
--------
    python CWD_optimal_control.py --mesh-folder outputs \
        --output-folder control_outputs

    python CWD_optimal_control.py --mesh-folder outputs --gradient-check

    python CWD_optimal_control.py --mesh-folder outputs \
        --initial-control control_outputs/optimal_control.npy --max-iterations 20

    # one plain forward simulation, no optimization (the v = 0 baseline)
    python CWD_optimal_control.py --mesh-folder outputs \
        --forward-only --output-folder baseline_outputs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from dolfinx import io
from dolfinx.fem import Function

WORKFLOW_ROOT = Path(__file__).resolve().parent
UTILITIES_DIR = WORKFLOW_ROOT / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from mesh_bundle import add_bundle_arguments, resolve_input_files  # noqa: E402
from shared_parameters import load_parameters  # noqa: E402

from cwd_control_problem import CWDControlProblem  # noqa: E402


############################
# command line
############################
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Solve the culling optimal-control problem constrained by the CWD "
            "reaction-diffusion model, using a forward/backward sweep with an "
            "Armijo-backtracking projected gradient step."
        )
    )
    add_bundle_arguments(parser)
    parser.add_argument(
        "--output-folder",
        default="control_outputs",
        help="Folder for optimal-control results (default: control_outputs).",
    )
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override optimal_control.max_iterations.")
    parser.add_argument(
        "--optimizer", choices=("lbfgs", "projected-gradient"), default=None,
        help=(
            "Override optimal_control.optimizer. 'lbfgs' is SciPy's L-BFGS-B "
            "on the reduced problem; 'projected-gradient' is the original "
            "steepest descent with an Armijo line search."
        ),
    )
    parser.add_argument(
        "--control-block-years", type=float, default=None,
        help=(
            "Override optimal_control.control_block_years, the span over which "
            "the culling effort is held constant. Pass 0 for one value per "
            "time step."
        ),
    )
    parser.add_argument("--initial-step-size", type=float, default=None,
                        help="Override optimal_control.initial_step_size.")
    parser.add_argument(
        "--auto-initial-step", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Scale the first Armijo step so the most sensitive node moves "
            "across the full admissible control range (default: on). Disable "
            "to use optimal_control.initial_step_size verbatim."
        ),
    )
    parser.add_argument("--initial-control", default=None,
                        help="Warm start from a saved optimal_control.npy.")
    parser.add_argument(
        "--trajectory-file",
        default=None,
        help=(
            "Memory-map the recorded forward trajectory to this .npy path "
            "instead of holding it in RAM. Use for large meshes or long runs."
        ),
    )
    parser.add_argument("--output-every", type=int, default=3,
                        help="Write every Nth time step to XDMF (default: 3).")
    parser.add_argument("--write-adjoint", action="store_true",
                        help="Also write the adjoint fields of the final sweep.")

    # Both of these exit without optimizing, so asking for both is a mistake
    # rather than a preference.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gradient-check", action="store_true",
        help=(
            "Run a Taylor test of the reduced gradient against finite "
            "differences and exit without optimizing."
        ),
    )
    mode.add_argument(
        "--forward-only", action="store_true",
        help=(
            "Run a single forward simulation with a fixed control, write the "
            "state series, and exit without optimizing. With no "
            "--initial-control this is the uncontrolled (v = 0) baseline; with "
            "one it replays that saved strategy. No adjoint is solved, so the "
            "trajectory is never stored and the run needs no extra memory."
        ),
    )
    return parser


############################
# reporting helpers
############################
class Reporter:
    """Root-rank-only printing with an in-place sweep progress bar."""

    def __init__(self, comm):
        self.is_root = comm.rank == 0
        self._bar_open = False

    def line(self, message=""):
        if not self.is_root:
            return
        if self._bar_open:
            sys.stdout.write("\n")
            self._bar_open = False
        print(message, flush=True)

    def bar(self, label, width=32):
        def update(completed, total):
            if not self.is_root:
                return
            if completed != total and completed % 10 != 0:
                return
            fraction = completed / total if total else 1.0
            filled = min(width, int(width * fraction))
            sys.stdout.write(
                f"\r    {label:<22s}[{'=' * filled}{'-' * (width - filled)}] "
                f"{fraction:5.1%}"
            )
            sys.stdout.flush()
            self._bar_open = True
        return update

    def close_bar(self):
        if self.is_root and self._bar_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._bar_open = False


def format_bytes(count):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.1f} {unit}"
        count /= 1024


############################
# gradient verification
############################
def gradient_check(problem, reporter):
    """Taylor test: the first-order remainder must fall like O(eps^2).

    R0(eps) = |J(v + eps*d) - J(v)|                  should be O(eps)
    R1(eps) = |J(v + eps*d) - J(v) - eps*<g, d>|     should be O(eps^2)

    The test is run at its own control, not at whatever the driver was going to
    optimize from: it must sit strictly inside the bounds so that no projection
    is active and the reduced objective is genuinely differentiable along d.
    """
    trajectory = problem.new_trajectory()
    control = problem.new_control(0.5 * (problem.v_min + problem.v_max))

    rng = np.random.default_rng(0)
    direction = rng.standard_normal(control.shape)
    direction /= max(problem.norm(direction), 1e-30)

    base_cost, _, _ = problem.forward(control, trajectory)
    gradient = problem.backward(control, trajectory)
    slope = problem.inner_product(gradient, direction)

    reporter.line(f"  J(v)              = {base_cost:.12e}")
    reporter.line(f"  <grad J(v), d>    = {slope:.12e}")
    reporter.line("")
    reporter.line("      eps          R0            rate       R1            rate")

    previous = None
    ok = True
    for exponent in range(1, 7):
        eps = 10.0 ** (-exponent)
        perturbed_cost, _, _ = problem.forward(control + eps * direction)
        r0 = abs(perturbed_cost - base_cost)
        r1 = abs(perturbed_cost - base_cost - eps * slope)
        if previous is None:
            reporter.line(f"    {eps:8.1e}  {r0:.4e}     --     {r1:.4e}     --")
        else:
            rate0 = np.log(previous[0] / r0) / np.log(10) if r0 > 0 else np.inf
            rate1 = np.log(previous[1] / r1) / np.log(10) if r1 > 0 else np.inf
            reporter.line(
                f"    {eps:8.1e}  {r0:.4e}  {rate0:6.2f}   {r1:.4e}  {rate1:6.2f}"
            )
            # only judge the range where round-off has not yet taken over
            if exponent <= 4 and r1 > 1e-13 * max(1.0, abs(base_cost)):
                ok = ok and rate1 > 1.6
        previous = (r0, r1)

    reporter.line("")
    reporter.line(
        "  Taylor test PASSED (first-order remainder converges at ~2)."
        if ok else
        "  Taylor test INCONCLUSIVE: check the R1 column above for a rate near 2."
    )
    return ok


############################
# L-BFGS-B on the reduced problem
############################
def run_lbfgs(problem, reporter, control, trajectory, gradient, settings,
              history):
    """Minimize the reduced objective with SciPy's L-BFGS-B.

    The bound constraint v_min <= v <= v_max is exactly the simple box that
    L-BFGS-B is built for, and the reduced objective and its gradient are
    already available and verified by --gradient-check, so the whole optimizer
    is a wrapper: each function evaluation is one forward solve followed by one
    backward sweep.

    Why this rather than projected gradient: projected steepest descent
    converges linearly at a rate set by the conditioning of the reduced
    Hessian, whereas L-BFGS-B accumulates curvature information and converges
    superlinearly on the inactive set. The cost per iteration is essentially
    the same -- one forward and one backward -- so the saving is in the number
    of iterations.

    Returns (control, stop_reason) and appends to ``history`` in place.
    """
    try:
        from scipy.optimize import Bounds, minimize
    except ImportError as error:
        raise SystemExit(
            "optimizer = 'lbfgs' needs SciPy, which is not importable "
            f"({error}). Install it, or set optimal_control.optimizer to "
            "'projected-gradient' (or pass --optimizer projected-gradient)."
        )

    shape = control.shape
    max_iterations = int(settings["max_iterations"])
    latest = {"costs": None, "evaluations": 0, "iterations": 0}

    def objective(x):
        """J and dJ/dv at x, as the (value, gradient) pair L-BFGS-B wants."""
        candidate = x.reshape(shape)
        costs = problem.forward(candidate, trajectory)
        problem.backward(candidate, trajectory, gradient)
        latest["costs"] = costs
        latest["evaluations"] += 1
        # SciPy keeps a reference to what it is handed, and `gradient` is
        # overwritten by the next sweep, so hand over a copy.
        return costs[0], gradient.ravel().copy()

    def callback(xk):
        # Deliberately the one-argument signature: SciPy only switches to the
        # newer OptimizeResult-style callback when the parameter is named
        # `intermediate_result`, so this stays correct across versions.
        latest["iterations"] += 1
        cost, state_cost, control_cost = latest["costs"]
        history.append({
            "iteration": latest["iterations"],
            "cost": cost,
            "state_cost": state_cost,
            "control_cost": control_cost,
            "gradient_norm": problem.norm(gradient),
            "projected_gradient_norm": problem.norm(
                problem.project_control(xk.reshape(shape) - gradient)
                - xk.reshape(shape)
            ),
            "step_size": None,
            "line_search_trials": latest["evaluations"],
        })
        reporter.line(
            f"  iteration {latest['iterations']:3d}   J = {cost:.10e}   "
            f"||grad J|| = {history[-1]['gradient_norm']:.6e}   "
            f"({latest['evaluations']} evaluations so far)"
        )

    reporter.line(
        f"L-BFGS-B: up to {max_iterations} iterations, "
        f"{int(settings['lbfgs_memory'])} correction pairs"
    )
    result = minimize(
        objective,
        control.ravel().copy(),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(problem.v_min, problem.v_max),
        callback=callback,
        options={
            "maxiter": max_iterations,
            "maxcor": int(settings["lbfgs_memory"]),
            # SciPy's ftol is the same relative-decrease test the projected
            # gradient path applies, so the configured tolerance carries over.
            "ftol": float(settings["relative_cost_tolerance"]),
            # gtol is the max-norm of the projected gradient, NOT the
            # mass-weighted L2 measure reported above; the two differ in scale.
            "gtol": float(settings["gradient_tolerance"]),
            "maxls": int(settings["armijo_max_backtracks"]),
        },
    )

    control[:] = problem.project_control(result.x.reshape(shape))

    # L-BFGS-B may return an iterate that is not the last one evaluated, so the
    # stored trajectory need not belong to `control`. Everything downstream --
    # the final state series, the optional adjoint write -- assumes it does, so
    # re-solve once here rather than leaving a stale trajectory behind.
    costs = problem.forward(control, trajectory, reporter.bar("final forward"))
    reporter.close_bar()
    history.append({
        "iteration": latest["iterations"] + 1,
        "cost": costs[0],
        "state_cost": costs[1],
        "control_cost": costs[2],
        "gradient_norm": None,
        "projected_gradient_norm": None,
        "step_size": None,
        "line_search_trials": latest["evaluations"],
    })

    message = result.message
    if isinstance(message, bytes):
        message = message.decode("utf-8", "replace")
    return control, (
        f"L-BFGS-B: {message} "
        f"({result.nit} iterations, {latest['evaluations']} evaluations)"
    )


############################
# Armijo backtracking line search
############################
def armijo_line_search(problem, reporter, control, cost, gradient,
                       step_size, settings, trajectory):
    """Projected-gradient backtracking search.

    Accepts the first step for which

        J(P[v - alpha*g]) <= J(v) + eta * <g, P[v - alpha*g] - v>,

    where P projects onto [v_min, v_max]. The inner product on the right is
    never positive, so this is the standard sufficient-decrease test and it
    reduces to plain Armijo when no bound is active.

    On return, ``trajectory`` always holds the states belonging to the control
    that is returned, so no extra forward solve is needed afterwards. Returns
    (new_control, (cost, state_cost, control_cost), accepted_step, trials,
    accepted).
    """
    eta = float(settings["armijo_sufficient_decrease"])
    tau = float(settings["armijo_backtrack_factor"])
    max_backtracks = int(settings["armijo_max_backtracks"])
    minimum_step = float(settings["minimum_step_size"])

    for trial in range(1, max_backtracks + 1):
        candidate = problem.project_control(control - step_size * gradient)
        difference = candidate - control
        predicted = problem.inner_product(gradient, difference)

        candidate_costs = problem.forward(candidate, trajectory)

        if candidate_costs[0] <= cost + eta * predicted:
            return candidate, candidate_costs, step_size, trial, True

        reporter.line(
            f"    backtrack {trial:2d}: alpha = {step_size:.4e}  "
            f"J = {candidate_costs[0]:.10e}  (target <= "
            f"{cost + eta * predicted:.10e})"
        )
        step_size *= tau
        if step_size < minimum_step:
            break

    # No acceptable step: restore the trajectory belonging to the incumbent.
    incumbent_costs = problem.forward(control, trajectory)
    return control, incumbent_costs, step_size, max_backtracks, False


############################
# output
############################
class ControlOutputWriter:
    """XDMF time series for the optimal state, control, and adjoint fields."""

    def __init__(self, problem, output_folder, write_adjoint):
        self.problem = problem
        self.write_adjoint = write_adjoint
        domain = problem.domain
        comm = domain.comm

        def series(name):
            handle = io.XDMFFile(comm, str(output_folder / f"{name}.xdmf"), "w")
            handle.write_mesh(domain)
            return handle

        self.files = {
            "S": series("CWD_S"),
            "I": series("CWD_I"),
            "D": series("CWD_D"),
            "E": series("CWD_E"),
            "V": series("CWD_control"),
        }
        self.fields = {
            "S": Function(problem.V_scal, name="Susceptible"),
            "I": Function(problem.V_scal, name="Infected"),
            "D": Function(problem.V_scal, name="Dying"),
            "E": Function(problem.V_scal, name="Environment"),
            "V": Function(problem.V_scal, name="Culling_Control"),
        }
        if write_adjoint:
            for key, label in (("LS", "Adjoint_S"), ("LI", "Adjoint_I"),
                               ("LD", "Adjoint_D"), ("LE", "Adjoint_E")):
                self.files[key] = series(f"CWD_adjoint_{label[-1]}")
                self.fields[key] = Function(problem.V_scal, name=label)

        with io.XDMFFile(comm, str(output_folder / "land_cover_classes.xdmf"),
                         "w") as handle:
            handle.write_mesh(domain)
            handle.write_function(problem.land_cover_func)

    def write_state(self, state_function, control_slab, t):
        for index, key in enumerate(("S", "I", "D", "E")):
            self.fields[key].interpolate(state_function.sub(index))
            self.files[key].write_function(self.fields[key], t)
        if control_slab is not None:
            self.fields["V"].x.array[:] = control_slab
            self.fields["V"].x.scatter_forward()
            self.files["V"].write_function(self.fields["V"], t)

    def write_adjoint_fields(self, adjoint_function, t):
        if not self.write_adjoint:
            return
        for index, key in enumerate(("LS", "LI", "LD", "LE")):
            self.fields[key].interpolate(adjoint_function.sub(index))
            self.files[key].write_function(self.fields[key], t)

    def close(self):
        for handle in self.files.values():
            handle.close()


def _state_series_observer(problem, control, writer, output_every):
    """Observer that writes every output_every-th step, plus the endpoints."""

    def observer(step, t):
        if step == 0 or step % output_every == 0 or step == problem.total_steps:
            writer.write_state(problem.Y, problem.control_slab(control, step), t)

    return observer


def write_final_series(problem, control, writer, output_every, reporter,
                       trajectory):
    """Replay the optimal control once, writing the state and control series."""
    observer = _state_series_observer(problem, control, writer, output_every)
    progress = reporter.bar("writing solution")
    problem.forward(control, trajectory, progress, observer)
    reporter.close_bar()


def run_forward_only(problem, reporter, control, output_folder, output_every,
                     control_source):
    """One forward simulation with a fixed control; no optimization.

    This is the do-nothing baseline when the control is zero, and a replay of a
    saved strategy when one was loaded. Because no adjoint is solved, nothing
    downstream needs the state history, so forward() is called with
    trajectory=None: the run costs one state solve and stores no trajectory at
    all, which is what makes it usable on meshes and horizons where the
    optimization loop would not fit in memory.
    """
    writer = ControlOutputWriter(problem, output_folder, write_adjoint=False)
    observer = _state_series_observer(problem, control, writer, output_every)

    started = time.time()
    progress = reporter.bar("forward solve")
    cost, state_cost, control_cost = problem.forward(
        control, None, progress, observer
    )
    reporter.close_bar()
    writer.close()
    elapsed = time.time() - started

    reporter.line("")
    reporter.line(f"  J = {cost:.10e}   "
                  f"(state {state_cost:.6e} + control {control_cost:.6e})")
    if control_source == "zero":
        reporter.line(
            "  This is the uncontrolled baseline: the value an optimized "
            "strategy must beat."
        )
    reporter.line(f"  Elapsed: {elapsed:.1f} s")

    summary = {
        "mode": "forward_only",
        "control_source": control_source,
        "final_time_years": problem.T,
        "time_step_years": problem.dt,
        "total_steps": problem.total_steps,
        "objective": cost,
        "state_cost": state_cost,
        "control_cost": control_cost,
        "cost_weights": {"c1_shedding": problem.c1,
                         "c2_environment": problem.c2,
                         "c3_control": problem.c3,
                         "c4_control_l1": problem.c4},
        "seconds": elapsed,
    }
    if problem.comm.rank == 0:
        with (output_folder / "forward_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(summary, stream, indent=2)
    problem.comm.barrier()
    reporter.line(f"  Wrote {output_folder / 'forward_summary.json'}")
    return 0


############################
# main
############################
def main():
    comm = MPI.COMM_WORLD
    args = build_parser().parse_args()
    reporter = Reporter(comm)

    mesh_path, classes_path, diffusivity_path, parameter_path = \
        resolve_input_files(args)
    spatial_parameters = load_parameters(parameter_path)
    settings = spatial_parameters["optimal_control"]

    if args.max_iterations is not None:
        settings["max_iterations"] = args.max_iterations
    if args.initial_step_size is not None:
        settings["initial_step_size"] = args.initial_step_size
    if args.optimizer is not None:
        settings["optimizer"] = args.optimizer
    if args.control_block_years is not None:
        settings["control_block_years"] = (
            None if args.control_block_years <= 0.0 else args.control_block_years
        )
    optimizer_name = settings["optimizer"]

    output_folder = Path(args.output_folder).expanduser().resolve()
    if comm.rank == 0:
        output_folder.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    reporter.line(f"Mesh:                   {mesh_path}")
    reporter.line(f"Land-cover classes:     {classes_path}")
    reporter.line(f"Land-cover diffusivity: {diffusivity_path}")
    reporter.line(f"Model parameters:       {parameter_path}")
    reporter.line(f"Outputs:                {output_folder}")
    reporter.line("")

    problem = CWDControlProblem(
        mesh_path, classes_path, diffusivity_path, spatial_parameters, comm,
        equilibrium_array_path=args.susceptible_equilibrium,
        recompute_equilibrium=args.recompute_equilibrium,
        disease_state_path=args.disease_state,
        recompute_disease_state=args.recompute_disease_state,
    )

    reporter.line(
        f"Scalar degrees of freedom: {problem.n_global}   "
        f"time steps: {problem.total_steps}   dt = {problem.dt} yr   "
        f"T = {problem.T} yr"
    )
    reporter.line(
        f"Implicit operator: {problem.solver.distinct_factorizations} "
        f"factorization(s) of {problem.n_global} dofs "
        f"({problem.solver.factor_package}), W block diagonal"
    )
    reporter.line(
        f"Cost weights: c1 (shedding) = {problem.c1}, "
        f"c2 (environment) = {problem.c2}, c3 (control) = {problem.c3}"
        + (f", c4 (control L1) = {problem.c4}" if problem.c4 > 0 else "")
    )
    if problem.c4 > 0 and problem.c3 > 0:
        reporter.line(
            f"  elastic net: v = 0 wherever lambda_R*R < {problem.c4}, "
            f"then ramping at 1/(2*c3) = {0.5 / problem.c3:.4g}"
        )
    elif problem.c4 > 0:
        reporter.line(
            f"  pure L1 (bang-bang): v = v_max wherever lambda_R*R > "
            f"{problem.c4}, else 0"
        )
    reporter.line(
        f"Admissible control: {problem.v_min} <= v <= {problem.v_max} per year"
    )
    if problem.n_blocks == problem.total_steps:
        blocking = "one value per time step"
    else:
        blocking = (
            f"piecewise constant on {problem.block_years:g}-year blocks "
            f"({problem.block_step_counts[0]:.0f} steps each)"
        )
    reporter.line(
        f"Control: {problem.n_blocks} blocks x {problem.n_global} nodes "
        f"= {problem.control_bytes() / 2**20:.1f} MiB per copy, {blocking}"
    )
    if optimizer_name == "lbfgs":
        pairs = int(settings["lbfgs_memory"])
        reporter.line(
            f"Optimizer: L-BFGS-B, {pairs} correction pairs "
            f"(~{(2 * pairs + 5) * problem.control_bytes() / 2**30:.2f} GiB "
            "workspace)"
        )
    else:
        reporter.line("Optimizer: projected gradient with Armijo line search")
    if args.forward_only:
        reporter.line(
            "Stored forward trajectory: none "
            f"(--forward-only solves no adjoint, saving "
            f"{format_bytes(problem.trajectory_bytes())} per rank)"
        )
    else:
        reporter.line(
            "Stored forward trajectory: "
            f"{format_bytes(problem.trajectory_bytes())} per rank"
            + (f" (memory-mapped to {args.trajectory_file})"
               if args.trajectory_file else " (in memory)")
        )
    reporter.line("")

    ############################
    # initial control
    ############################
    initial_value = float(settings["initial_control"])
    control = problem.new_control(initial_value)
    control_source = "zero" if initial_value == 0.0 else f"uniform {initial_value}"
    if args.initial_control is not None:
        loaded = np.load(Path(args.initial_control).expanduser().resolve())
        if loaded.shape != control.shape:
            raise ValueError(
                f"--initial-control has shape {loaded.shape}, but this problem "
                f"expects {control.shape} ({problem.n_blocks} control blocks of "
                f"{problem.block_years:g} yr). A control saved under a "
                "different optimal_control.control_block_years, or before "
                "block controls existed, has one slab per time step and cannot "
                "be reused directly."
            )
        control = problem.project_control(loaded.astype(np.float64))
        control_source = str(args.initial_control)
        reporter.line(f"Warm start from {args.initial_control}")

    ############################
    # forward-only mode
    ############################
    if args.forward_only:
        if control_source == "zero":
            reporter.line("Forward simulation only, uncontrolled (v = 0)")
        else:
            reporter.line(f"Forward simulation only, replaying {control_source}")
        reporter.line("")
        return run_forward_only(problem, reporter, control, output_folder,
                                args.output_every, control_source)

    ############################
    # gradient check mode
    ############################
    if args.gradient_check:
        reporter.line("Gradient check (Taylor test of the discrete adjoint)")
        reporter.line("")
        passed = gradient_check(problem, reporter)
        return 0 if passed else 1

    ############################
    # forward/backward optimization loop
    ############################
    trajectory = problem.new_trajectory(args.trajectory_file)
    gradient = np.empty_like(control)

    reporter.line("Initial forward solve")
    started = time.time()
    cost, state_cost, control_cost = problem.forward(
        control, trajectory, reporter.bar("forward solve")
    )
    reporter.close_bar()
    reporter.line(
        f"  J = {cost:.10e}   (state {state_cost:.6e} + control {control_cost:.6e})"
    )
    reporter.line("")

    step_size = float(settings["initial_step_size"])
    growth = float(settings["armijo_step_growth"])
    gradient_tolerance = float(settings["gradient_tolerance"])
    cost_tolerance = float(settings["relative_cost_tolerance"])
    max_iterations = int(settings["max_iterations"])

    history = [{
        "iteration": 0,
        "cost": cost,
        "state_cost": state_cost,
        "control_cost": control_cost,
        "gradient_norm": None,
        "projected_gradient_norm": None,
        "step_size": None,
        "line_search_trials": 0,
    }]
    stop_reason = f"reached max_iterations = {max_iterations}"

    if optimizer_name == "lbfgs":
        control, stop_reason = run_lbfgs(
            problem, reporter, control, trajectory, gradient, settings, history
        )
        cost = history[-1]["cost"]
        gradient_iterations = ()
    else:
        gradient_iterations = range(1, max_iterations + 1)

    for iteration in gradient_iterations:
        reporter.line(f"Iteration {iteration}")

        problem.backward(control, trajectory, gradient,
                         reporter.bar("backward solve"))
        reporter.close_bar()

        gradient_norm = problem.norm(gradient)
        # Projected-gradient stationarity measure: how far a unit step moves us.
        projected = problem.project_control(control - gradient) - control
        projected_norm = problem.norm(projected)
        reporter.line(
            f"    ||grad J|| = {gradient_norm:.6e}   "
            f"||P[v - grad J] - v|| = {projected_norm:.6e}"
        )

        if projected_norm <= gradient_tolerance:
            stop_reason = (
                f"projected gradient norm {projected_norm:.3e} <= "
                f"gradient_tolerance {gradient_tolerance:.3e}"
            )
            reporter.line(f"  Converged: {stop_reason}")
            break

        if iteration == 1 and args.auto_initial_step:
            # Size the first trial step so that -alpha*g moves the most
            # gradient-sensitive node across the whole admissible range. The
            # raw parameter value is otherwise unit-dependent guesswork, and
            # Armijo only halves from wherever it starts.
            local_peak = float(np.abs(gradient[:, : problem.n_owned]).max()) \
                if problem.n_owned else 0.0
            peak = comm.allreduce(local_peak, op=MPI.MAX)
            if peak > 0.0:
                step_size = (problem.v_max - problem.v_min) / peak
                reporter.line(f"    auto initial step alpha = {step_size:.4e}")
        elif iteration > 1:
            step_size *= growth
        control, costs, step_size, trials, accepted = armijo_line_search(
            problem, reporter, control, cost, gradient, step_size,
            settings, trajectory
        )
        new_cost, state_cost, control_cost = costs

        if not accepted:
            stop_reason = (
                "Armijo line search failed to find a decreasing step "
                f"(alpha fell below {settings['minimum_step_size']:.1e})"
            )
            reporter.line(f"  Stopping: {stop_reason}")
            break

        relative_decrease = abs(cost - new_cost) / max(abs(cost), 1e-300)
        reporter.line(
            f"    accepted alpha = {step_size:.4e} after {trials} "
            f"evaluation(s);  J = {new_cost:.10e}  "
            f"(relative decrease {relative_decrease:.3e})"
        )

        history.append({
            "iteration": iteration,
            "cost": new_cost,
            "state_cost": state_cost,
            "control_cost": control_cost,
            "gradient_norm": gradient_norm,
            "projected_gradient_norm": projected_norm,
            "step_size": step_size,
            "line_search_trials": trials,
        })
        cost = new_cost

        if relative_decrease <= cost_tolerance:
            stop_reason = (
                f"relative cost decrease {relative_decrease:.3e} <= "
                f"relative_cost_tolerance {cost_tolerance:.3e}"
            )
            reporter.line(f"  Converged: {stop_reason}")
            break

    elapsed = time.time() - started
    reporter.line("")
    reporter.line(f"Stopped: {stop_reason}")

    ############################
    # final report and output
    ############################
    # `trajectory` already belongs to `control`: the line search leaves it that
    # way on both the accepting and the rejecting path, and the other exits
    # never touched it. No extra forward solve is needed here.
    final = history[-1]
    final_cost = final["cost"]
    initial_cost = history[0]["cost"]

    if initial_cost > 0:
        reduction = f"{100.0 * (1.0 - final_cost / initial_cost):.2f}% reduction"
    else:
        reduction = "no positive baseline to compare against"
    reporter.line(
        f"J(initial) = {initial_cost:.10e}  ->  J(final) = {final_cost:.10e}   "
        f"({reduction})"
    )
    reporter.line(
        f"  final split: state {final['state_cost']:.6e} + "
        f"control {final['control_cost']:.6e}"
    )

    local_max = float(control[:, : problem.n_owned].max()) if problem.n_owned else 0.0
    control_max = comm.allreduce(local_max, op=MPI.MAX)
    ones = np.ones_like(control)
    space_time_measure = problem.inner_product(ones, ones)  # |Gamma| * T
    mean_control = problem.inner_product(control, ones) / space_time_measure
    reporter.line(
        f"  control: peak {control_max:.4f} /yr, "
        f"space-time mean {mean_control:.4f} /yr"
    )

    # Sparsity of the strategy: what share of the space-time domain the
    # optimizer leaves completely alone, and what share it drives to the bound.
    # These are the two numbers to watch when sweeping the L1 weight -- the
    # quadratic penalty alone produces almost no exact zeros, while a growing
    # c4 should push the untouched fraction up and pull the map toward
    # cull-here-not-there.
    at_zero = (control <= problem.v_min + 1e-12).astype(np.float64)
    at_bound = (control >= problem.v_max - 1e-12).astype(np.float64)
    zero_fraction = problem.inner_product(at_zero, ones) / space_time_measure
    bound_fraction = problem.inner_product(at_bound, ones) / space_time_measure
    reporter.line(
        f"  strategy: {100.0 * zero_fraction:.1f}% of space-time at v = "
        f"{problem.v_min} (no culling), "
        f"{100.0 * bound_fraction:.1f}% at v = {problem.v_max}"
    )
    reporter.line(f"  elapsed: {elapsed:.1f} s")
    reporter.line("")

    if comm.rank == 0:
        np.save(output_folder / "optimal_control.npy", control)
        with (output_folder / "optimization_history.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "stop_reason": stop_reason,
                    "elapsed_seconds": elapsed,
                    "cost_weights": {
                        "c1_shedding": problem.c1,
                        "c2_environment": problem.c2,
                        "c3_control": problem.c3,
                        "c4_control_l1": problem.c4,
                    },
                    "control_bounds": [problem.v_min, problem.v_max],
                    "optimizer": optimizer_name,
                    "control_blocks": {
                        "count": problem.n_blocks,
                        "block_years": problem.block_years,
                        "steps_per_block": problem.block_step_counts.tolist(),
                    },
                    "time": {"final_time_years": problem.T,
                             "time_step_years": problem.dt,
                             "total_steps": problem.total_steps},
                    "iterations": history,
                },
                stream,
                indent=2,
            )
    comm.barrier()

    output_every = max(1, args.output_every)
    writer = ControlOutputWriter(problem, output_folder, args.write_adjoint)
    write_final_series(problem, control, writer, output_every, reporter,
                       trajectory)

    if args.write_adjoint:
        def adjoint_observer(step, t):
            if step % output_every == 0 or step == problem.total_steps:
                writer.write_adjoint_fields(problem.Lam, t)

        progress = reporter.bar("writing adjoint")
        problem.backward(control, trajectory, gradient, progress,
                         adjoint_observer)
        reporter.close_bar()

    writer.close()

    reporter.line(f"Done. Results written to {output_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
