"""Controlled CWD state system, its discrete adjoint, and the reduced gradient.

This module holds everything that the forward solve, the backward (adjoint)
solve, the reduced objective, and the reduced gradient must agree on: the mesh,
the mixed function space, the land-cover data fields, the anisotropic diffusion
tensor, and the variational forms. CWD_optimal_control.py drives it; the
uncontrolled reference solver CWD_solver.py sits beside it and shares the same
mesh bundle, the same disease-free spin-up, and the same block solver.

Model (code names in parentheses; see the notation table in the write-up)
------------------------------------------------------------------------
    S (uS)  susceptible deer
    I (uI)  infected, sub-clinical, non-shedding deer
    R (uD)  prion-shedding deer, pre-clinical through clinical   ("dying")
    W (uE)  environmental prion contamination                    ("environment")

There is no separate background-mortality term: sigma is the only exit from I
and alpha the only exit from R, so every infected animal reaches the shedding
class and alpha absorbs all removal of shedding deer. Reinstating a non-CWD
mortality mu on I and R means adding -mu*I and -mu*R here, the matching -mu
entries to the (I,I) and (R,R) diagonal of the adjoint below, and rescaling
beta_1 and beta_2, which were back-solved from a target R0 that mu enters.

The culling control v(x, t) removes *shedding* deer, entering the R equation as
a -v*R term. The objective minimized is

    J(v) = int_0^T int_Gamma [ c1*R + c2*W + c3*v^2 ] dGamma dt.

R is the right target on both counts: it is what drives direct transmission
(beta_1*S*R) and it is what generates the environmental reservoir (rho*R), so
culling it acts on both routes at once. Culling I, as an earlier revision did,
only delays the same animals' arrival in R.

Discretization
--------------
The state system is advanced with the same semi-implicit (IMEX) Euler step the
uncontrolled solver uses: diffusion implicit, reactions and the control term
explicit at the previous time level. Writing the four-compartment nodal vector
as y^n and the block operator assembled from the bilinear form as B,

    B y^(n+1) = L(y^n, v^n),        n = 0, ..., N-1.

The adjoint implemented here is the *discrete* adjoint of that scheme rather
than a re-discretization of the continuous adjoint PDE, so the gradient it
produces is the exact derivative of the discrete reduced objective (verify with
--gradient-check). Two consequences are worth noting:

  * B is symmetric: the mass matrix is symmetric and the diffusion tensor DTens
    is symmetric, so B^T = B. The factorizations built before the time loop
    therefore serve the forward *and* the backward sweep; no second matrix and
    no second factorization is ever assembled.

    B is also block diagonal by compartment -- the bilinear form has no
    cross-compartment terms -- so it is never assembled as one matrix at all.
    Each block is factorized separately, blocks with equal mobility share one
    factorization, and the W block is the lumped mass matrix, which is diagonal.
    That is an exact restatement of the same linear system, not an
    approximation, so the discrete adjoint identity is unaffected; see
    utilities/block_solver.py.
  * The adjoint right-hand side below is the exact transpose of the Jacobian of
    L with respect to y^n, including the K < 0.01 branch that swaps logistic
    growth for exponential decay over water.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.io.gmsh import read_from_msh
from ufl import (
    CellNormal, Identity, TestFunction,
    as_vector, dot, dx, exp, outer, split,
)

UTILITIES_DIR = Path(__file__).resolve().parent / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from block_solver import CompartmentBlockSolver  # noqa: E402
from shared_parameters import land_cover_to_carrying_capacity  # noqa: E402
from susceptible_spinup import (  # noqa: E402
    default_equilibrium_paths,
    susceptible_initial_condition,
)
from disease_spinup import (  # noqa: E402
    default_disease_paths,
    disease_initial_state,
    disease_signature,
    run_disease_spinup,
)


# Every form that participates in the adjoint identity -- the forward
# right-hand side, its transpose, the running cost, and the gradient -- is
# integrated with this one fixed rule. Quadrature is a linear functional, so
# differentiating a quadrature sum and quadrature-summing the analytic
# derivative agree *exactly* when the rule is the same; letting UFL estimate a
# different degree per form would leave the adjoint only approximately
# transposed and would show up as a first-order Taylor test. Degree 4 is what
# UFL estimates for the cubic transmission terms divided by the carrying
# capacity, so this also preserves the uncontrolled solver's behaviour.
QUADRATURE_DEGREE = 4


class CWDControlProblem:
    """Everything the optimization loop needs, assembled once."""

    def __init__(self, mesh_path, classes_path, diffusivity_path,
                 spatial_parameters, comm=MPI.COMM_WORLD,
                 equilibrium_array_path=None, equilibrium_signature_path=None,
                 recompute_equilibrium=False, disease_state_path=None,
                 recompute_disease_state=False):
        self.comm = comm
        self.parameters = spatial_parameters

        # Right-hand-side vectors, one per compiled form, reassembled in place
        # each step rather than reallocated. Keyed by a caller-supplied name so
        # the lifetime is explicit.
        self._rhs_buffers = {}

        # Cached disease-free susceptible field; defaults to sitting beside the
        # mesh, since it belongs to the bundle rather than to any one run.
        default_array, default_signature = default_equilibrium_paths(mesh_path)
        self.equilibrium_array_path = (
            Path(equilibrium_array_path).expanduser().resolve()
            if equilibrium_array_path is not None
            else default_array
        )
        self.equilibrium_signature_path = (
            Path(equilibrium_signature_path).expanduser().resolve()
            if equilibrium_signature_path is not None
            else (
                default_signature
                if equilibrium_array_path is None
                else self.equilibrium_array_path.with_suffix(".json")
            )
        )
        self.recompute_equilibrium = bool(recompute_equilibrium)

        # Cached spun-up epizootic. Same reasoning as the susceptible field: it
        # belongs to the mesh bundle plus a parameter set, not to a run.
        default_disease_array, default_disease_signature = (
            default_disease_paths(mesh_path)
        )
        self.disease_array_path = (
            Path(disease_state_path).expanduser().resolve()
            if disease_state_path is not None
            else default_disease_array
        )
        self.disease_signature_path = (
            default_disease_signature
            if disease_state_path is None
            else self.disease_array_path.with_suffix(".json")
        )
        self.recompute_disease_state = bool(recompute_disease_state)

        tensor_parameters = spatial_parameters["diffusion_tensor"]
        time_parameters = spatial_parameters["time"]
        kinetic_parameters = spatial_parameters["reaction"]
        control_parameters = spatial_parameters["optimal_control"]
        self.initial_condition_parameters = spatial_parameters["initial_conditions"]

        ############################
        # model, time, and cost parameters
        ############################
        self.T = float(time_parameters["final_time_years"])
        self.dt = float(time_parameters["time_step_years"])
        self.total_steps = int(self.T / self.dt)
        if self.total_steps < 1:
            raise ValueError(
                "time.final_time_years / time.time_step_years must be at least 1"
            )

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
        # control time blocks
        #
        # The control is piecewise constant on blocks of time steps rather than
        # on every step. Two reasons, one practical and one about the model.
        #
        # Practically, the optimization variable is a full space-time field:
        # one CG1 field per block, so total_steps blocks on a 210k-node mesh
        # over 1000 steps is 1.7 GiB per copy, and any quasi-Newton method
        # holding a handful of correction pairs is immediately out of reach.
        # Annual blocks cut that by fifty.
        #
        # About the model: a per-step control lets the culling effort change
        # every 7.3 days, which no management agency does. Quotas are set
        # annually. Restricting the admissible set to block-constant controls is
        # therefore a closer description of what can actually be implemented,
        # not merely a discretization convenience -- though it *is* a
        # restriction, so the optimal J is no lower than the per-step optimum.
        block_years = control_parameters["control_block_years"]
        self.block_years = self.dt if block_years is None else float(block_years)
        step_start_times = np.arange(self.total_steps) * self.dt
        self.block_of_step = np.floor(
            step_start_times / self.block_years + 1e-9
        ).astype(np.int64)
        self.n_blocks = int(self.block_of_step[-1]) + 1
        self.block_step_counts = np.bincount(
            self.block_of_step, minlength=self.n_blocks
        ).astype(np.float64)
        # Each block's share of the horizon, used to weight the space-time L2
        # inner product so that it remains the true L2 product of the
        # block-constant functions rather than a per-block sum.
        self.block_weights = self.block_step_counts * self.dt

        self.c1 = float(control_parameters["cost_shedding"])
        self.c2 = float(control_parameters["cost_environment"])
        self.c3 = float(control_parameters["cost_control"])
        # L1 (elastic-net) control weight. Because v >= 0 is already imposed,
        # int|v| = int v on the admissible set: the kink of the absolute value
        # sits exactly on the constraint boundary, so this term is a smooth
        # linear functional and needs no proximal machinery. What it buys is a
        # threshold -- the optimum is exactly zero wherever lambda_R*R < c4 --
        # instead of the quadratic penalty's whisper of effort everywhere.
        self.c4 = float(control_parameters["cost_control_l1"])
        self.v_min = float(control_parameters["control_minimum"])
        self.v_max = float(control_parameters["control_maximum"])

        ############################
        # domain and function spaces
        ############################
        mesh_data = read_from_msh(str(mesh_path), comm, gdim=3)
        domain = mesh_data.mesh
        self.domain = domain

        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        mixed = basix.ufl.mixed_element([P1, P1, P1, P1])
        self.P = fem.functionspace(domain, mixed)
        self.V_scal = fem.functionspace(domain, ("CG", 1))

        index_map = self.V_scal.dofmap.index_map
        block_size = self.V_scal.dofmap.index_map_bs
        self.n_owned = index_map.size_local * block_size
        self.n_local = self.n_owned + index_map.num_ghosts * block_size
        self.n_global = index_map.size_global * block_size

        # No TrialFunction: the bilinear form is assembled and factorized
        # inside CompartmentBlockSolver, and this class builds load vectors
        # only.
        V = TestFunction(self.P)
        vS, vI, vD, vE = split(V)

        # State at the current time level; also the coefficient buffer for the
        # explicit reaction terms and for the adjoint linearization point.
        self.Y = Function(self.P)
        uS0, uI0, uD0, uE0 = split(self.Y)
        self.Y_next = Function(self.P)

        # Adjoint at the *later* time level, lambda^(n+1).
        self.Lam = Function(self.P)
        lS, lI, lD, lE = split(self.Lam)
        self.Lam_next = Function(self.P)

        # Control on one time slab, v^n, as a nodal CG1 field.
        self.v_func = Function(self.V_scal)
        self.v_func.name = "Culling_Control"

        ############################
        # land cover, diffusivity, carrying capacity
        ############################
        land_cover_classes = np.load(classes_path).astype(np.int32)
        self.land_cover_func = Function(self.V_scal)
        self.land_cover_func.name = "Land_Cover_Class"
        self._check_nodal_array(land_cover_classes, "land_cover_classes.npy")
        self.land_cover_func.x.array[:] = land_cover_classes
        self.land_cover_func.x.scatter_forward()

        lcD_values = np.load(diffusivity_path)
        self.lcD_values = lcD_values
        lcD_scale = Function(self.V_scal)
        self._check_nodal_array(lcD_values, "land_cover_diffusivity.npy")
        lcD_scale.x.array[:] = lcD_values
        lcD_scale.x.scatter_forward()

        K_values = land_cover_to_carrying_capacity(
            land_cover_classes, spatial_parameters
        )
        K_func = Function(self.V_scal)
        K_func.x.array[:] = K_values
        K_func.x.scatter_forward()
        self.K_func = K_func

        ############################
        # diffusion tensor (identical to the uncontrolled solver)
        ############################
        n = CellNormal(domain)
        g = as_vector((0, 0, -1))
        cos_theta = dot(g, n)

        es_raw = g - dot(g, n) * n
        es_norm = ufl.sqrt(dot(es_raw, es_raw))

        xdir = as_vector((1, 0, 0))
        x_proj_raw = xdir - dot(xdir, n) * n
        x_proj = x_proj_raw / ufl.sqrt(dot(x_proj_raw, x_proj_raw))

        es = ufl.conditional(ufl.gt(es_norm, 1e-6), es_raw / es_norm, x_proj)

        activation = (
            1 + exp(activation_steepness * (activation_cosine_threshold - 1))
        ) / (
            1 + exp(
                activation_steepness
                * (activation_cosine_threshold - abs(cos_theta))
            )
        )
        Ds = kappa * (isotropy + (1 - isotropy) * activation)
        Dc = kappa
        DTens = lcD_scale * (
            (Ds - Dc) * outer(es, es) + Dc * (Identity(3) - outer(n, n))
        )
        self.DTens = DTens

        ############################
        # quadrature measures
        #
        # dq  : the shared, explicitly pinned Gauss rule used by every form
        #       that takes part in the adjoint identity -- see
        #       QUADRATURE_DEGREE above.
        # dl  : a vertex rule, used for the mass terms only. At the vertices
        #       phi_i*phi_j = delta_ij, so integrating the mass this way gives
        #       the LUMPED mass matrix M_L = diag(M*1) by construction.
        #
        # Lumping matters for more than convenience. With the consistent mass
        # matrix, M carries strictly positive off-diagonal entries, which
        # destroys the M-matrix structure of (M + dt*A) and lets the implicit
        # diffusion step undershoot on sharp data -- producing negative
        # populations, which then flip the sign of the logistic growth term and
        # blow the solve up. With M_L the off-diagonals are zero and, on a
        # well-shaped mesh, (M_L + dt*A) is an M-matrix, so a non-negative
        # right-hand side gives a non-negative solution. For P1 elements
        # lumping is an O(h^2) consistent quadrature, so no accuracy order is
        # lost. This is also what the write-up has always claimed the solver
        # does.
        #
        # Both sides of the step are lumped: the mass block of B below, and the
        # y^n*psi terms of the load vector, so that the scheme really is
        # (M_L + dt*A) y^(n+1) = M_L y^n + dt*F. The reaction terms stay on dq.
        ############################
        dq = ufl.Measure(
            "dx", domain=domain,
            metadata={"quadrature_degree": QUADRATURE_DEGREE},
        )
        dl = ufl.Measure(
            "dx", domain=domain,
            metadata={"quadrature_rule": "vertex", "quadrature_degree": 1},
        )
        self.dq = dq
        self.dl = dl

        ############################
        # bilinear form  ->  B   (symmetric; factorized once, used both ways)
        #
        # Lumped mass + Gauss stiffness. B stays symmetric -- a diagonal mass
        # is symmetric and the diffusion tensor is -- so B^T = B still holds
        # and the single LU factorization still serves both sweeps.
        ############################
        # a(U, Psi) is written out here for the record, but is never assembled
        # as one matrix: it has no cross-compartment terms, so B is block
        # diagonal and is solved blockwise. See utilities/block_solver.py.
        #
        #   a = (uS*vS + uI*vI + uD*vD + uE*vE) * dl
        #     + dt * ( DS*dot(DTens*grad(uS), grad(vS))
        #            + DI*dot(DTens*grad(uI), grad(vI))
        #            + DD*dot(DTens*grad(uD), grad(vD)) ) * dx
        #
        # E carries no diffusion term, so its block is the lumped mass matrix
        # alone -- diagonal, and solved by a multiply rather than by any
        # factorization. Each block is symmetric and so is B, which is what lets
        # the same factorizations serve the forward and the backward sweep.
        self.solver = CompartmentBlockSolver(
            self.P, self.dt, DTens, (DS, DI, DD),
            mass_measure=dl, stiffness_measure=dx, comm=comm,
        )

        ############################
        # forward right-hand side  L(y^n, v^n)
        #
        # The y^n*psi mass terms use the vertex rule dl, matching the mass
        # block of B; the reaction terms use the pinned Gauss rule dq. Each
        # term's derivative below is taken on the same rule as the term itself,
        # which is what keeps the discrete adjoint an exact transpose.
        ############################
        # Wrapped as Constants so that the forward growth term and the
        # derivatives of it in the adjoint form below are built from the same
        # UFL objects, not from a float in one place and a Constant in the
        # other.
        zero_c = fem.Constant(domain, PETSc.ScalarType(0.0))
        r_c = fem.Constant(domain, PETSc.ScalarType(r))

        water = ufl.lt(K_func, 0.01)
        growth = ufl.conditional(
            water,
            -r_c * uS0,                                    # exponential decay in water
            r_c * uS0 * (1 - (uS0 + uI0 + uD0) / K_func),  # logistic growth on land
        )

        self.state_rhs_form = fem.form(
            # ---- M_L y^n : lumped, matching the mass block of B
            (uS0 * vS + uI0 * vI + uD0 * vD + uE0 * vE) * dl

            # ---- dt * F(y^n, v^n) : reactions, on the pinned Gauss rule
            + self.dt * (
                # S: growth - direct transmission - environmental transmission
                vS * (
                    growth
                    - beta * uS0 * uD0
                    - rho * uE0 * uS0
                )

                # I: transmission inflow - progression
                + vI * (
                      beta * uS0 * uD0
                    + rho * uE0 * uS0
                    - sigma * uI0
                )

                # D: progression inflow - removal - CULLING
                + vD * (
                      sigma * uI0
                    - alpha * uD0
                    - self.v_func * uD0      # <-- the control enters here
                )

                # E: shedding inflow - environmental decay
                + vE * (
                      p_env * uD0
                    - delta_e * uE0
                )
            ) * dq
        )

        ############################
        # adjoint right-hand side   (d L / d y^n)^T lambda^(n+1)  +  dt * g
        #
        # Row k below collects  sum_j  lambda_j * d(L_j)/d(y_k), which is the
        # exact transpose of the Jacobian of the forward right-hand side. The
        # two derivatives of the growth term inherit the same water branch the
        # forward form uses.
        ############################
        growth_dS = ufl.conditional(
            water,
            -r_c,
            r_c * (1 - (uS0 + uI0 + uD0) / K_func) - r_c * uS0 / K_func,
        )
        growth_dN = ufl.conditional(water, zero_c, -r_c * uS0 / K_func)

        # The running cost weights R and W, so it sources the D and E rows.
        running_cost_rhs = self.dt * (self.c1 * vD + self.c2 * vE)

        self.adjoint_rhs_form = fem.form(
            # ---- M_L lambda^(n+1) : the transpose of the lumped mass terms of
            #      the forward load vector, so it must use the same rule
            (lS * vS + lI * vI + lD * vD + lE * vE) * dl

            # ---- dt * (dF/dy)^T lambda^(n+1) : on the pinned Gauss rule
            + self.dt * (
                # row S
                vS * (
                      lS * (growth_dS - beta * uD0 - rho * uE0)
                    + lI * (beta * uD0 + rho * uE0)
                )

                # row I
                + vI * (
                      lS * growth_dN
                    - lI * sigma
                    + lD * sigma
                )

                # row D  (the control enters the -(alpha + v) coefficient)
                + vD * (
                      lS * (growth_dN - beta * uS0)
                    + lI * beta * uS0
                    - lD * (alpha + self.v_func)
                    + lE * p_env
                )

                # row E
                + vE * (
                    - lS * rho * uS0
                    + lI * rho * uS0
                    - lE * delta_e
                )
            ) * dq

            # ---- dt * g : derivative of the running cost, which is itself
            #      assembled on dq, so this term stays there too
            + running_cost_rhs * dq
        )

        # Terminal adjoint: B lambda^N = dt * g, with g the running-cost vector.
        self.terminal_rhs_form = fem.form(running_cost_rhs * dq)

        ############################
        # reduced objective and reduced gradient
        ############################
        q = TestFunction(self.V_scal)
        self.state_cost_form = fem.form((self.c1 * uD0 + self.c2 * uE0) * dq)
        self.control_cost_form = fem.form(
            (self.c3 * self.v_func**2 + self.c4 * self.v_func) * dq
        )
        # dJ/dv on one slab, in weak (mass-weighted) form; dividing by the
        # lumped mass below turns it into the L2 Riesz representative. The
        # control enters only the R equation, where d f_R / d v = -R, so the
        # sensitivity pairs R with lambda_R.
        # d/dv of (c3*v^2 + c4*v) is 2*c3*v + c4; the state sensitivity term is
        # unchanged. The constant c4 shifts the stationarity condition to
        # v* = P[(lambda_R*R - c4) / (2*c3)], hence the exact zeros.
        self.gradient_form = fem.form(
            (2.0 * self.c3 * self.v_func + self.c4 - uD0 * lD) * q * dq
        )
        self.lumped_mass = self._assemble_scalar_vector(fem.form(q * dq))

        # Domain-wide prevalence, for the disease spin-up stopping test. Both
        # integrals are over the whole mesh, so this is the population
        # prevalence rather than a nodal average, and it is the quantity the
        # surveillance figures the target is drawn from actually estimate.
        uS_c, uI_c, uD_c, _uE_c = ufl.split(self.Y)
        self.infected_mass_form = fem.form((uI_c + uD_c) * dq)
        self.deer_mass_form = fem.form((uS_c + uI_c + uD_c) * dq)

        ############################
        # initial state y^0 (control-independent)
        ############################
        self.V_S, self.dof_map_S = self.P.sub(0).collapse()
        self.Y_initial = Function(self.P)
        self._build_initial_state()

    # ------------------------------------------------------------------ setup

    def _check_nodal_array(self, values, name):
        if values.size != self.n_local:
            raise ValueError(
                f"{name} has {values.size} values, but this mesh/function space "
                f"expects {self.n_local}. The mesh and arrays are not a "
                "matching bundle."
            )

    def _build_initial_state(self):
        """S = disease-free equilibrium, I = Gaussian foci, R = W = 0.

        S does *not* start at the carrying capacity K. K is a piecewise-constant
        land-cover lookup and is not a steady state of the S equation, so
        starting there superimposes an order-one demographic transient on the
        epidemic -- spurious structure the optimizer would spend control effort
        on. The relaxed disease-free field is computed once per mesh bundle and
        cached; see utilities/susceptible_spinup.py.
        """
        S_values = susceptible_initial_condition(
            self.domain, self.V_scal, self.DTens, self.K_func, self.lcD_values,
            self.parameters, self.equilibrium_array_path,
            self.equilibrium_signature_path, comm=self.comm,
            recompute=self.recompute_equilibrium,
        )
        S_equilibrium = Function(self.V_scal)
        S_equilibrium.x.array[:] = S_values
        S_equilibrium.x.scatter_forward()

        S_initial = Function(self.V_S)
        S_initial.interpolate(
            fem.Expression(S_equilibrium, self.V_S.element.interpolation_points)
        )
        self.Y_initial.sub(0).x.array[self.dof_map_S] = S_initial.x.array

        infected_gaussians = self.initial_condition_parameters["infected_gaussians"]

        def infected_initial_condition(x):
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

        self.Y_initial.sub(1).interpolate(infected_initial_condition)
        self.Y_initial.sub(2).interpolate(
            lambda x: np.zeros(x.shape[1], dtype=np.float64)
        )
        self.Y_initial.sub(3).interpolate(
            lambda x: np.zeros(x.shape[1], dtype=np.float64)
        )
        self.Y_initial.x.scatter_forward()

        if bool(self.parameters["disease_spinup"]["enabled"]):
            self._apply_disease_spinup(S_values)

    def _current_prevalence(self):
        """Domain-wide (I + R) / (S + I + R) for the state held in self.Y."""
        infected = self._global_sum(fem.assemble_scalar(self.infected_mass_form))
        total = self._global_sum(fem.assemble_scalar(self.deer_mass_form))
        return infected / total if total > 0.0 else 0.0

    def _apply_disease_spinup(self, susceptible_values):
        """Replace the index-case initial state with a spun-up epizootic.

        Runs the uncontrolled system forward from the index case until
        domain-wide prevalence reaches disease_spinup.target_prevalence, and
        adopts that state as y^0. The control is identically zero throughout,
        so this is the do-nothing trajectory and the resulting state is the
        correct common starting point for the controlled and uncontrolled
        drivers alike. See utilities/disease_spinup.py.
        """
        expected = disease_signature(
            self.domain, self.K_func.x.array, self.lcD_values,
            susceptible_values, self.parameters,
        )

        def compute():
            # Start from the index case and step with v = 0. The zero control
            # matters: state_rhs_form reads v_func, which the optimizer will
            # later overwrite.
            self.Y.x.array[:] = self.Y_initial.x.array
            self.Y.x.scatter_forward()
            self.v_func.x.array[:] = 0.0
            self.v_func.x.scatter_forward()

            def advance():
                self._solve_block(self.state_rhs_form, self.Y_next, "state")
                self.Y.x.array[:] = self.Y_next.x.array
                self.Y.x.scatter_forward()

            return run_disease_spinup(
                advance, self._current_prevalence,
                lambda: self.Y.x.array.copy(),
                self.parameters, comm=self.comm,
            )

        values = disease_initial_state(
            expected, self.Y_initial.x.array.size, compute,
            self.disease_array_path, self.disease_signature_path,
            comm=self.comm, recompute=self.recompute_disease_state,
        )
        self.Y_initial.x.array[:] = values
        self.Y_initial.x.scatter_forward()

    # -------------------------------------------------------- linear algebra

    def _assemble_scalar_vector(self, form):
        """Assemble a V_scal form and return its owned entries as a NumPy array."""
        b = assemble_vector(form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        values = b.array_r[: self.n_owned].copy()
        b.destroy()
        return values

    def _solve_block(self, form, out_function, buffer_key):
        """Solve  B x = assemble(form)  into out_function.

        The right-hand side vector is allocated once and reassembled in place;
        this runs of the order of 10^5 times over an optimization, so a malloc
        and free per call is not free.
        """
        b = self._rhs_buffers.get(buffer_key)
        if b is None:
            b = assemble_vector(form)
            self._rhs_buffers[buffer_key] = b
        else:
            with b.localForm() as local:
                local.set(0.0)
            assemble_vector(b, form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.solver.solve(b, out_function)

    def _global_sum(self, local_value):
        return self.comm.allreduce(local_value, op=MPI.SUM)

    # --------------------------------------------------- control array helpers

    def new_control(self, value=0.0):
        """A control array of shape (n_blocks, n_local), one slab per block."""
        return np.full((self.n_blocks, self.n_local), float(value),
                       dtype=np.float64)

    def control_bytes(self):
        return self.n_blocks * self.n_local * 8

    def control_slab(self, control, step):
        """The control field in force during time step ``step``."""
        return control[self.block_of_step[min(step, self.total_steps - 1)]]

    def new_trajectory(self, path=None):
        """Storage for y^0 .. y^N, in memory or memory-mapped to disk."""
        shape = (self.total_steps + 1, self.Y.x.array.size)
        if path is None:
            return np.empty(shape, dtype=np.float64)
        path = Path(path)
        if self.comm.size > 1:
            # each rank owns a different slice of the mesh, so give each its
            # own backing file rather than letting them overwrite one another
            path = path.with_name(f"{path.stem}_rank{self.comm.rank}{path.suffix}")
        return np.lib.format.open_memmap(
            str(path), mode="w+", dtype=np.float64, shape=shape
        )

    def trajectory_bytes(self):
        return (self.total_steps + 1) * self.Y.x.array.size * 8

    def project_control(self, control):
        """Pointwise projection onto the admissible box [v_min, v_max]."""
        return np.clip(control, self.v_min, self.v_max)

    def inner_product(self, first, second):
        """Space-time L2 inner product, lumped-mass weighted, MPI-reduced.

        Each block is weighted by the span of time it covers, so this stays the
        genuine L2 product of the block-constant controls and the reduced
        gradient below stays its Riesz representative -- which is what keeps
        the Taylor test at second order.
        """
        local = float(np.einsum(
            "bi,bi,i,b->",
            first[:, : self.n_owned],
            second[:, : self.n_owned],
            self.lumped_mass,
            self.block_weights,
        ))
        return self._global_sum(local)

    def norm(self, field):
        return float(np.sqrt(max(0.0, self.inner_product(field, field))))

    def _set_control_slab(self, control, step):
        self.v_func.x.array[:] = control[self.block_of_step[step]]
        self.v_func.x.scatter_forward()

    # --------------------------------------------------------- forward solve

    def forward(self, control, trajectory=None, progress=None, observer=None):
        """Advance the controlled state system and evaluate J(v).

        trajectory (optional) must have shape (total_steps + 1, mixed local
        size); it is filled with y^0 .. y^N. observer (optional) is called as
        observer(step, t) after each state update, with self.Y holding y^step;
        it is the hook the driver uses to write the XDMF series. Returns
        (cost, state_cost, control_cost).
        """
        self.Y.x.array[:] = self.Y_initial.x.array
        self.Y.x.scatter_forward()
        if trajectory is not None:
            trajectory[0] = self.Y.x.array
        if observer is not None:
            observer(0, 0.0)

        state_cost = 0.0
        control_cost = 0.0

        for step in range(self.total_steps):
            self._set_control_slab(control, step)
            control_cost += fem.assemble_scalar(self.control_cost_form)

            self._solve_block(self.state_rhs_form, self.Y_next, "state")

            self.Y.x.array[:] = self.Y_next.x.array
            self.Y.x.scatter_forward()
            if trajectory is not None:
                trajectory[step + 1] = self.Y.x.array

            state_cost += fem.assemble_scalar(self.state_cost_form)

            if observer is not None:
                observer(step + 1, (step + 1) * self.dt)
            if progress is not None:
                progress(step + 1, self.total_steps)

        state_cost = self.dt * self._global_sum(state_cost)
        control_cost = self.dt * self._global_sum(control_cost)
        return state_cost + control_cost, state_cost, control_cost

    # -------------------------------------------------------- backward solve

    def backward(self, control, trajectory, gradient=None, progress=None,
                 observer=None):
        """Backward adjoint sweep; returns the reduced gradient array.

        Solves  B lambda^N = dt*g  and, for n = N-1 down to 1,
        B lambda^n = (dL(y^n, v^n)/dy)^T lambda^(n+1) + dt*g, accumulating
        grad J(v) at slab n as  2*c3*v^n - R^n * lambda_R^(n+1)  on the way
        down. trajectory must hold the states forward() produced for this same
        control. observer (optional) is called as observer(step + 1, t) with
        self.Lam holding lambda^(step+1); times therefore arrive in decreasing
        order.
        """
        if gradient is None:
            gradient = np.empty((self.n_blocks, self.n_local), dtype=np.float64)
        # Accumulated over the steps in each block, so it must start clean.
        gradient[:] = 0.0

        # lambda^N
        self._solve_block(self.terminal_rhs_form, self.Lam, "terminal")

        for step in range(self.total_steps - 1, -1, -1):
            # self.Lam currently holds lambda^(step+1)
            if observer is not None:
                observer(step + 1, (step + 1) * self.dt)
            self.Y.x.array[:] = trajectory[step]
            self.Y.x.scatter_forward()
            self._set_control_slab(control, step)

            # Gradient contribution of this step: assemble the weak form, then
            # divide by the lumped mass to recover the L2 representative. The
            # control is constant across the block containing this step, so by
            # the chain rule the block's derivative is the SUM of its steps'
            # contributions; the average is taken after the loop, against the
            # matching block weight in inner_product.
            raw = self._assemble_scalar_vector(self.gradient_form)
            block = self.block_of_step[step]
            gradient[block, : self.n_owned] += raw / self.lumped_mass

            if step >= 1:
                self._solve_block(self.adjoint_rhs_form, self.Lam_next, "adjoint")
                self.Lam.x.array[:] = self.Lam_next.x.array
                self.Lam.x.scatter_forward()

            if progress is not None:
                progress(self.total_steps - step, self.total_steps)

        # One control value covers block_step_counts[b] steps, so the summed
        # contributions become a per-unit-time density here. Ghosts are filled
        # once at the end rather than once per step -- _scatter_slab borrows
        # v_func, which the adjoint right-hand side also reads.
        gradient[:, : self.n_owned] /= self.block_step_counts[:, None]
        for block in range(self.n_blocks):
            self._scatter_slab(gradient, block)

        return gradient

    def _scatter_slab(self, array, step):
        """Fill the ghost entries of one control-shaped slab."""
        if self.n_local == self.n_owned:
            return
        self.v_func.x.array[: self.n_owned] = array[step, : self.n_owned]
        self.v_func.x.scatter_forward()
        array[step] = self.v_func.x.array
