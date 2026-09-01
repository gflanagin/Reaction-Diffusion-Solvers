"""Load and validate model parameters shared by meshing, the PDE solver, and the
optimal-control driver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMETERS = ROOT / "parameters.json"


# Defaults for the optimal-control block. A parameter file written before the
# control work existed (for example an ``effective_parameters.json`` emitted by
# an older mesh-generation run) is still accepted; the missing keys are filled
# in from this table so the state solver and the control driver can read the
# same file.
OPTIMAL_CONTROL_DEFAULTS = {
    # Weights follow the balance recipe of the write-up. With W settling near
    # rho*R/delta = 2*R, a weight of 0.5 on W puts the environmental term on the
    # same footing as the shedding term, and c3 = 2 makes c3*v^2 at half effort
    # comparable to c1*R at R of order 0.5 deer/km^2, the scale of the endemic
    # shedding density. control_maximum is a per-capita removal rate on R, so
    # 1.0/yr already means 63% of the shedding class removed over a year -- at
    # or beyond the ceiling of a real agency cull program. Provenance for every
    # value lives in the "_notes" blocks of parameters.json and under
    # "Parameter Values and Their Provenance" in CWD_optimal_control.tex.
    #
    # cost_shedding weights R, the shedding class, which is also what the
    # control acts on. It was called cost_infected while the control acted on
    # I; _validate_optimal_control rejects the old key outright rather than
    # letting it fall through to this default.
    "cost_shedding": 1.0,
    "cost_environment": 0.5,
    "cost_control": 2.0,
    # Elastic-net L1 weight on the control. Defaults to 0.0, so a file written
    # before this existed reproduces the pure-quadratic objective exactly.
    "cost_control_l1": 0.0,
    # Length in years of the interval over which the culling effort is held
    # constant. null means one value per time step, which is what this workflow
    # did before the block control existed. See the "_notes" block in
    # parameters.json for why 1.0 is the default.
    "control_block_years": 1.0,
    # Optimizer: "lbfgs" (projected quasi-Newton via SciPy's L-BFGS-B) or
    # "projected-gradient" (the original steepest descent with Armijo).
    "optimizer": "lbfgs",
    # L-BFGS-B correction pairs. Memory is about (2*maxcor + 5) control-sized
    # arrays, so raise it only when the control is small.
    "lbfgs_memory": 10,
    "control_minimum": 0.0,
    "control_maximum": 1.0,
    "initial_control": 0.0,
    "max_iterations": 40,
    "initial_step_size": 1.0,
    "armijo_sufficient_decrease": 1.0e-4,
    "armijo_backtrack_factor": 0.5,
    "armijo_step_growth": 2.0,
    "armijo_max_backtracks": 25,
    "minimum_step_size": 1.0e-12,
    "gradient_tolerance": 1.0e-8,
    "relative_cost_tolerance": 1.0e-8,
}


# Defaults for the disease-free spin-up block. As with the control defaults
# above, a parameter file written before the spin-up existed is still accepted;
# the missing keys are filled in from this table. See
# utilities/susceptible_spinup.py for what the step does and why.
SUSCEPTIBLE_SPINUP_DEFAULTS = {
    "enabled": True,
    # A cap, not a target: the solve stops as soon as drift_tolerance is met,
    # which on a typical bundle happens well before this. It is generous because
    # overshooting costs one cheap scalar solve, while stopping early leaves a
    # demographic transient in every run built on the cached field.
    "max_duration_years": 60.0,
    # null means "use time.time_step_years". Only raise it if the spin-up is a
    # noticeable share of run time; the explicit growth term is stable for
    # dt*r << 1, so there is room, but the diffusion is only first-order
    # accurate in time and a coarse step biases the smoothed profile.
    "time_step_years": None,
    # Stop when max|dS/dt| falls below this fraction of peak K per year. 1e-5
    # means the field moves by less than 0.001% of peak density per year: at
    # K_0 = 10 deer/km^2 that is under 1e-4 deer/km^2/yr, i.e. under a
    # five-hundredth of a deer per km^2 over the whole 20-year horizon.
    "drift_tolerance": 1.0e-5,
}


def _validate_susceptible_spinup(parameters):
    """Fill in spin-up defaults in place, then range-check them."""
    spinup = dict(SUSCEPTIBLE_SPINUP_DEFAULTS)
    spinup.update(parameters.get("susceptible_spinup", {}))
    parameters["susceptible_spinup"] = spinup

    if float(spinup["max_duration_years"]) <= 0:
        raise ValueError(
            "susceptible_spinup.max_duration_years must be positive"
        )
    if spinup["time_step_years"] is not None:
        if float(spinup["time_step_years"]) <= 0:
            raise ValueError(
                "susceptible_spinup.time_step_years must be positive, or null "
                "to reuse time.time_step_years"
            )
    if float(spinup["drift_tolerance"]) <= 0:
        raise ValueError("susceptible_spinup.drift_tolerance must be positive")
    return spinup


def _validate_optimal_control(parameters):
    """Fill in optimal-control defaults in place, then range-check them."""
    supplied = parameters.get("optimal_control", {})

    # The control used to act on I and the running cost used to weight I. Both
    # now act on R, the shedding class. A file still carrying the old key would
    # otherwise be accepted in silence, with cost_shedding quietly taking its
    # default -- a wrong answer rather than an error, so refuse it.
    if "cost_infected" in supplied:
        raise ValueError(
            "optimal_control.cost_infected is no longer used: the culling "
            "control and the running cost both act on the shedding class R "
            "(code name D), not on I. Rename the key to "
            "'cost_shedding'. Note that its meaning changed with it -- the "
            "endemic density of R is roughly half that of I, so a weight "
            "calibrated against I will need rescaling; see the '_notes' block "
            "in parameters.json."
        )

    control = dict(OPTIMAL_CONTROL_DEFAULTS)
    control.update(supplied)
    parameters["optimal_control"] = control

    for name in ("cost_shedding", "cost_environment"):
        if float(control[name]) < 0:
            raise ValueError(f"optimal_control.{name} cannot be negative")
    if float(control["cost_control"]) < 0:
        raise ValueError("optimal_control.cost_control cannot be negative")
    if float(control["cost_control_l1"]) < 0:
        raise ValueError("optimal_control.cost_control_l1 cannot be negative")
    if float(control["cost_control"]) + float(control["cost_control_l1"]) <= 0:
        raise ValueError(
            "at least one of optimal_control.cost_control (quadratic) and "
            "optimal_control.cost_control_l1 must be positive; with neither, "
            "the reduced objective has no minimizer that penalizes effort at "
            "all and the optimizer will simply saturate the control bound"
        )
    if float(control["control_minimum"]) < 0:
        raise ValueError("optimal_control.control_minimum cannot be negative")
    if float(control["control_maximum"]) <= float(control["control_minimum"]):
        raise ValueError(
            "optimal_control.control_maximum must exceed control_minimum"
        )
    initial_control = float(control["initial_control"])
    if not (
        float(control["control_minimum"])
        <= initial_control
        <= float(control["control_maximum"])
    ):
        raise ValueError(
            "optimal_control.initial_control must lie in "
            "[control_minimum, control_maximum]"
        )
    block_years = control["control_block_years"]
    if block_years is not None and float(block_years) <= 0:
        raise ValueError(
            "optimal_control.control_block_years must be positive, or null to "
            "give the control one value per time step"
        )
    if control["optimizer"] not in ("lbfgs", "projected-gradient"):
        raise ValueError(
            "optimal_control.optimizer must be 'lbfgs' or "
            f"'projected-gradient', not {control['optimizer']!r}"
        )
    if int(control["lbfgs_memory"]) < 1:
        raise ValueError("optimal_control.lbfgs_memory must be at least 1")
    if int(control["max_iterations"]) < 1:
        raise ValueError("optimal_control.max_iterations must be at least 1")
    if float(control["initial_step_size"]) <= 0:
        raise ValueError("optimal_control.initial_step_size must be positive")
    if not 0 < float(control["armijo_sufficient_decrease"]) < 1:
        raise ValueError(
            "optimal_control.armijo_sufficient_decrease must be in (0, 1)"
        )
    if not 0 < float(control["armijo_backtrack_factor"]) < 1:
        raise ValueError(
            "optimal_control.armijo_backtrack_factor must be in (0, 1)"
        )
    if float(control["armijo_step_growth"]) < 1:
        raise ValueError("optimal_control.armijo_step_growth must be at least 1")
    if int(control["armijo_max_backtracks"]) < 1:
        raise ValueError(
            "optimal_control.armijo_max_backtracks must be at least 1"
        )
    if float(control["minimum_step_size"]) <= 0:
        raise ValueError("optimal_control.minimum_step_size must be positive")
    if float(control["gradient_tolerance"]) < 0:
        raise ValueError("optimal_control.gradient_tolerance cannot be negative")
    if float(control["relative_cost_tolerance"]) < 0:
        raise ValueError(
            "optimal_control.relative_cost_tolerance cannot be negative"
        )
    return control


def _validate_reaction(parameters):
    """Range-check the reaction block and catch parameter files that carry mu."""
    reaction = parameters["reaction"]

    # The model has no background-mortality term: sigma is the only exit from I
    # and alpha the only exit from R. A file that still carries mu is not
    # merely carrying an unused key -- its direct_transmission_rate and
    # environmental_transmission_rate were back-solved from a target R0 whose
    # formula contains mu, so silently dropping the term would leave the betas
    # too large and raise the effective R0 from 2.0 to about 3.7. Refuse it.
    if "background_mortality_rate" in reaction:
        raise ValueError(
            "reaction.background_mortality_rate is present, but this model "
            "carries no background-mortality term: sigma is the only exit "
            "from I and alpha the only exit from R. Its presence means the "
            "file was written for the variant that did model non-CWD "
            "mortality, whose direct_transmission_rate and "
            "environmental_transmission_rate were calibrated with mu in the "
            "R0 formula. Dropping mu while keeping those rates raises the "
            "effective R0 from 2.0 to roughly 3.7. Remove the key and reset "
            "the two transmission rates to beta_1 = R0_dir*alpha/K0 and "
            "beta_2 = R0_env*delta*alpha/(K0*rho), i.e. 0.0981 and 0.04905 at "
            "the shipped alpha, delta, rho and K0; see the '_notes' block in "
            "parameters.json."
        )

    positive = (
        "clinical_removal_rate",
        "infection_to_clinical_rate",
        "environmental_decay_rate",
    )
    for name in positive:
        if float(reaction[name]) <= 0:
            raise ValueError(f"reaction.{name} must be positive")

    nonnegative = (
        "direct_transmission_rate",
        "intrinsic_growth_rate",
        "environmental_shedding_rate",
        "environmental_transmission_rate",
    )
    for name in nonnegative:
        if float(reaction[name]) < 0:
            raise ValueError(f"reaction.{name} cannot be negative")

    return reaction


def _load_json(path):
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_parameters(path=DEFAULT_PARAMETERS):
    parameters = _load_json(path)
    mesh = parameters["mesh"]
    tensor = parameters["diffusion_tensor"]
    land_cover = parameters["land_cover"]
    time = parameters["time"]
    infected_gaussians = parameters["initial_conditions"]["infected_gaussians"]

    if float(time["final_time_years"]) <= 0:
        raise ValueError("time.final_time_years must be positive")
    if float(time["time_step_years"]) <= 0:
        raise ValueError("time.time_step_years must be positive")

    if not isinstance(infected_gaussians, list):
        raise ValueError("initial_conditions.infected_gaussians must be a list")
    for index, gaussian in enumerate(infected_gaussians):
        center = gaussian.get("center")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].center "
                "must contain [x, y]"
            )
        if not all(np.isfinite(float(value)) for value in center):
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].center "
                "must contain finite values"
            )
        if not np.isfinite(float(gaussian["mean"])) or float(gaussian["mean"]) < 0:
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].mean "
                "must be finite and nonnegative"
            )
        if not np.isfinite(float(gaussian["standard_deviation"])) or float(
            gaussian["standard_deviation"]
        ) <= 0:
            raise ValueError(
                f"initial_conditions.infected_gaussians[{index}].standard_deviation "
                "must be finite and positive"
            )

    if int(mesh["downsample"]) < 1:
        raise ValueError("mesh.downsample must be at least 1")
    if float(mesh["smoothing_sigma"]) < 0:
        raise ValueError("mesh.smoothing_sigma cannot be negative")
    if float(mesh["hmax"]) <= 0:
        raise ValueError("mesh.hmax must be positive")
    if not 0 < float(mesh["min_land_cover_squish"]) <= 1:
        raise ValueError("mesh.min_land_cover_squish must be in (0, 1]")
    if not 0 < float(tensor["isotropy"]) <= 1:
        raise ValueError("diffusion_tensor.isotropy must be in (0, 1]")
    if float(tensor["kappa"]) <= 0:
        raise ValueError("diffusion_tensor.kappa must be positive")
    if float(tensor["activation_steepness"]) <= 0:
        raise ValueError("diffusion_tensor.activation_steepness must be positive")
    if not 0 <= float(tensor["activation_cosine_threshold"]) <= 1:
        raise ValueError(
            "diffusion_tensor.activation_cosine_threshold must be in [0, 1]"
        )
    if float(land_cover["carrying_capacity_base"]) < 0:
        raise ValueError("land_cover.carrying_capacity_base cannot be negative")

    class_codes = set(land_cover["class_names"])
    for mapping_name in ("diffusivity", "carrying_capacity_fractions"):
        missing = set(land_cover[mapping_name]) - class_codes
        if missing:
            raise ValueError(
                f"land_cover.class_names is missing codes used by {mapping_name}: "
                + ", ".join(sorted(missing))
            )

    _validate_reaction(parameters)
    _validate_susceptible_spinup(parameters)
    _validate_optimal_control(parameters)

    return parameters


def land_cover_to_diffusivity(land_cover_array, spatial_parameters):
    config = spatial_parameters["land_cover"]
    mapping = {int(code): float(value) for code, value in config["diffusivity"].items()}
    result = np.full(
        np.asarray(land_cover_array).shape,
        float(config["diffusivity_default"]),
        dtype=np.float64,
    )
    for code, value in mapping.items():
        result[np.asarray(land_cover_array) == code] = value
    return result


def land_cover_to_carrying_capacity(land_cover_array, spatial_parameters):
    config = spatial_parameters["land_cover"]
    base = float(config["carrying_capacity_base"])
    fractions = {
        int(code): float(value)
        for code, value in config["carrying_capacity_fractions"].items()
    }
    result = np.full(
        np.asarray(land_cover_array).shape,
        base * float(config["carrying_capacity_fraction_default"]),
        dtype=np.float64,
    )
    for code, fraction in fractions.items():
        result[np.asarray(land_cover_array) == code] = base * fraction
    return result
