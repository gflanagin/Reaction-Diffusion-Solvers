"""Load and validate model parameters shared by meshing and the PDE solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMETERS = ROOT / "parameters.json"


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
