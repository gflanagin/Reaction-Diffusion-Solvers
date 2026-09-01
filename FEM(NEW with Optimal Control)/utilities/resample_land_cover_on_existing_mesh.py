from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI
import numpy as np
import rasterio
from rasterio.transform import rowcol
import argparse
from pathlib import Path
import sys

UTILITIES_DIR = Path(__file__).resolve().parent
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import (  # noqa: E402
    DEFAULT_PARAMETERS,
    land_cover_to_diffusivity,
    load_parameters,
)

def sample_nlcd_at_nodes(nlcd_aligned_path, mesh_verts, x_offset, y_offset):
    with rasterio.open(nlcd_aligned_path) as src:
        data = src.read(1)
        transform = src.transform

    xs = mesh_verts[:, 0] + x_offset
    ys = mesh_verts[:, 1] + y_offset

    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(rows, 0, data.shape[0] - 1)
    cols = np.clip(cols, 0, data.shape[1] - 1)

    land_cover = data[rows, cols].astype(np.int32)
    print(f"Unique land cover classes at mesh nodes: {np.unique(land_cover)}")
    return land_cover

def build_parser():
    parser = argparse.ArgumentParser(
        description="Regenerate nodal land-cover arrays for an existing mesh."
    )
    parser.add_argument("--mesh", default="terrain.msh")
    parser.add_argument("--land-cover", default="land_cover_aligned.tif")
    parser.add_argument("--coord-offsets", default="coord_offsets.npy")
    parser.add_argument("--classes-output", default="land_cover_classes.npy")
    parser.add_argument("--diffusivity-output", default="land_cover_diffusivity.npy")
    parser.add_argument("--parameters", default=str(DEFAULT_PARAMETERS))
    return parser


def main():
    args = build_parser().parse_args()
    spatial_parameters = load_parameters(args.parameters)

    mesh_data = read_from_msh(args.mesh, MPI.COMM_WORLD, gdim=3)
    domain = mesh_data.mesh
    coords = domain.geometry.x
    x_offset, y_offset = np.load(args.coord_offsets)

    land_cover = sample_nlcd_at_nodes(
        args.land_cover, coords, x_offset, y_offset
    )
    diffusivity = land_cover_to_diffusivity(land_cover, spatial_parameters)

    np.save(args.diffusivity_output, diffusivity)
    np.save(args.classes_output, land_cover.astype(np.int32))

    water_value = float(spatial_parameters["land_cover"]["diffusivity"].get("11", 0.0))
    print(f"Saved! min={diffusivity.min():.4g}, max={diffusivity.max():.4g}")
    print(f"Water nodes (barrier): {(diffusivity == water_value).sum()}")


if __name__ == "__main__":
    main()
