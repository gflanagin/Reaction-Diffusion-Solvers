"""Check that a land-cover raster actually covers a DEM, before meshing.

The land cover is sampled directly at the mesh nodes by
generate_mesh_and_attributes.py, so it is never reprojected or resampled onto
the DEM grid and there is no aligned intermediate raster to inspect. What can
still go wrong is the tile being somewhere else, which is easy to do by
accident when the DEM and the land-cover download were chosen at different
times. That is what this checks.

generate_mesh_and_attributes.py runs the same check itself before it starts
meshing, so this script is only needed to test a download up front, without
committing to a mesh run.
"""

import argparse

from pathlib import Path
import sys

UTILITIES_DIR = Path(__file__).resolve().parent / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from land_cover_sampling import check_coverage  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check that a land-cover raster covers a DEM's footprint."
    )
    parser.add_argument("--land-cover", default="land_cover.tif",
                        help="Categorical land-cover raster (default: "
                             "land_cover.tif). May be in any CRS at any "
                             "resolution.")
    parser.add_argument("--dem", default="dem.tif",
                        help="Reference DEM (default: dem.tif).")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Report instead of failing when the land-cover "
                             "raster does not cover the DEM.")
    return parser


def main():
    args = build_parser().parse_args()
    check_coverage(args.land_cover, args.dem, args.allow_partial)
    print("\nLand cover covers the DEM; ready to mesh.")


if __name__ == "__main__":
    main()
