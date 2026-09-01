"""Compare a DEM against the land-cover raster aligned to it.

Run after align_land_cover.py and before generate_mesh_and_attributes.py. The
aligned land cover should report the DEM's CRS, size, and bounds exactly; a
mismatch in any of the three means the two rasters will not sample onto the
same mesh nodes. The class listing and the zero/nodata count are the other
things worth reading: an unexpected code, or a large zero count, means the NLCD
tile did not cover the whole DEM extent.
"""

from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.warp import transform_bounds


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dem", default="dem.tif",
                        help="Reference DEM (default: dem.tif).")
    parser.add_argument("--land-cover", default="land_cover_aligned.tif",
                        help="Aligned NLCD raster (default: "
                             "land_cover_aligned.tif).")
    return parser


def main():
    args = build_parser().parse_args()

    with rasterio.open(args.dem) as dem:
        print(f"DEM: {args.dem}")
        print(f"  CRS:    {dem.crs}")
        print(f"  Size:   {dem.width} x {dem.height}")
        print(f"  Bounds: {dem.bounds}")
        print(f"  Bounds in lat/lon: "
              f"{transform_bounds(dem.crs, 'EPSG:4326', *dem.bounds)}")
        dem_signature = (dem.crs, dem.width, dem.height, dem.bounds)

    with rasterio.open(args.land_cover) as land_cover:
        data = land_cover.read(1)
        print(f"\nAligned land cover: {args.land_cover}")
        print(f"  CRS:    {land_cover.crs}")
        print(f"  Size:   {land_cover.width} x {land_cover.height}")
        print(f"  Bounds: {land_cover.bounds}")
        print(f"  Classes present:   {np.unique(data)}")
        print(f"  Zero/nodata pixels: {(data == 0).sum()}")
        land_cover_signature = (
            land_cover.crs, land_cover.width, land_cover.height,
            land_cover.bounds,
        )

    if dem_signature == land_cover_signature:
        print("\nGrids match: same CRS, size, and bounds.")
        return 0
    print("\nGRIDS DO NOT MATCH. Re-run align_land_cover.py against this DEM.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
