import rasterio
from rasterio.warp import (
    calculate_default_transform, reproject, transform_bounds, Resampling,
)
import numpy as np
import argparse
from pathlib import Path

# NLCD uses class 0 for background/nodata, and align_nlcd_to_dem writes 0 into
# every output pixel that falls outside the source raster. An NLCD tile that
# does not reach the DEM therefore produces a perfectly valid-looking GeoTIFF
# that is entirely class 0, which the mesher will happily consume: class 0 maps
# to a real diffusivity and a real carrying capacity in parameters.json, so the
# model runs on a uniform landscape with no land-cover structure at all. These
# thresholds turn that silent degradation into a hard failure.
COVERAGE_ERROR_FRACTION = 0.995     # below this, refuse without --allow-partial
UNKNOWN_CLASS_WARN_FRACTION = 0.01  # above this, say so
UNKNOWN_CLASS_ERROR_FRACTION = 0.50 # above this, refuse without --allow-partial

def reproject_nlcd(input_path, output_path, target_crs="EPSG:32613"):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
    
    print(f"Reprojected NLCD saved to {output_path}")
    with rasterio.open(output_path) as src:
        print(f"New bounds: {src.bounds}")
        print(f"New size: {src.width} x {src.height}")


def align_nlcd_to_dem(nlcd_path, dem_path, output_path):
    """Reproject and resample NLCD to exactly match DEM grid."""
    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs
        dem_transform = dem.transform
        dem_width = dem.width
        dem_height = dem.height
        dem_profile = dem.profile

    with rasterio.open(nlcd_path) as nlcd:
        profile = dem_profile.copy()
        profile.update({
            'dtype': 'int16',
            'nodata': 0
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(nlcd, 1),
                destination=rasterio.band(dst, 1),
                src_transform=nlcd.transform,
                src_crs=nlcd.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.nearest  # nearest neighbor for categorical data
            )
    print(f"Aligned NLCD saved to {output_path}")


def check_coverage(nlcd_path, dem_path, allow_partial=False):
    """Verify the land-cover raster actually reaches the DEM, before warping.

    The NLCD only has to *contain* the DEM extent; it may be far larger, in a
    different CRS, at a different resolution. What it must not be is somewhere
    else, which is easy to do by accident when the DEM tile and the land-cover
    download were chosen at different times.
    """
    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise SystemExit(f"{dem_path} has no CRS; cannot check coverage.")
        dem_box = transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)
    with rasterio.open(nlcd_path) as nlcd:
        if nlcd.crs is None:
            raise SystemExit(f"{nlcd_path} has no CRS; cannot check coverage.")
        nlcd_box = transform_bounds(nlcd.crs, "EPSG:4326", *nlcd.bounds)

    overlap_x = max(0.0, min(dem_box[2], nlcd_box[2]) - max(dem_box[0], nlcd_box[0]))
    overlap_y = max(0.0, min(dem_box[3], nlcd_box[3]) - max(dem_box[1], nlcd_box[1]))
    dem_area = (dem_box[2] - dem_box[0]) * (dem_box[3] - dem_box[1])
    covered = (overlap_x * overlap_y) / dem_area if dem_area > 0 else 0.0

    def box(b):
        return f"{b[0]:10.4f} {b[1]:9.4f} {b[2]:10.4f} {b[3]:9.4f}"

    print("Footprints in lon/lat (W S E N):")
    print(f"  DEM        {box(dem_box)}")
    print(f"  Land cover {box(nlcd_box)}")
    print(f"  DEM covered by land cover: {covered * 100:.2f}%")

    if covered >= COVERAGE_ERROR_FRACTION:
        if covered < 1.0:
            print("  NOTE: coverage is not quite complete; expect a thin "
                  "border of class 0 along one edge.")
        return covered

    margin_x = 0.05 * (dem_box[2] - dem_box[0])
    margin_y = 0.05 * (dem_box[3] - dem_box[1])
    wanted = (dem_box[0] - margin_x, dem_box[1] - margin_y,
              dem_box[2] + margin_x, dem_box[3] + margin_y)
    problem = ("does not overlap the DEM at all" if covered == 0
               else f"covers only {covered * 100:.1f}% of the DEM")
    message = (
        f"\n{nlcd_path} {problem}.\n\n"
        f"  DEM        {box(dem_box)}\n"
        f"  Land cover {box(nlcd_box)}\n\n"
        "Every DEM pixel outside the land-cover raster becomes class 0 "
        "(nodata),\nand class 0 is a mapped class in parameters.json -- so the "
        "mesh would build\nand the model would run, on a landscape with no "
        "land-cover structure at all.\n\n"
        "Download a land-cover tile from https://www.mrlc.gov/viewer/ covering "
        "at least:\n"
        f"    {wanted[0]:.4f} W to {wanted[2]:.4f} W, "
        f"{wanted[1]:.4f} N to {wanted[3]:.4f} N\n\n"
        "It may be much larger than that, and in any CRS or resolution; this "
        "script\nclips and resamples it onto the DEM grid. It just has to "
        "contain the DEM.\n\n"
        "Pass --allow-partial to proceed anyway and accept class-0 regions."
    )
    if allow_partial:
        print(message.replace("\n", "\n  "))
        print("\n  --allow-partial given; continuing.")
        return covered
    raise SystemExit(message)


def report_class_coverage(aligned_path, allow_partial=False):
    """Check what actually landed in the aligned raster.

    The footprint test above catches the common case, but not an NLCD whose
    bounding box overlaps while its valid data does not -- a tile with a large
    nodata margin, or one clipped to a neighbouring state.
    """
    with rasterio.open(aligned_path) as src:
        data = src.read(1)
    values, counts = np.unique(data, return_counts=True)
    unknown = int(counts[values == 0].sum()) if (values == 0).any() else 0
    fraction = unknown / data.size

    listed = ", ".join(f"{int(v)}" for v in values if v != 0) or "(none)"
    print(f"Land-cover classes present: {listed}")
    if fraction:
        print(f"  Class 0 (unknown/nodata): {fraction * 100:.2f}% of pixels")

    if fraction >= UNKNOWN_CLASS_ERROR_FRACTION and not allow_partial:
        raise SystemExit(
            f"\n{aligned_path} is {fraction * 100:.1f}% class 0 "
            "(unknown/nodata).\n\n"
            + ("No land-cover classes were transferred at all. "
               if not [v for v in values if v != 0] else
               "Most of the DEM received no land-cover data. ")
            + "The land-cover\nraster's bounding box reaches the DEM, but its "
            "valid data does not -- check\nfor a large nodata margin, or a tile "
            "clipped to a neighbouring area.\n\n"
            "Pass --allow-partial to proceed anyway."
        )
    if fraction >= UNKNOWN_CLASS_WARN_FRACTION:
        print("  WARNING: class 0 is mapped in parameters.json, so these "
              "cells will\n  silently become ordinary habitat in the model.")
    return fraction


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproject a categorical land-cover raster and align it to a DEM grid."
    )
    parser.add_argument("--input-land-cover", default="land_cover.tif",
                        help="Source categorical land-cover raster (default: land_cover.tif).")
    parser.add_argument("--reprojected-land-cover", default="land_cover_reprojected.tif",
                        help="Intermediate reprojected raster.")
    parser.add_argument("--dem", default="dem.tif",
                        help="Reference DEM whose grid will be copied (default: dem.tif).")
    parser.add_argument("--output", default="land_cover_aligned.tif",
                        help="Final aligned land-cover raster.")
    parser.add_argument("--target-crs", default=None,
                        help="Target CRS for the intermediate raster. Defaults to the DEM CRS.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Proceed even if the land-cover raster does not "
                             "cover the DEM. The uncovered area becomes class "
                             "0, which is a mapped class, so the model will "
                             "run on landscape with no land-cover structure.")
    return parser


def main():
    args = build_parser().parse_args()
    for output_path in (args.reprojected_land_cover, args.output):
        Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    # Checked before the warp, which is the expensive part, and before anything
    # is written.
    check_coverage(args.input_land_cover, args.dem, args.allow_partial)

    target_crs = args.target_crs
    if target_crs is None:
        with rasterio.open(args.dem) as dem:
            if dem.crs is None:
                raise ValueError("The DEM has no CRS; provide --target-crs explicitly.")
            target_crs = dem.crs.to_string()

    reproject_nlcd(args.input_land_cover, args.reprojected_land_cover, target_crs)
    align_nlcd_to_dem(args.reprojected_land_cover, args.dem, args.output)
    report_class_coverage(args.output, args.allow_partial)


if __name__ == "__main__":
    main()
