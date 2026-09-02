"""Sample a categorical land-cover raster at mesh nodes, with the coverage
checks that stop a mis-targeted tile from silently becoming a flat landscape.

The raster is read where the mesh actually is, so it does not have to be
reprojected and resampled onto the DEM grid first. The *query points* are
warped into the raster's own CRS instead, which means the source tile may be
in any CRS at any resolution, and the class each node receives comes from a
single nearest-neighbour lookup rather than two chained ones.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_points, transform_bounds

# NLCD uses class 0 for background/nodata, and class 0 is a *mapped* class in
# parameters.json: it has a real diffusivity and a real carrying capacity. A
# land-cover tile that does not reach the DEM therefore yields a mesh bundle
# that looks entirely valid and a model that runs on a landscape with no
# land-cover structure at all. These thresholds turn that silent degradation
# into a hard failure.
COVERAGE_ERROR_FRACTION = 0.995     # below this, refuse without allow_partial
UNKNOWN_CLASS_WARN_FRACTION = 0.01  # above this, say so
UNKNOWN_CLASS_ERROR_FRACTION = 0.50 # above this, refuse without allow_partial


def read_crs(raster_path):
    """CRS of a raster, as a rasterio CRS object."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise SystemExit(f"{raster_path} has no CRS.")
        return src.crs


def sample_land_cover_at_nodes(land_cover_path, mesh_verts, x_offset, y_offset,
                               mesh_crs):
    """Nearest-neighbour land-cover class at each mesh vertex.

    ``mesh_verts`` are in ``mesh_crs`` -- the DEM's CRS -- de-centred by the
    offsets recorded when the mesh was built. Nodes falling outside the raster
    are returned as class 0, matching NLCD's own nodata convention, so that
    :func:`report_node_class_coverage` can see them.
    """
    xs = np.asarray(mesh_verts[:, 0], dtype=np.float64) + x_offset
    ys = np.asarray(mesh_verts[:, 1], dtype=np.float64) + y_offset

    with rasterio.open(land_cover_path) as src:
        if src.crs is None:
            raise SystemExit(
                f"{land_cover_path} has no CRS; cannot sample it at mesh nodes."
            )
        if src.crs != mesh_crs:
            # Warp the points, not the raster. Feeding DEM-CRS metres straight
            # into a raster in a different CRS samples the wrong place and
            # raises nothing, so this branch is not optional.
            xs, ys = warp_points(mesh_crs, src.crs, xs.tolist(), ys.tolist())
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
        data = src.read(1)
        transform = src.transform

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (
        (rows >= 0) & (rows < data.shape[0])
        & (cols >= 0) & (cols < data.shape[1])
    )
    land_cover = np.zeros(rows.shape, dtype=np.int32)
    land_cover[inside] = data[rows[inside], cols[inside]]

    outside = int((~inside).sum())
    if outside:
        print(f"  {outside} of {land_cover.size} mesh nodes fall outside "
              f"{land_cover_path}; recorded as class 0.")
    return land_cover


def report_node_class_coverage(land_cover, allow_partial=False):
    """Audit the sampled classes for the tile-missed-the-mesh failure.

    Counted over the mesh nodes rather than over a raster grid, so it measures
    the values the model will actually use.
    """
    values, counts = np.unique(land_cover, return_counts=True)
    unknown = int(counts[values == 0].sum()) if (values == 0).any() else 0
    fraction = unknown / land_cover.size if land_cover.size else 0.0

    listed = ", ".join(f"{int(v)}" for v in values if v != 0) or "(none)"
    print(f"Land-cover classes at mesh nodes: {listed}")
    if fraction:
        print(f"  Class 0 (unknown/nodata): {fraction * 100:.2f}% of nodes")

    if fraction >= UNKNOWN_CLASS_ERROR_FRACTION and not allow_partial:
        raise SystemExit(
            f"\n{fraction * 100:.1f}% of mesh nodes received class 0 "
            "(unknown/nodata).\n\n"
            + ("No land-cover classes were transferred at all. "
               if not [v for v in values if v != 0] else
               "Most of the mesh received no land-cover data. ")
            + "The land-cover raster's\nfootprint may reach the DEM while its "
            "valid data does not -- check for a large\nnodata margin, or a "
            "tile clipped to a neighbouring area.\n\n"
            "Pass --allow-partial to proceed anyway."
        )
    if fraction >= UNKNOWN_CLASS_WARN_FRACTION:
        print("  WARNING: class 0 is mapped in parameters.json, so these "
              "nodes will\n  silently become ordinary habitat in the model.")
    return fraction


def check_coverage(land_cover_path, dem_path, allow_partial=False):
    """Verify the land-cover raster actually reaches the DEM.

    The land cover only has to *contain* the DEM extent; it may be far larger,
    in a different CRS, at a different resolution. What it must not be is
    somewhere else, which is easy to do by accident when the DEM tile and the
    land-cover download were chosen at different times.
    """
    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise SystemExit(f"{dem_path} has no CRS; cannot check coverage.")
        dem_box = transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)
    with rasterio.open(land_cover_path) as lc:
        if lc.crs is None:
            raise SystemExit(f"{land_cover_path} has no CRS; cannot check coverage.")
        lc_box = transform_bounds(lc.crs, "EPSG:4326", *lc.bounds)

    overlap_x = max(0.0, min(dem_box[2], lc_box[2]) - max(dem_box[0], lc_box[0]))
    overlap_y = max(0.0, min(dem_box[3], lc_box[3]) - max(dem_box[1], lc_box[1]))
    dem_area = (dem_box[2] - dem_box[0]) * (dem_box[3] - dem_box[1])
    covered = (overlap_x * overlap_y) / dem_area if dem_area > 0 else 0.0

    def box(b):
        return f"{b[0]:10.4f} {b[1]:9.4f} {b[2]:10.4f} {b[3]:9.4f}"

    print("Footprints in lon/lat (W S E N):")
    print(f"  DEM        {box(dem_box)}")
    print(f"  Land cover {box(lc_box)}")
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
        f"\n{land_cover_path} {problem}.\n\n"
        f"  DEM        {box(dem_box)}\n"
        f"  Land cover {box(lc_box)}\n\n"
        "Mesh nodes outside the land-cover raster become class 0 (nodata), "
        "and class 0\nis a mapped class in parameters.json -- so the mesh "
        "would build and the model\nwould run, on a landscape with no "
        "land-cover structure at all.\n\n"
        "Download a land-cover tile from https://www.mrlc.gov/viewer/ covering "
        "at least:\n"
        f"    {wanted[0]:.4f} W to {wanted[2]:.4f} W, "
        f"{wanted[1]:.4f} N to {wanted[3]:.4f} N\n\n"
        "It may be much larger than that, and in any CRS or resolution; it is "
        "sampled\ndirectly at the mesh nodes and never resampled onto the DEM "
        "grid. It just has\nto contain the DEM.\n\n"
        "Pass --allow-partial to proceed anyway and accept class-0 regions."
    )
    if allow_partial:
        print(message.replace("\n", "\n  "))
        print("\n  --allow-partial given; continuing.")
        return covered
    raise SystemExit(message)
