import numpy as np
import rasterio
from rasterio.transform import xy
import struct
import sys
import os

def tif_to_stl(input_tif, output_stl, downsample=1):
    """
    Convert a DEM GeoTIFF to an STL file for Gmsh.
    
    input_tif   : path to your .tif DEM file
    output_stl  : path for the output .stl file
    downsample  : keep every Nth pixel (1 = full res, 2 = half res, etc.)
    """
    print(f"Reading {input_tif}...")
    with rasterio.open(input_tif) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        rows, cols = data.shape

    print(f"Raster size: {cols} x {rows} pixels")

    # Replace nodata values with NaN
    if nodata is not None:
        data[data == nodata] = np.nan

    # Fill any remaining NaNs with the mean elevation (simple gap fill)
    nan_mask = np.isnan(data)
    if nan_mask.any():
        print(f"Filling {nan_mask.sum()} nodata pixels with mean elevation...")
        data[nan_mask] = np.nanmean(data)

    # Downsample if requested
    data = data[::downsample, ::downsample]
    rows, cols = data.shape
    print(f"After downsampling: {cols} x {rows} pixels")

    # Build X, Y coordinate grids
    # Sample pixel centers at the downsampled positions
    col_indices = np.arange(0, cols * downsample, downsample)
    row_indices = np.arange(0, rows * downsample, downsample)
    
    # Get real-world coordinates using the raster transform
    xs, _ = xy(transform, np.zeros_like(col_indices), col_indices)
    _, ys = xy(transform, row_indices, np.zeros_like(row_indices))
    
    X, Y = np.meshgrid(xs, ys)
    Z = data

    # Center coordinates around origin (Gmsh works best near origin)
    x_off = X.mean()
    y_off = Y.mean()
    X -= x_off
    Y -= y_off

    print(f"Coordinate offsets applied: X={x_off:.2f}, Y={y_off:.2f}")
    print(f"Z range: {np.nanmin(Z):.2f} to {np.nanmax(Z):.2f}")

    # Build triangles from the grid
    # Each quad cell -> 2 triangles
    print("Building triangles...")
    triangles = []

    for r in range(rows - 1):
        for c in range(cols - 1):
            # Four corners of the quad
            p00 = (X[r,   c],   Y[r,   c],   Z[r,   c])
            p10 = (X[r+1, c],   Y[r+1, c],   Z[r+1, c])
            p01 = (X[r,   c+1], Y[r,   c+1], Z[r,   c+1])
            p11 = (X[r+1, c+1], Y[r+1, c+1], Z[r+1, c+1])

            # Triangle 1: top-left, bottom-left, bottom-right
            triangles.append((p00, p10, p11))
            # Triangle 2: top-left, bottom-right, top-right
            triangles.append((p00, p11, p01))

    print(f"Total triangles: {len(triangles)}")

    # Write binary STL
    print(f"Writing {output_stl}...")
    with open(output_stl, 'wb') as f:
        # 80-byte header
        f.write(b'DEM to STL export' + b' ' * 62)
        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        for (p0, p1, p2) in triangles:
            # Compute normal vector
            v1 = np.array(p1) - np.array(p0)
            v2 = np.array(p2) - np.array(p0)
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len

            # Write normal + 3 vertices + attribute byte count
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *p0))
            f.write(struct.pack('<fff', *p1))
            f.write(struct.pack('<fff', *p2))
            f.write(struct.pack('<H', 0))

    print(f"Done! STL written to: {output_stl}")
    size_mb = os.path.getsize(output_stl) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


# --- Run it ---
input_tif  = "region.tif"   # <-- change this to your file path
output_stl = "terrain.stl"    # <-- output file name
downsample = 1                 # start with 1; increase