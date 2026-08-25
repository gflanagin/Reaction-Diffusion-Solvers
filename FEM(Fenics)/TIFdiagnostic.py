import numpy as np
import rasterio
import struct


# --- Diagnostic ---
input_tif = "region2.tif"  # <-- your file path

with rasterio.open(input_tif) as src:
    data = src.read(1).astype(np.float64)
    nodata = src.nodata
    transform = src.transform
    rows, cols = data.shape
    crs = src.crs
    bounds = src.bounds
    print(f"Resolution: {src.res}")

print(f"Size: {cols} x {rows}")
print(f"CRS: {crs}")
print(f"Bounds: {bounds}")
print(f"Nodata value: {nodata}")
print(f"Data min: {np.nanmin(data):.2f}")
print(f"Data max: {np.nanmax(data):.2f}")
print(f"Data mean: {np.nanmean(data):.2f}")
print(f"NaN count: {np.isnan(data).sum()}")
print(f"Transform: {transform}")

# Also check the STL file directly
stl_file = "terrain.stl"
with open(stl_file, 'rb') as f:
    header = f.read(80)
    num_triangles = struct.unpack('<I', f.read(4))[0]
    print(f"\nSTL header: {header[:30]}")
    print(f"STL triangle count: {num_triangles}")
    if num_triangles > 0:
        # Read first triangle
        normal = struct.unpack('<fff', f.read(12))
        v1 = struct.unpack('<fff', f.read(12))
        v2 = struct.unpack('<fff', f.read(12))
        v3 = struct.unpack('<fff', f.read(12))
        print(f"First triangle normal: {normal}")
        print(f"First triangle v1: {v1}")
        print(f"First triangle v2: {v2}")
        print(f"First triangle v3: {v3}")


with rasterio.open(input_tif) as src:
    data = src.read(1).astype(np.float64)
    nodata = src.nodata

print(f"Nodata value: {nodata}")
print(f"Nodata count: {np.sum(np.isclose(data, nodata))}")
print(f"Total pixels: {data.size}")

# Check for any internal nodata (not just border)
mask = np.isclose(data, nodata)
# Erode the border to find interior nodata
from scipy.ndimage import binary_erosion
border = mask.copy()
border[1:-1, 1:-1] = False
interior_nodata = mask & ~border
print(f"Interior nodata pixels: {interior_nodata.sum()}")