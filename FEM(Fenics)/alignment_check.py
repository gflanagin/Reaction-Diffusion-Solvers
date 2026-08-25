import rasterio
from rasterio.warp import transform_bounds
import numpy as np

with rasterio.open("region3_cropped.tif") as src:
    bounds_latlon = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    print(f"DEM bounds in lat/lon: {bounds_latlon}")
    # (west, south, east, north)

with rasterio.open("nlcd_aligned.tif") as src:
    data = src.read(1)
    print(f"CRS: {src.crs}")
    print(f"Size: {src.width} x {src.height}")
    print(f"Bounds: {src.bounds}")
    print(f"Unique classes: {np.unique(data)}")
    print(f"Nodata/zero pixels: {(data == 0).sum()}")

with rasterio.open("region3_cropped.tif") as dem:
    print(f"\nDEM CRS: {dem.crs}")
    print(f"DEM Size: {dem.width} x {dem.height}")
    print(f"DEM Bounds: {dem.bounds}")
