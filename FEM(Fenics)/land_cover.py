import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

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


reproject_nlcd("NLCD.tiff", "nlcd_utm.tif")

align_nlcd_to_dem("nlcd_utm.tif", "region3.tif", "nlcd_aligned.tif")