import rasterio
import numpy as np
from rasterio.windows import Window

marginx = [40, 40]
marginy = [30, 30]

with rasterio.open("dem.tif") as src:
    data = src.read(1)
    nodata = src.nodata
    
    # Find valid data bounds
    valid = ~np.isclose(data, nodata)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    
    # Add a small inset margin to avoid boundary artifacts
    row_min = rows[0] + marginy[0]
    row_max = rows[-1] - marginy[1]
    col_min = cols[0] + marginx[0]
    col_max = cols[-1] - marginx[1]
    
    window = Window(col_min, row_min, 
                    col_max - col_min, 
                    row_max - row_min)
    
    data_crop = src.read(1, window=window)
    transform_crop = src.window_transform(window)
    profile = src.profile.copy()
    profile.update({
        'width': col_max - col_min,
        'height': row_max - row_min,
        'transform': transform_crop
    })

with rasterio.open("dem_cropped.tif", 'w', **profile) as dst:
    dst.write(data_crop, 1)

print(f"Cropped size: {col_max-col_min} x {row_max-row_min}")
print(f"Remaining nodata: {np.isclose(data_crop, nodata).sum()}")
