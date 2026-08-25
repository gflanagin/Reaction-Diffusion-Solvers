from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI
import numpy as np
import rasterio
from rasterio.transform import rowcol

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

def land_cover_to_diffusivity(land_cover_array):
    lc_map = {
        0:  1.0,   # unknown/nodata -- don't block
        11: 0.0001,   # Open water -- barrier
        12: 0.3,   # Perennial ice/snow
        21: 0.7,   # Developed open space
        22: 0.4,   # Developed low intensity
        23: 0.2,   # Developed medium intensity
        24: 0.05,  # Developed high intensity
        31: 0.9,   # Barren rock
        41: 0.5,   # Deciduous forest
        42: 0.4,   # Evergreen forest
        43: 0.45,  # Mixed forest
        52: 0.8,   # Shrub/scrub
        71: 0.95,  # Grassland/herbaceous
        81: 0.85,  # Pasture/hay
        82: 0.8,   # Cultivated crops
        90: 0.2,   # Woody wetlands
        95: 0.3,   # Emergent herbaceous wetlands
    }
    result = np.ones(len(land_cover_array))
    for code, value in lc_map.items():
        result[land_cover_array == code] = value
    return result

# Load mesh
mesh_data = read_from_msh("terrain6.msh", MPI.COMM_WORLD, gdim=3)
domain = mesh_data.mesh
coords = domain.geometry.x

# Load coordinate offsets saved during meshing
x_offset, y_offset = np.load("coord_offsets.npy")

# Sample and convert
land_cover = sample_nlcd_at_nodes("nlcd_aligned.tif", coords, x_offset, y_offset)
diffusivity = land_cover_to_diffusivity(land_cover)

np.save("land_cover_diffusivity.npy", diffusivity)
np.save("land_cover_classes.npy", land_cover.astype(np.float64))

print(f"Saved! min={diffusivity.min():.2f}, max={diffusivity.max():.2f}")
print(f"Water pixels (barrier): {(diffusivity == 0.0).sum()}")