import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import rasterio
import matplotlib.pyplot as plt
import meshio


with rasterio.open("dem.tif") as src:
    elevation = src.read(1)
    transform = src.transform

elevation = elevation[500:800, 500:800]

elevation = gaussian_filter(elevation, sigma=10)

plt.imshow(elevation, cmap="terrain")
plt.colorbar()
plt.show()

nx, ny = elevation.shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

Z = elevation


points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


tri = Delaunay(points[:, :2])
cells = tri.simplices

mesh = meshio.Mesh(
    points=points,
    cells=[("triangle", cells)]
)

meshio.write(
    "terrain_smoothed.xdmf",
    mesh,
    file_format="xdmf"
)
