import meshio
import numpy as np

mesh = meshio.read("terrain.msh")
coords = mesh.points

print(f"X range: {coords[:,0].min():.1f} to {coords[:,0].max():.1f}")
print(f"Y range: {coords[:,1].min():.1f} to {coords[:,1].max():.1f}")
print(f"Z range: {coords[:,2].min():.1f} to {coords[:,2].max():.1f}")

# Centroid of all mesh nodes -- guaranteed to be in the domain
cx = coords[:,0].mean()
cy = coords[:,1].mean()
cz = coords[:,2].mean()
print(f"\nUse this as your gaussian center:")
print(f"x0, y0, z0 = {cx:.2f}, {cy:.2f}, {cz:.2f}")