from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI

mesh_data = read_from_msh("terrain.msh", MPI.COMM_WORLD, gdim=3)
domain = mesh_data.mesh
print(f"Mesh loaded: {domain.topology.index_map(0).size_global} vertices")
print(f"             {domain.topology.index_map(2).size_global} cells")