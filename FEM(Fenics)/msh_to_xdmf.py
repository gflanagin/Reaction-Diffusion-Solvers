import gmsh
from mpi4py import MPI
from dolfinx.io.gmsh import read_from_msh
from dolfinx.io import XDMFFile

mesh_data = read_from_msh("terrain5.msh", MPI.COMM_WORLD, gdim=3)
domain = mesh_data.mesh

with XDMFFile(MPI.COMM_WORLD, "terrain5.xdmf", "w") as xf:
    xf.write_mesh(domain)