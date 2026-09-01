"""Write an XDMF/HDF5 copy of a Gmsh .msh surface, for inspection in ParaView.

Optional: the PDE solvers read the .msh directly and never look at this file.
"""

from __future__ import annotations

import argparse

from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import read_from_msh


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", default="terrain.msh",
                        help="Gmsh mesh to read (default: terrain.msh).")
    parser.add_argument("--output", default="terrain.xdmf",
                        help="XDMF file to write (default: terrain.xdmf).")
    return parser


def main():
    args = build_parser().parse_args()
    mesh_data = read_from_msh(args.mesh, MPI.COMM_WORLD, gdim=3)
    with XDMFFile(MPI.COMM_WORLD, args.output, "w") as handle:
        handle.write_mesh(mesh_data.mesh)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
