"""Compartment-blocked direct solver for the implicit operator B.

Why this exists
---------------
The bilinear form of the state system carries no cross-compartment terms --
every term is either ``u_X*v_X`` or ``grad(u_X).grad(v_X)`` for a single
compartment X. The operator

    B = M_L + dt * A,   A = blockdiag(D_S*A_D, D_I*A_D, D_D*A_D, 0)

is therefore *block diagonal by compartment*, and the four blocks are

    B_S = M_L + dt*D_S*A_D
    B_I = M_L + dt*D_I*A_D
    B_R = M_L + dt*D_R*A_D
    B_W = M_L                  (no stiffness: W has no diffusion term)

Assembling these as one monolithic matrix on the mixed space and handing it to a
general sparse LU asks the factorization to rediscover a structure we already
know exactly. Three things are lost by doing so:

  * ``B_W`` is the lumped mass matrix, which is *diagonal*. A quarter of the
    degrees of freedom are pushed through a sparse triangular solve when they
    need one vector division.
  * When two compartment mobilities coincide -- and ``susceptible`` and
    ``infected`` are both 1.0 in the shipped parameters -- the corresponding
    blocks are the *same matrix*, and only one factorization is needed for both.
  * Fill-in is superlinear, so factoring one 4N-dof system costs more than
    factoring the distinct N-dof blocks it decomposes into, even when the
    ordering does manage to keep the components apart.

This module solves each block separately against its own cached factorization.
The result is *bit-for-bit the same linear system*, just solved blockwise: no
approximation is introduced, each block stays symmetric, and B^T = B still
holds, so the discrete adjoint identity of the control problem is untouched.
(Verify with ``--gradient-check``: the Taylor rate must stay at 2.)

Solver package
--------------
``configure_direct_solver`` is also the one place that picks the factorization
package. The drivers previously set ``preonly``/``lu`` programmatically and
never called ``setFromOptions``, which meant PETSc's options database was
bypassed entirely -- ``PETSC_OPTIONS=-pc_factor_mat_solver_type mumps`` was
silently ignored and reported as an unused option at exit. MUMPS is preferred
here when the build provides it, and ``setFromOptions`` is called last so that
an explicit ``PETSC_OPTIONS`` still wins.
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import TestFunction, TrialFunction, dot, grad


def configure_direct_solver(ksp, prefix=None):
    """Set up a KSP as a cached direct solve, preferring MUMPS when available.

    Always call this instead of setting ``preonly``/``lu`` by hand: it is what
    makes the PETSc options database apply to these solvers at all.
    """
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")

    try:
        has_mumps = PETSc.Sys.hasExternalPackage("mumps")
    except Exception:
        # Older petsc4py builds lack the query; fall back to PETSc's own LU
        # rather than requesting a package that may not be linked in.
        has_mumps = False
    if has_mumps:
        pc.setFactorSolverType("mumps")

    if prefix is not None:
        ksp.setOptionsPrefix(prefix)
    # Last, so that anything in PETSC_OPTIONS overrides the choices above.
    ksp.setFromOptions()
    return ksp


def _dofmap_array(V):
    """The cell-to-DOF connectivity of ``V`` as a flat array.

    DOLFINx has exposed this both as a raw array and as an AdjacencyList
    depending on version, so normalize rather than assume.
    """
    listing = V.dofmap.list
    return np.asarray(getattr(listing, "array", listing))


def lumped_mass_diagonal(V, mass_measure):
    """diag(M_L) on ``V``, as owned nodal values.

    For P1 the vertex rule is exact on linear integrands, so the row sums
    int(phi_i) that define the lumped mass are the same whichever of the two
    measures is used; ``mass_measure`` is taken as an argument only so the
    caller's rule is used verbatim.
    """
    v = TestFunction(V)
    b = assemble_vector(fem.form(v * mass_measure))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    values = b.array_r.copy()
    b.destroy()
    return values


class CompartmentBlockSolver:
    """Direct solves against B, one cached factorization per distinct block.

    ``solve(b, out)`` takes an assembled right-hand side on the mixed space
    (ghost-updated) and writes B^{-1} b into the mixed ``Function`` ``out``.
    """

    def __init__(self, P, dt, DTens, compartment_scales,
                 mass_measure, stiffness_measure, comm=MPI.COMM_WORLD):
        self.comm = comm
        if comm.size > 1:
            # The gather/scatter below indexes a Function's local array with the
            # collapsed sub-space DOF maps, which is only unambiguous when there
            # are no ghosts. The workflow is serial-only anyway -- the mesh
            # bundle's nodal .npy arrays are written in serial ordering and the
            # size check in the drivers rejects any other rank count before
            # reaching this point -- so this is not a new restriction. It is
            # spelled out rather than left to produce silently wrong numbers.
            raise NotImplementedError(
                "CompartmentBlockSolver supports serial runs only. The mesh "
                "bundle's nodal arrays are indexed by local DOF and are already "
                "serial-only; see utilities/susceptible_spinup.py for the same "
                "restriction."
            )

        self.P = P
        blocks = [P.sub(k).collapse() for k in range(4)]
        self.spaces = [space for space, _ in blocks]
        self.dof_maps = [np.asarray(dofs, dtype=np.int32) for _, dofs in blocks]

        V = self.spaces[0]
        # The four collapsed spaces are built from identical sub-elements on one
        # mesh, so their DOF maps coincide and one assembled block serves all
        # four. That is a property of how DOLFINx renumbers a collapsed
        # sub-space, not something the API promises, so check it rather than
        # assume it -- a mismatch here would silently permute the solution.
        reference = _dofmap_array(V)
        for k, space in enumerate(self.spaces[1:], start=1):
            if not np.array_equal(_dofmap_array(space), reference):
                raise RuntimeError(
                    "Collapsed sub-space %d has a different DOF map from "
                    "sub-space 0, so the compartment blocks cannot share an "
                    "assembled operator." % k
                )

        u = TrialFunction(V)
        v = TestFunction(V)

        # W block: pure lumped mass, hence diagonal. Stored as its reciprocal
        # so the per-step solve is a multiply.
        mass_diagonal = lumped_mass_diagonal(V, mass_measure)
        if np.any(mass_diagonal <= 0.0):
            raise RuntimeError(
                "The lumped mass matrix has a non-positive diagonal entry, so "
                "the mesh has a degenerate cell; the W block is not invertible."
            )
        self.inverse_mass_diagonal = 1.0 / mass_diagonal

        # One factorization per *distinct* mobility. With the shipped
        # parameters susceptible and infected are both 1.0, so S and I share.
        self.matrices = {}
        self.solvers = {}
        self.block_keys = []
        for index, scale in enumerate(compartment_scales):
            key = repr(float(scale))
            if key not in self.solvers:
                a = (
                    u * v * mass_measure
                    + dt * float(scale)
                    * dot(DTens * grad(u), grad(v)) * stiffness_measure
                )
                matrix = assemble_matrix(fem.form(a))
                matrix.assemble()
                ksp = PETSc.KSP().create(V.mesh.comm)
                ksp.setOperators(matrix)
                configure_direct_solver(ksp, prefix="block%d_" % index)
                self.matrices[key] = matrix
                self.solvers[key] = ksp
            self.block_keys.append(key)

        # Scratch vectors, allocated once rather than per solve.
        any_matrix = self.matrices[self.block_keys[0]]
        self.block_rhs = any_matrix.createVecRight()
        self.block_solution = any_matrix.createVecRight()

    @property
    def distinct_factorizations(self):
        """How many LU factorizations are actually held (2 of 3 typically)."""
        return len(self.solvers)

    @property
    def factor_package(self):
        """Which package is doing the factorization, for the start-up banner."""
        ksp = self.solvers[self.block_keys[0]]
        try:
            return str(ksp.getPC().getFactorSolverType())
        except Exception:
            return "unknown"

    def solve(self, b, out_function):
        """B x = b, blockwise. ``b`` must already be ghost-updated."""
        rhs = b.array_r
        out = out_function.x.array

        for index, key in enumerate(self.block_keys):
            dof_map = self.dof_maps[index]
            self.block_rhs.array[:] = rhs[dof_map]
            self.solvers[key].solve(self.block_rhs, self.block_solution)
            out[dof_map] = self.block_solution.array_r

        # W: diagonal lumped mass, so no triangular solve at all.
        dof_map = self.dof_maps[3]
        out[dof_map] = rhs[dof_map] * self.inverse_mass_diagonal

        out_function.x.scatter_forward()

    def destroy(self):
        for ksp in self.solvers.values():
            ksp.destroy()
        for matrix in self.matrices.values():
            matrix.destroy()
        self.block_rhs.destroy()
        self.block_solution.destroy()
        self.solvers.clear()
        self.matrices.clear()
