# What changed in this folder

A cleanup pass over the optimal-control workspace, prepared for sharing. The
messy working copy it came from is kept separately as
`Documents/Optimal Control (messy)`; nothing here was deleted from it.

Three of these changes alter numerical results. They are first.

---

## 1. Background mortality μ removed  ⚠️ changes results

The `−μI` and `−μR` terms are gone. `σ` is now the only exit from `I` and `α`
the only exit from `R`, so every infected deer reaches the shedding class and
`α` absorbs disease mortality together with any predation or harvest.

**This forced a recalibration.** `β₁` and `β₂` were back-solved from a target
`R₀` whose formula contains μ. Deleting μ and keeping the old coefficients
would have raised the effective `R₀` from 2.0 to about 3.7. At μ = 0 the
expected shedding time collapses to `Π = 1/α`, giving:

| | with μ | now |
|---|---|---|
| β₁ | 0.1805 | **0.0981** |
| β₂ | 0.0903 | **0.04905** |
| endemic `I*`, `R*` | 0.312, 0.414 | **0.358, 0.703** |
| prevalence / decline | 12.7% / 42.7% | **17.5% / 39.4%** |

`R₀ = 2` and `S* = K₀/R₀ = 5` still hold exactly — verified numerically.
These are the values the workflow carried before μ was introduced.

Two consequences worth knowing when reading a result:

- **`v ≡ 0` is now an unmanaged herd, not a status quo.** There is no harvest
  anywhere in the model, so the control `v` is *total* removal effort on `R`
  rather than effort above a normal hunt.
- **The optimizer treats every culled animal as a certain future shedder**,
  with no chance something else would have removed it first. That overstates
  the marginal value of culling.

The parameter loader now **rejects** any file still carrying
`reaction.background_mortality_rate`, with a message saying what to reset. The
recipe for reinstating μ later — which terms, which Jacobian entries, and the
β rescale — is recorded in `reaction._notes._no_background_mortality` in
`parameters.json` and in §"No background mortality" of the write-up.

## 2. Linear control cost `c₄·v` integrated  ⚠️ changes the objective

The objective is now

```
J(v) = ∫₀ᵀ ∫_Γ [ c₁·R + c₂·W + c₃·v² + c₄·v ] dΓ dt
```

`c₄` is a straight per-deer price; `c₃` prices congestion at high effort. It is
carried through the entire derivation — running cost, Hamiltonian, reduced
gradient, variational inequality, optimality condition, discrete objective and
gradient, and the algorithm — rather than being an appendix.

The point of it is a **threshold**: the optimal effort is exactly zero wherever
`λ_R·R ≤ c₄`, instead of the quadratic penalty's whisper of effort everywhere.
A five-year trial under the pure-quadratic objective gave peak effort
`0.042/yr` against a space-time mean of `0.0045/yr` — "cull everywhere,
barely", which is not implementable.

**`c₄` defaults to 0**, which reduces everything to the previous purely
quadratic objective, so no shipped result changes unless you raise it. Despite
looking like an L¹ penalty it is smooth on the admissible set, because `v ≥ 0`
puts the kink exactly on the constraint boundary — no proximal machinery, and
L-BFGS-B applies unchanged.

## 3. Mesh bundles must be regenerated  ⚠️

`example_region_mesh_outputs/effective_parameters.json` was updated in place,
but any *other* existing bundle still carries the old β's and μ, and will now
be refused by the loader. Regenerate, or hand-edit the reaction block.

---

## The write-up (`CWD_optimal_control.tex`)

The version circulated by email was used as the base. **Its Reaction–Diffusion
Model section is byte-for-byte unchanged** apart from four additive
cross-references and the replacement of `[value]` placeholders with pointers to
the values table.

Added from the working copy: parameter provenance with derivations, the surface
FEM discretization, the implementation section, and Sections 7–11 on the
control problem (formulation, adjoint derivation, optimality condition,
discrete adjoint of the IMEX scheme, and the control solver).

One deletion: the standalone `\section{Parameter Summary}` was removed. It
redefined `\label{tab:params}`, which the Parameters subsection already used,
so the document had a duplicate label; the provenance section replaces it.

Also updated: the document claimed MPI parallelism (the solver is serial-only),
listed L-BFGS-B and the L¹ term as future work (both are implemented), and gave
the control array shape as `(N_t, N)` rather than `(N_b, N)`.

### Two open questions for the group

- **What is `α`?** Section 3 describes `R` as the clinical class and `1/α` as
  survival after clinical onset. The parameter section derives
  `α = 52/53 yr⁻¹` by lumping Almberg's infectious *and* clinical stages, so
  `1/α` is the full shedding duration. The two readings differ; the numbers
  follow the lumped one. Flagged explicitly in the text rather than papered
  over.
- **Do Almberg's stage durations already net out mortality?** The derivation
  assumes they are progression times. If they are realized residence times,
  then any future reinstatement of μ would double-count and `α` needs reducing.

## Documentation

The single README is now two, cross-linked:

- **`README_mesh_generation.md`** — environment, DEM preparation, land-cover
  alignment, meshing, bundle validation. Follows the shape of the older
  `FEM(OLD)` README.
- **`README_solver.md`** — both drivers, the control problem, disease-free
  spin-up, control blocks, cost weights, outputs.

Corrected along the way: the docs described MPI parallelism, but
`CompartmentBlockSolver` refuses to run on more than one rank — the nodal
`.npy` arrays are indexed by local DOF and are meaningless under a different
partition. **Run serially.** The `cost_control_l1` weight was also implemented
but documented nowhere outside a `_notes` blob.

## Code cleanup

No behavioural change except where noted above.

| File | Change |
|---|---|
| `CWD_solver.py` | Used its own copy of ~110 lines of mesh-bundle discovery, duplicating `utilities/mesh_bundle.py` — which exists so the two drivers can't drift. Now imports it. |
| `CWD_solver.py`, `cwd_control_problem.py` | Removed dead trial functions (the bilinear form moved into `CompartmentBlockSolver`; `U = TrialFunction(P)` and its splits were left behind), a duplicate function space, three unused collapsed subspaces. |
| `utilities/resample_land_cover_on_existing_mesh.py` | **Bug:** put the workflow root on `sys.path` to import `shared_parameters`, which lives in `utilities/`. |
| `CWD_optimal_control.py` | `optimization_history.json` wrote the shedding weight under the old key `c1_infected`. `gradient_check()` took a `control` argument it discarded on its first line. |
| `utilities/shared_parameters.py` | Comments still described `c3 = 4` weighting `I` and a control acting on `I`. |
| `check_raster_alignment.py`, `crop_dem_valid_window.py`, `export_mesh_to_xdmf.py` | Had hardcoded filenames; the old README told you to edit the source before running them. They take flags now. |
| `align_land_cover.py`, `generate_mesh_and_attributes.py` | A shadowed import, two unused locals, duplicate `numpy`/`rasterio` imports. `mmgpy` is a side-effect import (it registers the PyVista `.mmg` accessor) and is now labelled as such. |

`pyflakes` is clean apart from those two intentional `mmgpy` imports.

## Folder contents

**11 GB → 7.3 MB.** Left behind: ~10.9 GB of run outputs (`output2/`,
`new_output_/`, `blocked_test/`, `mumps_test/`, …) and 943 MB of source
GeoTIFFs. All still in `Optimal Control (messy)`.

Kept: the scripts, `parameters.json`, `environment.yml`, both READMEs, the
write-up, and `example_region_mesh_outputs/` — a complete runnable bundle so
the solvers can be exercised without building a mesh first. Note its
`effective_parameters.json` carries `final_time_years = 5.0` rather than the
20-year horizon in `parameters.json`; it was generated for quick smoke runs.

A `.gitignore` now covers run outputs, mesh products, and source rasters, with
the example bundle whitelisted.

---

## What was verified, and what wasn't

**Executed:** the parameter loader, against both shipped parameter files — 18
checks covering `R₀ = 1.000000` on each transmission route, `S* = 5`, and that
both the μ-carrying and the old `cost_infected` key are refused. `pyflakes`
over every source file. A LaTeX structural check: 88 labels, 79 references, no
duplicates, no dangling references, no undefined citations, balanced
environments.

**Not executed:** anything needing DOLFINx — the solvers themselves and the
`--gradient-check` Taylor test — and the LaTeX has not been compiled. No
toolchain for either was available.

Worth doing before trusting a run: `python CWD_optimal_control.py
--mesh-folder example_region_mesh_outputs --gradient-check`. The adjoint's
forward and backward forms are still exact transposes after the μ removal, so
the `R1` column should still converge at rate ≈ 2.
