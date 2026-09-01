# What's new in this folder

This folder supersedes `FEM(OLD)`. If what you have seen so far is that folder
and the circulated write-up, this is the orientation.

Two things are new. One is an **optimal control problem** — choosing where and
when to cull — built on top of the existing model. The other is that **every
reaction parameter has changed**, because the old ones were rabies values.

Everything else here is documentation, corrections, and tidying.

---

## 1. New: optimal culling control

A culling control `v(x, t)` removes shedding deer, entering the `R` equation as
an extra `−v·R` loss. The objective minimized is

```
J(v) = ∫₀ᵀ ∫_Γ [ c₁·R + c₂·W + c₃·v² + c₄·v ] dΓ dt
```

over `v_min ≤ v ≤ v_max`, subject to the state system. `R` is the target
because it is the only compartment driving *both* transmission routes —
directly through `β₁·S·R` and indirectly by generating the environmental
reservoir through `ρ·R`.

Each iteration is one forward solve and one backward (adjoint) sweep, feeding
either SciPy's L-BFGS-B (the default) or a projected-gradient method with an
Armijo line search. The adjoint is the *discrete* adjoint of the IMEX scheme,
so the gradient is the exact derivative of what the computer minimizes; the
implicit operator is symmetric, so one set of factorizations serves both
sweeps.

New files: `CWD_optimal_control.py` (driver), `cwd_control_problem.py` (forms,
operator, gradient), `utilities/block_solver.py`, `utilities/mesh_bundle.py`,
`utilities/susceptible_spinup.py`. New `optimal_control` block in
`parameters.json`. Sections 7–11 of the write-up.

```bash
# verify the gradient before trusting a long run — the R1 column should
# converge at rate ~2
python CWD_optimal_control.py --mesh-folder example_region_mesh_outputs \
  --gradient-check

# the uncontrolled baseline, through the control problem's own forms
python CWD_optimal_control.py --mesh-folder example_region_mesh_outputs \
  --forward-only --output-folder baseline_outputs
```

`CWD_solver.py` still solves the uncontrolled system and is unchanged in
interface.

## 2. The reaction parameters were rabies values  ⚠️ changes every result

`FEM(OLD)/parameters.json` carried `σ = 13.04/yr` and `α = 73/yr`. Those are
`1/(28 days)` and `1/(5 days)` — the classical **red fox rabies** latency and
infectious period. CWD latency is measured in months and environmental
persistence in years, so every reaction parameter has been reset against the
CWD literature (chiefly Almberg et al. 2011, with Miller et al. 2004 for
environmental persistence).

| | `FEM(OLD)` | now | why |
|---|---|---|---|
| `σ` | 13.04 | **1.926** | 52/27 wk exposed period, not 1/(28 d) |
| `α` | 73 | **0.981** | 52/53 wk shedding period, not 1/(5 d) |
| `β₁` | 80 | **0.0981** | back-solved from `R₀_dir = 1` |
| `β₂` | 1.0 | **0.04905** | back-solved from `R₀_env = 1` |
| `ρ` | 0.5 | **1.0** | now a unit-defining normalization, exact |
| `δ` | 0.1 | **0.5** | 10-yr persistence → 2-yr |
| `r` | 1.5 | **0.35** | `λ = 4.5` is unattainable for a cervid |
| `K₀` | 2 | **10** | boreal density → Midwestern/eastern |
| `κ` | 200 | **4×10⁶** | now carries the whole magnitude, in m²/yr |
| compartment scales | 100, 100, 100 | **1, 1, 0.3** | now dimensionless mobility ratios |
| `T`, `Δt` | 5 yr, 0.01 | **20 yr, 0.02** | horizon a cull can be judged over |

The total is calibrated to `R₀ = 2`, split evenly between the direct and
environmental routes. That gives an endemic equilibrium of `S* = 5.0`,
`I* = 0.358`, `R* = 0.703`, i.e. a 39% population decline at 17.5% prevalence —
in the observed range for a CWD core area. `S* = K₀/R₀` holds exactly and is
the check to run after any recalibration.

**The two β's are tied to `K₀` and `α`** (and `β₂` also to `δ` and `ρ`):

```
beta_1 = R0_dir·α / K0
beta_2 = R0_env·δ·α / (K0·ρ)
```

Change any of those and both must be rescaled, or the epidemic threshold moves
as a side effect of an unrelated edit.

Three land-cover carrying-capacity entries were also corrected, in each case
because habitat quality had been conflated with permeability: woody wetlands
`0.35 → 0.90` (prime cover, merely hard to walk through), grassland
`1.00 → 0.50` (highly permeable, poor habitat), barren land `0.40 → 0.10`.
Open water's diffusivity rose from `10⁻⁴` to `10⁻²` — deer do swim, and the old
value forced a pathologically fine mesh along every shoreline.

Every value now carries its derivation in a `_notes` block beside it in
`parameters.json`, tagged `[LIT]`, `[DEF]`, `[CAL]`, `[GUESS]`, or
`[NUMERICAL]`, and the same material is written up with the arithmetic shown in
the new "Parameter Values and Their Provenance" section. The short version: the
disease parameters are now grounded; everything spatial and everything economic
is still a guess.

## 3. If you have an existing parameter file or mesh bundle  ⚠️

**An `FEM(OLD)`-era `parameters.json` still loads without complaint** —
confirmed by running it — and you get a simulation with fox-rabies kinetics
and no error. The missing `optimal_control` and `susceptible_spinup` blocks
are filled in from defaults, so nothing objects. Use the new
`parameters.json`; don't carry an old one forward.

**Existing mesh bundles must be regenerated.** `FEM(OLD)`'s bundle uses a
different filename and schema (`effective_spatial_parameters.json`) and is
refused outright. Beyond that, `diffusion_tensor.isotropy`,
`land_cover.diffusivity`, and `mesh.min_land_cover_squish` all feed the MMG
refinement metric as well as the PDE tensor, and the diffusivity table and
squish value both changed — so the mesh itself is stale, not just its
parameter sidecar.

## 4. Three numerical changes to the state solver

These affect the uncontrolled solver too, not just the control driver.

- **Mass lumping.** The write-up has always said the scheme uses
  `M_L = diag(M·1)`, but `FEM(OLD)/CWD_solver.py` assembled the consistent mass
  matrix. Its strictly positive off-diagonals destroy the M-matrix structure of
  `M + Δt·A` and let the implicit diffusion step undershoot on sharp initial
  data; the resulting negative densities flip the sign of the logistic term and
  the solve blows up, typically as a NaN a few hundred steps in. Integrating
  the mass terms on a vertex rule gives `M_L` directly. Row sums are preserved,
  so mass is still exactly conserved, and for P1 elements lumping is
  `O(h²)`-consistent.
- **Disease-free spin-up.** `S` no longer starts at the carrying capacity `K`.
  `K` is a piecewise-constant land-cover lookup and is not a steady state of
  the `S` equation — diffusion smooths it across every patch boundary, and
  water is an absorbing sink that draws the profile down around every
  shoreline. Starting at `S = K` lays an order-one demographic transient on top
  of the epidemic one. The `S` equation is now relaxed alone until it stops
  moving, and the result is cached in the mesh bundle, keyed by a signature of
  everything it depends on.
- **Block-diagonal solve.** The implicit operator has no cross-compartment
  terms, so it is factorized per compartment rather than monolithically; equal
  mobilities share a factorization, and the `W` block is diagonal. This is an
  exact restatement of the same linear system, not an approximation.

## 5. The write-up

The circulated version was used as the base. **Its Reaction–Diffusion Model
section is byte-for-byte unchanged**, apart from four added cross-references
and the replacement of the `[value]` placeholders with pointers to the new
values table.

Added: parameter provenance with derivations, an expanded surface-FEM section
(mass lumping, operator symmetry), the disease-free spin-up, and Sections 7–11
on the control problem — formulation, adjoint derivation, optimality condition,
the discrete adjoint of the IMEX scheme, and the control solver.

One deletion: the standalone `\section{Parameter Summary}` was removed. It
redefined `\label{tab:params}`, which the Parameters subsection already used,
so the document carried a duplicate label and its cross-references resolved to
the wrong table. The new provenance section replaces it.

Also corrected: the claim of MPI parallelism (the solver is serial-only, see
below), and a reference to the control array shape.

### A modelling note

The model has **no background-mortality term**: `σ` is the only exit from `I`
and `α` the only exit from `R`, so every infected deer reaches the shedding
class and `α` absorbs disease mortality together with any predation or harvest.
Two consequences follow. There is no harvest anywhere in the model, so `v ≡ 0`
is an *unmanaged* herd and `v` is total removal effort rather than effort above
a normal hunt; and the optimizer sees every culled animal as a certain future
shedder, which overstates the marginal value of culling. A section of the
write-up sets out what adding a non-CWD mortality `μ` would involve — which
terms, which Jacobian entries, and the β rescale it would force.

### Two open questions for the group

- **What is `α`?** The model section describes `R` as the clinical class and
  `1/α` as survival after clinical onset. The parameter derivation lumps
  Almberg's infectious *and* clinical stages, so `1/α` is the full shedding
  duration — which is what gives `α = 52/53`. The two readings differ and the
  adopted numbers follow the lumped one. Flagged in the text rather than
  papered over.
- **Do Almberg's stage durations already net out mortality?** The derivation
  assumes they are progression times. If they are realized residence times, `α`
  needs reducing.

## 6. Documentation

The single README is now two, cross-linked:

- **`README_mesh_generation.md`** — environment, DEM preparation, land-cover
  alignment, meshing, bundle validation.
- **`README_solver.md`** — both drivers, the control problem, disease-free
  spin-up, control blocks, cost weights, outputs.

**Run serially.** The old README described MPI parallelism, but the nodal
`.npy` arrays in a mesh bundle are indexed by local degree of freedom and are
meaningless under a different partition. The block solver now refuses to run on
more than one rank rather than return silently permuted results.

## 7. Fixes to scripts you already have

Small, no behavioural change.

| File | Fix |
|---|---|
| `utilities/resample_land_cover_on_existing_mesh.py` | **Bug:** put the workflow root on `sys.path` to import `shared_parameters`, which lives in `utilities/`. |
| `utilities/check_raster_alignment.py`, `crop_dem_valid_window.py`, `export_mesh_to_xdmf.py` | Had hardcoded filenames; the README told you to edit the source before running them. They take flags now. |
| `align_land_cover.py` | A module-level import shadowed by a function-local re-import; two unused locals. |
| `generate_mesh_and_attributes.py` | Duplicate `numpy`/`rasterio` imports. `mmgpy` is a side-effect import — it registers the PyVista `.mmg` accessor — and is now labelled as such. |
| `CWD_solver.py` | Bundle discovery moved into `utilities/mesh_bundle.py` so the two drivers cannot drift apart; dead trial functions and unused subspaces removed. |

## 8. Folder contents

Self-contained: scripts, `parameters.json`, `environment.yml`, both READMEs,
the write-up, and `example_region_mesh_outputs/` — a complete runnable bundle,
so the solvers can be exercised without building a mesh first. Note its
`effective_parameters.json` carries `final_time_years = 5.0` rather than the
20-year horizon in `parameters.json`; it was generated for quick smoke runs.

A `.gitignore` covers run outputs, mesh products, and source rasters, with the
example bundle whitelisted.
