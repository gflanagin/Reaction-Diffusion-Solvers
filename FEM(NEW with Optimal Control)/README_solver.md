# Running the solvers: state simulation and optimal culling control

This is half of the documentation for this folder. It assumes you already have
a **mesh bundle** — a `terrain.msh` with its two nodal `.npy` arrays and an
`effective_parameters.json`. Building one is documented in
[`README_mesh_generation.md`](README_mesh_generation.md).

There are two drivers:

| Script | What it does |
| --- | --- |
| `CWD_solver.py` | the uncontrolled state system: one forward simulation, no optimization |
| `CWD_optimal_control.py` | the culling control problem: forward/backward sweeps driving an optimizer |

Both read the same bundle, the same parameter file, and the same disease-free
spin-up, and both discretize the state system identically, so their
trajectories are directly comparable.

The full mathematical write-up — the model, the parameter provenance, the
adjoint derivation, the discrete adjoint of the IMEX scheme, and the optimizer
— is `CWD_optimal_control.tex`.

> **Run serially.** The nodal arrays in a mesh bundle are indexed by local
> degree of freedom, so they are meaningful only for the partition that wrote
> them. `CompartmentBlockSolver` refuses to run under more than one MPI rank
> rather than return silently permuted results.

---

## The uncontrolled solver

```bash
conda activate cwd-fem
python CWD_solver.py --mesh-folder example_region_mesh_outputs \
  --output-folder solver_outputs
```

`--mesh-folder` discovers the folder's single `*.msh`,
`land_cover_classes.npy`, `land_cover_diffusivity.npy`, and
`effective_parameters.json` (falling back to `parameters.json`). It stops with
a descriptive error if a required file is absent, if the folder holds more than
one `.msh`, or if either nodal array length does not match the loaded mesh.

`--mesh`, `--land-cover-classes`, `--land-cover-diffusivity`, and
`--parameters` override folder discovery individually.

Outputs are `CWD_S`, `CWD_I`, `CWD_D`, `CWD_E`, and `land_cover_classes` as
XDMF/HDF5 series for ParaView.

Note the code names: the shedding compartment written `R` in the LaTeX is `D`
("dying") in the code and the output files, and the environmental compartment
`W` is `E` ("environment").

---

## The control problem

A culling control `v(x, t)` removes **shedding** deer, entering the `R`
equation as an extra `- v*R` loss. The objective minimized is

```
J(v) = ∫₀ᵀ ∫_Γ [ c₁·R + c₂·W + c₃·v² + c₄·v ] dΓ dt
```

over controls in the admissible box `v_min ≤ v ≤ v_max`, subject to the state
system. `c₃` and `c₄` are the quadratic and linear parts of the cost of the
intervention; `c₄` defaults to zero, which reduces the objective to the purely
quadratic one. See [Cost weights](#cost-weights).

`R` is the right target because it is the only compartment driving **both**
transmission routes — directly through `β₁·S·R`, and indirectly by generating
the environmental reservoir through `ρ·R`. Culling `I` (which an earlier
revision did) only delays the same animals' arrival in `R`. The upstream class
is not ignored: `I` feeds `R`, so the cost on `R` propagates back through the
adjoint and `λ_I` stays positive — a sub-clinical animal is priced at the
shadow price of the shedding animal it will become.

Note the caveat: `R` lumps pre-clinical shedders (~36 of its 53 weeks, and
indistinguishable from healthy deer in the field) with visibly clinical
animals. A control acting on all of `R` presupposes test-and-cull. A
sight-based cull would reach only the clinical third, and modelling it needs
`R` split in two.

The model carries **no background mortality**: `σ` is the only exit from `I`
and `α` the only exit from `R`, so every infected deer reaches the shedding
class and `α` absorbs disease mortality together with any predation or harvest
that targets symptomatic animals. Two things follow, and both bear on how a
result should be read. There is no harvest anywhere in the model, so `v ≡ 0` is
an *unmanaged* herd rather than a status-quo managed one, and `v` is total
removal effort on the shedding class rather than effort above a normal hunt.
And the optimizer sees every culled animal as a certain future shedder, with no
chance that something else would have removed it first, which overstates the
marginal value of culling.

Reinstating a non-CWD mortality `μ` on `I` and `R` is a contained change — the
two reaction rows, the matching `(I,I)` and `(R,R)` diagonal entries of the
adjoint, and a rescale of `β₁` and `β₂` — and the recipe is recorded in
`reaction._notes._no_background_mortality` in `parameters.json`.

### How an iteration works

Each outer iteration is one **forward solve** and one **backward solve**:

1. **Forward.** Advance the controlled state system with the current control,
   recording the whole trajectory and evaluating `J(v)`.
2. **Backward.** Sweep the discrete adjoint from `λ^N` down to `λ^1`,
   accumulating the reduced gradient
   `∇J(v)|_b = 2·c₃·v_b + c₄ − mean_{n∈b} Rⁿ·λ_Rⁿ⁺¹`, over the time steps in
   each control block (see [Control blocks](#control-blocks)).

That gradient feeds one of two optimizers, set by `optimal_control.optimizer`
or `--optimizer`:

- **`lbfgs`** (default) — SciPy's **L-BFGS-B** on the reduced problem. The
  admissible set is exactly the simple box L-BFGS-B is built for, and it
  accumulates curvature, so it converges superlinearly on the inactive set
  rather than linearly. Same cost per iteration; far fewer iterations.
- **`projected-gradient`** — the original projected steepest descent with an
  **Armijo** backtracking line search, kept as a reference implementation and
  as the fallback when SciPy is unavailable. Each backtrack costs one more
  forward solve.

Both consume the identical adjoint gradient, so `--gradient-check` validates
either.

Two properties make this cheap. The implicit operator `B` is **symmetric** (the
mass matrix is, and the anisotropic diffusion tensor is), so the block
factorizations built at start-up serve the forward *and* the backward sweep —
no second matrix is ever assembled. And the adjoint is the **discrete** adjoint
of the IMEX scheme rather than a re-discretization of the continuous adjoint
PDE, so the computed gradient is the exact derivative of the discrete
objective.

---

## Running the control driver

```bash
conda activate cwd-fem
python CWD_optimal_control.py --mesh-folder outputs \
  --output-folder control_outputs
```

It takes the same bundle flags as `CWD_solver.py`.

### Verify the gradient first

Before trusting a long optimization run, check the adjoint with a Taylor test:

```bash
python CWD_optimal_control.py --mesh-folder outputs --gradient-check
```

This perturbs the control along a random direction and prints two remainders:

```
R0 = |J(v+εd) − J(v)|                      should fall like O(ε)   (rate ≈ 1)
R1 = |J(v+εd) − J(v) − ε·⟨∇J(v), d⟩|       should fall like O(ε²)  (rate ≈ 2)
```

The `R1` rate is the real test. A rate near 2 means the gradient is correct; a
rate near 1 means it is not, and no amount of tuning the line search will help.
Expect the rate to degrade at the smallest `ε` as round-off takes over — that
is normal and the check ignores those rows. The test runs at its own interior
control, not at whatever `initial_control` says, so that no bound is active and
the reduced objective is genuinely differentiable.

Re-run it after any change to the forms, the quadrature rules, or the
parameters.

### Just running the model

To simulate the state system once with no optimization at all:

```bash
python CWD_optimal_control.py --mesh-folder outputs --forward-only \
  --output-folder baseline_outputs
```

This writes the same `CWD_S/I/D/E` series an optimization run produces, plus a
`forward_summary.json` recording `J` and its state/control split. With no
`--initial-control` the control is zero throughout, so this is the
**uncontrolled baseline** — the value an optimized strategy has to beat. Pass
`--initial-control some_run/optimal_control.npy` to replay a saved strategy
instead, without re-optimizing.

Because no adjoint is solved, nothing needs the state history: `--forward-only`
stores no trajectory at all. It therefore runs on meshes and horizons where the
optimization loop would not fit in memory, and the start-up banner reports how
much it saved.

`CWD_solver.py` solves the same uncontrolled system; the difference is that
`--forward-only` goes through the control problem's own forms, so it is the
right baseline to compare an optimized `J` against.

### Flags

| Flag | Effect |
| --- | --- |
| `--max-iterations N` | Override `optimal_control.max_iterations`. |
| `--optimizer lbfgs\|projected-gradient` | Override `optimal_control.optimizer`. |
| `--control-block-years Y` | Override the control block length; `0` means one value per time step. |
| `--initial-step-size A` | Override `optimal_control.initial_step_size`. |
| `--no-auto-initial-step` | Use that value verbatim instead of auto-scaling the first step. |
| `--initial-control FILE.npy` | Warm start from a previous run's `optimal_control.npy`. |
| `--trajectory-file FILE.npy` | Memory-map the stored forward trajectory instead of holding it in RAM. |
| `--output-every N` | Write every Nth time step to XDMF (default 3). |
| `--write-adjoint` | Also write the adjoint fields (times arrive in decreasing order). |
| `--gradient-check` | Run the Taylor test and exit. |
| `--forward-only` | Run one forward simulation and exit, without optimizing. |
| `--susceptible-equilibrium FILE.npy` | Cached disease-free `S` field; defaults to beside the mesh. |
| `--recompute-equilibrium` | Re-run the disease-free spin-up instead of reusing the cache. |

`--gradient-check` and `--forward-only` both exit without optimizing, so asking
for both at once is rejected rather than silently resolved.

---

## Disease-free spin-up

Before any run seeds an infection, `S` is relaxed to its **disease-free
equilibrium**, and that field — not the carrying capacity `K` — is the initial
susceptible condition. `K` is a piecewise-constant land-cover lookup and is not
a steady state of the `S` equation: diffusion smooths it across every patch
boundary, and water (`K = 0`, where growth becomes `−r·S`) is an absorbing sink
that draws the profile down for a diffusion length around every shoreline.
Starting at `S = K` therefore lays an order-one *demographic* transient on top
of the epidemic one, and at `r = 0.35/yr` the two relax on comparable
timescales, so it does not wash out before the interesting dynamics begin. For
the control problem it is worse than cosmetic — the spurious structure is
something the optimizer will spend control effort on.

The spin-up runs the `S` equation alone (`I = R = W = 0`) from `S = K` until
`max|dS/dt|` falls below `susceptible_spinup.drift_tolerance` times the peak
carrying capacity, or until `max_duration_years` runs out. It happens
automatically the first time either solver runs on a mesh bundle, and the result
is cached in the bundle for every later run:

```
outputs/
  terrain.msh
  land_cover_classes.npy
  land_cover_diffusivity.npy
  susceptible_equilibrium.npy     <- the relaxed field
  susceptible_equilibrium.json    <- what it was computed from
```

The `.json` sidecar records a signature of everything the answer depends on —
digests of the mesh geometry, the carrying capacity, and the diffusivity, plus
`r`, `kappa`, `isotropy`, the two activation parameters, the susceptible
diffusion scale, and the spin-up step and tolerance. If any of those change the
field is recomputed and the run says which key moved, so an edit to the land
cover or to `kappa` cannot leave a stale equilibrium in place.

Note that a *couple* of years is not enough. Logistic relaxation has an
e-folding time `1/r = 2.9 yr`, but diffusive smoothing across a patch of linear
scale `L` takes of order `L²/kappa` — 1 yr for a 2 km patch, 25 yr for a 10 km
one — and shoreline drawdown around a large lake is slower still. That is why
the stopping rule is a convergence test with a generous 60-year ceiling rather
than a fixed short run: overshooting costs one cheap scalar solve, while
stopping early contaminates every simulation built on the cached field.

`--susceptible-equilibrium FILE.npy` uses (and writes) the cache somewhere
other than beside the mesh; `--recompute-equilibrium` re-runs the spin-up even
when a matching cache exists. Both flags work on either driver. Setting
`susceptible_spinup.enabled` to `false` in `parameters.json` skips the step and
starts `S` at `K`, which is what this workflow did before the spin-up existed —
useful only for reproducing older runs.

---

## Control blocks

The culling effort is **piecewise constant in time**, held fixed over blocks of
`optimal_control.control_block_years` (default `1.0`, i.e. annual). The
optimizer therefore chooses one spatial field per year — 20 over the default
horizon — rather than one per 7.3-day time step.

This is a genuine **restriction of the admissible set**, so the optimal `J` it
reaches is no lower than the per-step optimum. But it restricts *toward*
realism: no agency re-tunes a cull weekly, whereas quotas are in fact set
annually. It is also what makes a quasi-Newton method affordable. The
optimization variable is one CG1 field per block:

| Blocking | vector length on a 210k-node mesh | one copy | L-BFGS-B at `lbfgs_memory=10` |
| --- | --- | --- | --- |
| per time step (`null`) | 210M | 1.6 GiB | ~39 GiB — **will not run** |
| annual (default) | 4.2M | 34 MiB | ~0.8 GiB |

Set `control_block_years` to `null` (or pass `--control-block-years 0`) to
recover one value per step, which is what this workflow did before blocking
existed; shorten it to `0.25` for seasonal structure at four times the cost.
`lbfgs_memory` trades directly against it — lower that before shortening the
block.

Because the control's shape changes with the block length, an
`optimal_control.npy` saved under a different `control_block_years` cannot be
reused with `--initial-control`; the driver rejects it with an explicit message
rather than silently reshaping.

## Memory

The backward sweep evaluates the state Jacobian at every `yⁿ`, so the entire
forward trajectory must be kept. That is `4 · (nodes) · (steps+1) · 8` bytes —
roughly 1.6 GB for a 100k-node mesh over 500 steps. The driver prints the figure
at start-up. If it does not fit, pass `--trajectory-file` to back it with an
on-disk memory map, or `--forward-only` if you do not need the optimization.

---

## Cost weights

The `optimal_control` block of `parameters.json`:

```json
"optimal_control": {
  "cost_shedding": 1.0,          // c1: weight on the shedding burden R
  "cost_environment": 0.5,       // c2: weight on the environmental prion load
  "cost_control": 2.0,           // c3: quadratic weight on the culling effort
  "cost_control_l1": 0.0,        // c4: linear per-deer weight; 0 = pure quadratic
  "control_block_years": 1.0,    // v held constant over 1-yr blocks; null = per step
  "optimizer": "lbfgs",          // or "projected-gradient"
  "lbfgs_memory": 10,            // L-BFGS-B correction pairs (SciPy "maxcor")
  "control_minimum": 0.0,
  "control_maximum": 1.0,        // v_max, per year
  "initial_control": 0.0,
  "max_iterations": 40,
  "initial_step_size": 1.0,
  "armijo_sufficient_decrease": 1e-4,
  "armijo_backtrack_factor": 0.5,
  "armijo_step_growth": 2.0,
  "armijo_max_backtracks": 25,
  "minimum_step_size": 1e-12,
  "gradient_tolerance": 1e-8,
  "relative_cost_tolerance": 1e-8
}
```

Only the ratios `c1 : c2 : c3 : c4` matter, since scaling `J` does not move the
minimizer. The weights carry units, so pick them so the terms are comparable in
magnitude — otherwise the smallest is effectively ignored. A practical
calibration is to run once, read the `state_cost` / `control_cost` split
printed at iteration 0 and at the end, and rescale. The values above come from
applying that recipe by hand; all of them are guesses, and `c3` is the one most
in need of a real per-deer cull cost.

At least one of `cost_control` and `cost_control_l1` must be positive. With
neither, nothing penalizes effort and the optimizer simply saturates the bound.

### The linear control weight `c4`

`cost_control_l1` (`c4`) is the linear part of the intervention cost — a
straight per-deer price, where `c3` prices congestion at high effort. Both are
carried through the derivation in the write-up. It defaults to `0.0`, which
reduces the objective to the purely quadratic one, so every shipped result is
the familiar case.

Raising it shifts stationarity from `v* = P[λ_R·R/(2·c3)]` to
`v* = P[(λ_R·R − c4)/(2·c3)]`. The practical effect is a **threshold**: the
optimum is exactly zero wherever `λ_R·R ≤ c4`, instead of the quadratic
penalty's whisper of effort spread over the whole domain. That is the "not
here" half of a cull-here-not-there map, and it is the half a management plan
actually needs. A five-year trial under the pure-quadratic objective put peak
effort at `0.042/yr` and the space-time mean at `0.0045/yr` — "cull everywhere,
barely", which is not implementable as stated.

Despite looking like an L1 penalty, this is **not** a nonsmooth term. Because `v ≥ 0` is already
imposed, `∫|v| = ∫v` on the admissible set, so the kink of the absolute value
sits exactly on the constraint boundary and the objective stays smooth —
L-BFGS-B applies unchanged, with no proximal or FISTA machinery. Do not reach
for a smoothed surrogate such as `sqrt(v² + ε)`: its curvature spans about six
orders of magnitude across the box at `ε = 1e-6`, which manufactures
ill-conditioning.

The real cost of raising `c4` while lowering `c3` is the loss of strong
convexity: `c3` contributes a uniform `2·c3·I` floor to the reduced Hessian,
and without it only the compact state-coupling operator remains, whose spectrum
decays to zero. `c3 → 0` is the pure bang-bang limit and is where conditioning
degrades — a continuous dial, not a cliff. Sweep `c4` and watch two numbers:
the iteration count, and the fraction of space-time at exactly zero, which the
driver prints at the end of every run.

### Backward compatibility

The whole `optimal_control` block is optional: a parameter file written before
this work existed still loads, with the defaults above filled in by
`utilities/shared_parameters.py`.

Two keys are **not** forgiving, because silently defaulting them would give a
wrong answer rather than an error:

- A file carrying the old `cost_infected` is rejected outright — the cost now
  weights `R`, whose endemic density is about half that of `I`, so the weight
  needs rescaling and not just renaming.
- A file that still carries `reaction.background_mortality_rate` is rejected
  too. Its transmission rates were back-solved from an `R₀` formula containing
  `μ`, so dropping the term while keeping those rates raises the effective `R₀`
  from 2.0 to about 3.7.

Both errors say what to do.

---

## Where the numbers come from

Every value in `parameters.json` carries its own justification, in a `_notes`
block sitting alongside the values in each section. The same material is written
up properly in `CWD_optimal_control.tex`, under "Parameter Values and Their
Provenance", with the derivations shown. Five tags are used:

| Tag | Meaning |
|---|---|
| `[LIT]` | derived from published CWD data, derivation shown |
| `[DEF]` | a unit-defining normalization — exact by construction, not an estimate |
| `[CAL]` | back-solved from a stated target (a target `R0`, a term balance) |
| `[GUESS]` | expert judgement, no citation for this study region |
| `[NUMERICAL]` | exists for conditioning or discretization; represents nothing about deer or prions |

In the `.tex`, guessed values are **shaded** in the summary table so they can be
found at a glance. The short version: the *disease* parameters are grounded —
stage durations, environmental decay, host growth rate all trace to Almberg et
al. (2011) and Miller et al. (2004), and the two transmission coefficients
follow from those by an explicit `R0` calculation. Everything *spatial* and
everything *economic* is a guess: carrying capacity, the diffusivity scale and
its anisotropy, the whole land-cover table, and all the cost and bound
parameters.

Two things are easy to trip over:

- **`beta_1` and `beta_2` are tied to `carrying_capacity_base` — and to `α`,
  `δ`, `ρ`.** They were back-solved from `R0_dir = R0_env = 1`:

  ```
  beta_1 = R0_dir·α / K0
  beta_2 = R0_env·δ·α / (K0·ρ)
  ```

  Change any input and both must be rescaled, or you move the epidemic
  threshold as a side effect of an edit that had nothing to do with
  transmission. This is not hypothetical: an earlier revision added a
  background mortality `μ` to `I` and `R` without rescaling, which dropped
  `R0` from 2.00 to 1.09, nearly to threshold. A good check after any
  recalibration is `S* = K0/R0`, which holds exactly.
- **`diffusion_tensor.isotropy` and `land_cover.diffusivity` also drive the mmg
  mesh metric**, not just the solver, so a change to either needs a new mesh
  bundle before `effective_parameters.json` reflects it. See
  [`README_mesh_generation.md`](README_mesh_generation.md).

---

## Mass lumping

Both solvers lump the mass matrix. The consistent mass matrix `M_ij = ∫ φᵢφⱼ`
has strictly positive off-diagonal entries, which destroy the M-matrix
structure of `M + Δt·A` and let the implicit diffusion step **undershoot** on
sharp initial data. The resulting negative densities flip the sign of the
logistic term `r·S·(1 − N/K)` and the solve blows up — typically as a NaN cost
a few hundred steps in. Narrow initial-outbreak Gaussians are the usual
trigger.

Integrating the mass terms with a vertex rule (`quadrature_rule: "vertex"`)
gives `M_L = diag(M·1)` directly, since `φᵢφⱼ = δᵢⱼ` at the vertices. Row sums
are preserved, so total mass is still exactly conserved, and for P1 elements
lumping is `O(h²)`-consistent, so no accuracy order is lost. Both sides of the
step are lumped, so the scheme is genuinely
`(M_L + Δt·A)·yⁿ⁺¹ = M_L·yⁿ + Δt·F(yⁿ)`. The nonlinear reaction terms stay on
a Gauss rule.

This does **not** disturb the adjoint. `B` still appears identically on the left
of both sweeps and a diagonal mass is still symmetric, so `B^T = B` holds and
the derivation is untouched. The mass terms of the adjoint load vector are
lumped alongside their forward counterparts so each term is still
differentiated on the rule it was assembled with. Re-run `--gradient-check`
after any change here; it should still give rate 2.

Lumping is a *structural* fix — the operator itself now preserves positivity.
Clipping the state to `≥ 0` instead would be a post-hoc patch that breaks
differentiability of `v ↦ y(v)`, and hence silently invalidates the gradient
wherever it activates.

If undershoot persists after lumping, the initial condition is under-resolved:
the Gaussian `standard_deviation` needs to span several elements (roughly
`σ ≳ 3–4·h` locally), so refine the mesh where the outbreak is seeded rather
than narrowing σ further.

---

## Outputs

Written to `--output-folder` (default `control_outputs`):

| File | Contents |
| --- | --- |
| `CWD_S`, `CWD_I`, `CWD_D`, `CWD_E` `.xdmf`/`.h5` | optimal state time series |
| `CWD_control.xdmf`/`.h5` | the optimal control field over time |
| `CWD_adjoint_S/I/D/E` | adjoint fields, with `--write-adjoint` |
| `optimal_control.npy` | control array, shape `(blocks, nodes)`; feed to `--initial-control` |
| `optimization_history.json` | per-iteration cost, gradient norm, step size, backtrack count |
| `forward_summary.json` | `J` and its split, from `--forward-only` runs |
| `land_cover_classes.xdmf` | land-cover classes, for reference |

The cost reported at iteration 0 of a run started from `v ≡ 0` is the
do-nothing baseline; compare the final cost against it. The end-of-run report
also gives the peak and space-time mean of the control, and the fraction of
space-time left at `v_min` and driven to `v_max` — the two numbers to watch
when sweeping `c4`.

---

## A caveat on the optimization

The reduced objective is **not convex** — the state system is nonlinear in
`(S, I, R, W)` and in `v`, so `v ↦ y(v)` is nonlinear. Both optimizers are
descent methods and find a *stationary* point, not necessarily a global
minimum. Re-run from a few different initial controls (via `--initial-control`,
or by changing `initial_control`) before treating a result as the answer.

---

## Solver-side folder contents

- `CWD_solver.py` — the uncontrolled driver.
- `CWD_optimal_control.py` — the control driver: argument parsing, the
  forward/backward loop, both optimizers, the Taylor test, and output.
- `cwd_control_problem.py` — `CWDControlProblem`: mesh, spaces, land-cover
  fields, diffusion tensor, the operator `B` and its factorization, and every
  variational form (state RHS, adjoint RHS, cost, gradient). Keeping them in
  one class is what guarantees the forward and adjoint forms share a mesh,
  coefficients, and — critically — a quadrature rule.
- `CWD_optimal_control.tex` — the full write-up.
- `utilities/mesh_bundle.py` — the bundle-discovery logic and the shared
  command-line flags, so the two drivers cannot drift apart.
- `utilities/block_solver.py` — `CompartmentBlockSolver`: `B` is block diagonal
  by compartment, so each block is factorized separately, blocks with equal
  mobility share a factorization, and the `W` block (lumped mass, hence
  diagonal) is a multiply rather than a triangular solve. This is an exact
  restatement of the same linear system, not an approximation.
- `utilities/susceptible_spinup.py` — the disease-free relaxation, its on-disk
  cache, and the signature check that invalidates it.
- `utilities/shared_parameters.py` — the validated parameter loader, shared with
  mesh generation.
