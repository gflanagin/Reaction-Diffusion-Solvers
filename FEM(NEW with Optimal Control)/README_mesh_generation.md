# Mesh generation: DEM and land cover to a solver bundle

This is half of the documentation for this folder. It covers everything up to
and including the production of a **mesh bundle**: the terrain surface plus the
per-node land-cover arrays that both solvers read. Running the solvers on that
bundle — the uncontrolled state solver and the optimal-culling driver — is
documented in [`README_solver.md`](README_solver.md).

The pipeline is unchanged from the older `FEM(OLD)` workflow; the scripts are
mirrored here so this folder is self-contained.

---

## Software environment

Run the workflow in Linux. The tested setup is WSL2 with Ubuntu 24.04 and the
Conda environment defined by `environment.yml` (Python 3.11, DOLFINx 0.11,
OpenMPI 5.0, Gmsh 4.15, meshio 5.3, Rasterio 1.4, SciPy 1.16, PyVista, and
mmgpy 0.16).

For a first-time setup, install WSL/Ubuntu if needed, install Miniforge or
another Conda-compatible manager inside Ubuntu, then run from this directory:

```bash
conda env create --file environment.yml
```

Activate it in each new shell:

```bash
conda activate cwd-fem
```

After `environment.yml` changes, update in place:

```bash
conda env update --name cwd-fem --file environment.yml --prune
```

Keep DOLFINx, PETSc, HDF5, MPI, Basix, UFL, and their Python bindings in this
one Conda environment so their compiled versions stay compatible. ParaView is
optional and can be installed separately.

Check the environment before doing anything else:

```bash
python utilities/verify_environment.py
```

It reports the version of every required package and confirms that the PyVista
`.mmg` accessor registered, which is what the remeshing step in step 3 needs.

### Run serially

**This workflow is serial-only.** The nodal `.npy` arrays in a mesh bundle are
indexed by local degree of freedom and carry no global node IDs, so they are
meaningful only for the partition that wrote them. `CompartmentBlockSolver`
refuses to run under more than one MPI rank rather than produce silently
permuted results. Generate the bundle and consume it in the same serial run.

---

## What a mesh bundle is

A solver run needs these files in one folder:

| File | Contents | Used by the solver as |
| --- | --- | --- |
| `terrain.msh` | 2-D triangular terrain surface embedded in 3-D | DOLFINx mesh (`gdim=3`) |
| `land_cover_classes.npy` | one NLCD class code per CG1 node | input to the carrying-capacity mapping |
| `land_cover_diffusivity.npy` | one diffusivity multiplier per CG1 node | spatial multiplier `ℓ(x)` in the diffusion tensor |
| `effective_parameters.json` | the parameter file actually used for this mesh, CLI overrides folded in | every model constant |

These four files form **one inseparable bundle**. Never combine a `.msh` from
one generation run with `.npy` files from another: the arrays depend on the
final mesh node ordering, and a mismatch is not always caught by the length
check.

Two further files are written alongside them and are not solver inputs:

- `coord_offsets.npy` — the original X/Y centre subtracted during meshing. Only
  `utilities/resample_land_cover_on_existing_mesh.py` reads it.
- `susceptible_equilibrium.npy` / `.json` — the cached disease-free susceptible
  field, written by the *first* solver run on the bundle and reused by every
  later one. See [`README_solver.md`](README_solver.md#disease-free-spin-up).

Carrying capacity is **not** stored in the bundle: the solvers call
`land_cover_to_carrying_capacity()` at startup. Likewise
`land_cover_classes.xdmf` is a solver *output* written for visualization, not a
mesh-generation input.

---

## Model parameters

`parameters.json` is the single configuration input for both mesh generation
and the solvers. It holds mesh controls, terrain-diffusion settings,
land-cover mappings, time settings, reaction parameters, initial conditions,
spin-up settings, and the optimal-control block. `land_cover.class_names`
spells out each NLCD code.

Every value carries its justification in a `_notes` block sitting beside it,
tagged `[LIT]`, `[DEF]`, `[CAL]`, `[GUESS]`, or `[NUMERICAL]`. The same
material is written up with derivations in `CWD_optimal_control.tex`, under
"Parameter Values and Their Provenance".

`utilities/shared_parameters.py` is the single authoritative loader and holds
the only implementations of the land-cover-to-diffusivity and
land-cover-to-carrying-capacity conversions. JSON object keys representing NLCD
classes are strings because JSON requires string keys; the loader converts them
to integer class codes.

Command-line values such as `--hmax` and `--isotropy` override the corresponding
value for that mesh run. With no override, the value comes from
`parameters.json`.

### Parameters that feed the mesher as well as the solver

`diffusion_tensor.isotropy` drives the MMG refinement metric *and* the PDE
tensor, and the `land_cover.diffusivity` table is baked into the mesh bundle as
`land_cover_diffusivity.npy`. Changing either should be followed by mesh
regeneration, not just a new solver run. Changing a *carrying-capacity* value
affects only the solver and needs no new mesh.

Important mesh parameters:

- `downsample` — elevation grid stride before the initial triangulation.
- `sigma` — Gaussian elevation smoothing strength (`--no-smooth` disables it).
- `hmax` — nominal, coarsest target edge scale in raster coordinate units.
- `isotropy` — minimum slope-direction scale ratio, currently `0.02`. This is a
  degeneracy floor, not a movement parameter; see the `_notes` entry. It also
  sets the finest edge length, `hmin = isotropy * hmax` (here `3 m`).

---

## The pipeline

Every workflow script exposes its paths and parameters as command-line flags.
Run any of them with `--help` for the complete interface. Defaults reproduce
the production configuration.

**A note on file extensions.** The examples below write `.tif`, but `.tiff`
works exactly the same and needs no flag — MRLC downloads in particular often
arrive as `.tiff`. GDAL identifies a GeoTIFF from the file's contents, not its
name, so every `--dem` and `--land-cover` path accepts either spelling (or no
extension at all). The `.tif` in the defaults is just a default string, not a
requirement; `.gitignore` covers both.

### 1. Prepare the DEM

Download an elevation GeoTIFF for the study area from the USGS
[National Map Downloader](https://www.usgs.gov/tools/download-data-maps-national-map).
The 3D Elevation Program (3DEP) products provide suitable DEMs; choose a
resolution appropriate for the size of the study area and the computing
resources available.

**3DEP tiles arrive in geographic coordinates — units of degrees — and the
mesher needs metres.** `mesh.hmax`, the Gaussian outbreak widths, and
`diffusion_tensor.kappa` are all lengths in the mesh's own units, and the slope
that drives the anisotropic diffusion tensor is only meaningful when horizontal
and vertical units agree. A DEM left in degrees is wrong in all of those places.

**Cropping does not fix this.** Clipping in QGIS is CRS-preserving, so a crop of
a geographic DEM is still in degrees. Reprojection is a separate operation.

`utilities/reproject_dem.py` does the crop and the reprojection in one pass, so
an intermediate degrees-crop never has to exist:

```bash
python utilities/reproject_dem.py --dem dem.tif --output region_utm.tif \
  --bounds -107.70 38.43 -107.22 38.91
```

With no `--target-crs` it selects the UTM zone containing the cropped region's
centroid; with no `--bounds` it reprojects the whole tile. It refuses to run on
an already-projected raster (`--force` overrides, for changing zone or
resolution), and it reports the triangulation size the result would hand to the
mesher.

Two defaults are deliberate:

- **Resampling is bilinear, not nearest.** Nearest-neighbour on continuous
  elevation leaves stair-steps that become spurious slope, feeding straight into
  `cos θ` and the diffusion tensor. Nearest is correct for the categorical NLCD
  raster in step 2 and wrong here.
- **Output resolution defaults to 30 m**, not the native ~10 m. Nothing
  downstream can use finer — NLCD is natively 30 m and `mesh.hmax = 150`
  coarsens edges to 150 m regardless — and at landscape scale 10 m produces a
  triangulation too large to remesh. A 43 × 54 km region is ~290k vertices at
  30 m, ~2.6M at 10 m, and ~26k at 100 m.

Cropping keeps the mesh manageable and avoids terrain irrelevant to the
simulation. To clip in QGIS instead, use **Raster > Extraction > Clip Raster by
Extent** (or **Clip Raster by Mask Layer**; see the
[raster extraction documentation](https://docs.qgis.org/latest/en/docs/user_manual/processing_algs/gdal/rasterextraction.html)),
then follow with **Raster > Projections > Warp (Reproject)** — target CRS the
appropriate UTM zone, resampling **Bilinear**, resolution 30.

Confirm the result before meshing:

```bash
gdalinfo region_utm.tif | head -20
```

You want a `PROJCS[... UTM zone ...]` block and a pixel size in metres.

If the DEM still has a nodata border, crop it away:

```bash
python utilities/crop_dem_valid_window.py --dem region_utm.tif \
  --output region_cropped.tif --margin 40 40 30 30
```

The mesher can fill nodata with the mean, but removing a large invalid border
avoids creating artificial flat terrain around the study area.

Finally, download an NLCD land-cover TIFF from the
[MRLC Viewer](https://www.mrlc.gov/viewer/) using its Data Download tool. The
NLCD raster does **not** need the same bounds, resolution, or CRS as the DEM. It
only needs to completely contain the DEM's geographic extent; a much larger NLCD
region is fine, because it is sampled directly at the mesh nodes.

### 2. Check that the land cover covers the DEM

```bash
python check_land_cover_coverage.py   --land-cover land_cover.tif   --dem region_cropped.tif
```

The land-cover raster is **not** reprojected or resampled onto the DEM grid.
`generate_mesh_and_attributes.py` samples it directly at the mesh nodes,
warping the query points into the raster's own CRS, so the tile may be in any
CRS at any resolution. That is one nearest-neighbour lookup instead of two
chained ones, which matters for categorical data: every resampling step shifts
class boundaries by up to a pixel.

What can still go wrong is the tile being somewhere else. That failure is
silent: NLCD uses class 0 for background, nodes outside the raster are recorded
as class 0, and class 0 maps to a real diffusivity and a real carrying capacity
in `parameters.json` — so a tile that misses the DEM gives a mesh bundle that
looks entirely valid and a model running on a uniform landscape with no
land-cover structure at all. This script exits non-zero on that case and prints
the lon/lat box to re-download.

Step 3 runs the same check itself before it starts meshing, and additionally
audits the sampled node classes afterwards, so step 2 is optional — it just
lets you test a download without committing to a mesh run. `--allow-partial`
overrides the refusal in both places when a partial overlap really is intended.

### 3. Generate the terrain `.msh` and node attributes

```bash
python generate_mesh_and_attributes.py \
  --dem region_cropped.tif \
  --land-cover land_cover.tiff \
  --parameters parameters.json \
  --output-folder outputs \
  --downsample 3 \
  --sigma 3 \
  --hmax 150 \
  --isotropy 0.02
```

`--output-folder` supplies consistent default locations for the whole bundle;
any individual output-path flag still overrides its own location. The script
validates the numeric ranges before reading or writing mesh data.

It carries out the full production path:

1. Reads and optionally smooths/downsamples the elevation TIFF.
2. Creates a triangulated surface from pixel-centre coordinates.
3. Centres X, Y, and Z for numerical stability and saves the original X/Y
   centre as `coord_offsets.npy`.
4. Samples aligned NLCD classes at the initial vertices.
5. Builds a surface metric from slope and land-cover diffusivity contrast.
6. Remeshes with MMG.
7. Adds the `terrain_surface` physical group and writes a Gmsh 2.2 `.msh`.
8. Reloads that **final** mesh through DOLFINx, samples NLCD again in final
   node order, and writes `land_cover_classes.npy` and
   `land_cover_diffusivity.npy`.
9. Writes `effective_parameters.json` — `parameters.json` with every CLI
   override folded in.

**Use the generated `effective_parameters.json` for the matching solver run**,
not `parameters.json`. That is what stops `isotropy` and the land-cover
mappings from drifting between meshing and simulation, and it is what the
solvers pick up automatically from `--mesh-folder`.

Because mesh generation is expensive, note that the initial-condition Gaussian
centres in `parameters.json` are expressed in the **centred** coordinates stored
in `terrain.msh`, not in the original raster coordinates. `coord_offsets.npy`
records the offset if you need to convert.

### 4. Validate the bundle

The generated folder looks like this:

```text
outputs/
├── terrain.msh                     <- solver input
├── effective_parameters.json       <- solver input
├── land_cover_classes.npy          <- solver input
├── land_cover_diffusivity.npy      <- solver input
├── coord_offsets.npy               <- for re-sampling attributes later
├── terrain_input.vtk               <- intermediate, inspection only
└── terrain_remeshed.mesh           <- intermediate, inspection only
```

Before running a solver, check that:

- the `.msh` opens with DOLFINx as a 2-D surface with geometric dimension 3;
- each `.npy` array length equals the scalar CG1 function-space local array
  length for the same mesh (the solvers check this and refuse otherwise);
- the class codes are the NLCD codes you expect;
- the diffusivity values are finite and inside the intended map range;
- the folder holds exactly one `.msh`, unless you will pass `--mesh`
  explicitly.

### 5. Optional: export for ParaView

```bash
python utilities/export_mesh_to_xdmf.py --mesh outputs/terrain.msh \
  --output outputs/terrain.xdmf
```

Useful for inspecting the mesh before committing to a long run. The solvers do
not read it.

### Re-deriving the arrays without remeshing

If the land-cover *tables* change but the mesh does not, the two `.npy` arrays
can be regenerated against the existing mesh:

```bash
python utilities/resample_land_cover_on_existing_mesh.py \
  --mesh outputs/terrain.msh \
  --land-cover land_cover.tif \
  --dem region_cropped.tif \
  --coord-offsets outputs/coord_offsets.npy \
  --classes-output outputs/land_cover_classes.npy \
  --diffusivity-output outputs/land_cover_diffusivity.npy
```

The mesh geometry does not depend on land cover at all, so this covers a
changed *carrying-capacity* table **and** a changed `land_cover.diffusivity`
table — neither needs a new mesh. `--dem` is read only for its CRS, which is
the CRS the mesh coordinates are in.

It is **not** a substitute for remeshing after a change to `isotropy`, the
activation parameters, `hmax`, or the DEM itself, all of which set the
refinement metric.

---

## Land-cover mappings

Both mappings live in `parameters.json` and are applied by
`utilities/shared_parameters.py`. The two columns measure different things and
must not be collapsed into one:

- **carrying capacity** `c(x)` measures *habitat quality* — how many deer a
  cover type supports;
- **diffusivity** `ℓ(x)` measures *permeability* — how readily deer move
  through it.

For several classes the two are anti-correlated. Open grassland is highly
permeable but poor white-tailed deer habitat. Woody wetlands are the reverse:
prime cover and winter browse, but slow going. Open water has zero carrying
capacity and a diffusivity of `0.01` — deer do swim, so large water bodies are
strong but not absolute barriers.

Class `0` (unknown/nodata) defaults to a mid-range `0.5` in both tables, rather
than to prime maximally-permeable habitat, so that missing NLCD coverage
degrades gently instead of inventing good habitat. Review the zero/nodata count
from step 2 before accepting a run.

Every entry in both tables is expert judgement with no supporting citation for
any particular study region, and both should be calibrated against regional
density estimates stratified by cover type.

---

## Example bundle

`example_mesh_output/` is a complete, ready-to-run bundle for a small test
region, kept so the solvers can be exercised without running the whole pipeline
first:

```bash
python CWD_solver.py --mesh-folder example_mesh_output \
  --output-folder solver_outputs
```

It holds 38,980 nodes and carries ten NLCD classes (11, 21, 22, 31, 41, 42, 52,
71, 90, 95), with a nodal diffusivity multiplier spanning the full table range,
0.01 over open water to 1.0 on grassland and pasture. No node received class 0,
so the whole mesh sits inside the land-cover tile. Its
`effective_parameters.json` records the production configuration it was built
with — `hmax = 150`, `isotropy = 0.02`, `downsample = 3`, `sigma = 3` — and the
full 20-year horizon, so a solver run against it is not a smoke test and will
take a while; drop `time.final_time_years` to shorten it.

The bundle was generated with the current pipeline, so it samples land cover
directly at the mesh nodes and has no aligned intermediate raster. The two
mesher scratch files left in the folder, `terrain_input.vtk` and
`terrain_remeshed.mesh`, are gitignored and are not read by anything
downstream. `susceptible_equilibrium.npy` and its `.json` sidecar appear after
the first solver run caches them there — see
[Disease-free spin-up](README_solver.md#disease-free-spin-up).

The source GeoTIFFs it was built from are not included; they are large and are
re-downloadable from the two links in step 1.

---

## Folder contents

### Mesh-generation scripts

- `parameters.json` — combined mesh, diffusion, land-cover, time,
  initial-condition, spin-up, and reaction configuration, with a `_notes`
  block of provenance beside every value.
- `check_land_cover_coverage.py` — step 2: verify the NLCD tile covers the DEM.
  Optional; step 3 runs the same check itself.
- `generate_mesh_and_attributes.py` — step 3: the integrated
  TIFF-to-MSH-plus-attributes pipeline.
- `land_cover_to_capacity.py` — thin compatibility wrapper around the shared
  carrying-capacity mapping.
- `environment.yml` — the Conda environment.
- `example_mesh_output/` — the one committed mesh bundle, so the solvers can be
  run without building a mesh first. See [Example bundle](#example-bundle).

### `utilities/`

- `shared_parameters.py` — the validated loader and the authoritative
  land-cover conversion functions.
- `verify_environment.py` — package and version check, including the PyVista
  `.mmg` accessor.
- `smoke_test_fenics.py` — minimal DOLFINx/PETSc assembly check.
- `reproject_dem.py` — step 1: reproject a DEM from geographic degrees to
  projected metres, cropping in the same pass.
- `crop_dem_valid_window.py` — step 1: crop a DEM to its valid-data window.
- `land_cover_sampling.py` — sampling the land-cover raster at mesh nodes,
  plus the coverage and class-0 checks. Shared by step 3 and the re-sampler.
- `resample_land_cover_on_existing_mesh.py` — regenerate the two `.npy` arrays
  for an already-created mesh.
- `export_mesh_to_xdmf.py` — optional ParaView export.

The remaining files in `utilities/` — `mesh_bundle.py`, `block_solver.py`, and
`susceptible_spinup.py` — belong to the solvers and are documented in
[`README_solver.md`](README_solver.md).
