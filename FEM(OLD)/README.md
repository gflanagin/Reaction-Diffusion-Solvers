# FEM terrain mesh and land-cover workflow

## Software environment

Run the workflow in Linux. The tested setup is WSL2 with Ubuntu 24.04 and the
Conda environment defined by `environment.yml` (Python 3.11, DOLFINx 0.11,
OpenMPI 5.0, Gmsh 4.15, meshio 5.3, Rasterio 1.4, SciPy 1.16, PyVista, and
mmgpy 0.16).

For a first-time setup, install WSL/Ubuntu if needed, install Miniforge or
another Conda-compatible manager inside Ubuntu, then run from this directory:

```bash
conda env create --file environment.yml
conda activate cwd-fem
```

For normal use, activate the environment in each new shell with
`conda activate cwd-fem`. After `environment.yml` changes, update it with:

```bash
conda env update --name cwd-fem --file environment.yml --prune
conda activate cwd-fem
```

Keep DOLFINx, PETSc, HDF5, MPI, Basix, UFL, and their Python bindings in this
Conda environment so their compiled versions remain compatible. ParaView is
optional and can be installed separately. See `ENVIRONMENT.md` for additional
development and troubleshooting notes.

## What the PDE solver requires

A solver run needs these three matching files in its working directory:

| File | Contents | Used by the solver as |
| --- | --- | --- |
| `terrain.msh` | 2-D triangular terrain surface embedded in 3-D | DOLFINx mesh (`gdim=3`) |
| `land_cover_classes.npy` | one NLCD class code per CG1 mesh degree of freedom/node | input to carrying-capacity mapping |
| `land_cover_diffusivity.npy` | one diffusivity multiplier per CG1 mesh degree of freedom/node | spatial multiplier in the diffusion tensor |

`land_cover_classes.xdmf` is **not** a mesh-generation input. `CWD_solver.py`
creates it for visualization when the solver starts. Likewise, carrying
capacity is not stored in a separate mesh file: the solver calls
`land_cover_to_carrying_capacity()` at startup.

The three required files form one inseparable bundle. Never combine a `.msh`
from one generation run with `.npy` files from another run, because the arrays
depend on the final mesh node ordering.

## Model parameters

`parameters.json` is the single configuration input for mesh generation
and the solver. It contains mesh controls, terrain diffusion settings,
land-cover mappings, time settings, and reaction parameters. The
`land_cover.class_names` object explains each NLCD code (for example, open
water, evergreen forest, or cultivated crops).

`utilities/shared_parameters.py` is the single authoritative loader and contains the only
implementations of the land-cover-to-diffusivity and
land-cover-to-carrying-capacity conversions. JSON object keys representing NLCD
classes are strings because JSON requires string keys; the loader converts them
to integer class codes.

Command-line values such as `--hmax` and `--isotropy` override the corresponding
spatial-file value for that mesh run. If no override is supplied, the value is
read from `parameters.json`.

### Infected initial conditions

The solver initializes the infected (`I`) population from
`initial_conditions.infected_gaussians`. Add as many Gaussian entries as
needed, or use an empty list for an initially uninfected population:

```json
"initial_conditions": {
  "infected_gaussians": [
    {
      "center": [162.0, 139.0],
      "mean": 0.01,
      "standard_deviation": 500.0
    }
  ]
}
```

For each Gaussian, `center` is its `[x, y]` location, `mean` is the population
value at that center (the Gaussian peak), and `standard_deviation` controls its
spatial spread. Centers and standard deviations use the mesh coordinate units;
because mesh generation centers the terrain for numerical stability, centers
refer to the centered coordinates stored in `terrain.msh`. Contributions from
overlapping Gaussians are added together. The susceptible population starts at
carrying capacity, while the dying and environmental populations start at zero.

## Mesh generation workflow

The supported workflow scripts expose their paths and parameters as command-line
flags. Run either script with `--help` to see the complete interface. Defaults
preserve the filenames and numerical settings used by the original copies.

### 1. Prepare the DEM

Download an elevation GeoTIFF for the study area from the USGS
[National Map Downloader](https://www.usgs.gov/tools/download-data-maps-national-map).
The 3D Elevation Program (3DEP) products provide suitable DEMs; choose a
resolution appropriate for the size of the study area and the available
computing resources.

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
then either run the script on the clipped file or follow with **Raster >
Projections > Warp (Reproject)** — target CRS the appropriate UTM zone,
resampling **Bilinear**, resolution 30.

Confirm the result before meshing:

```bash
gdalinfo region_utm.tif | head -20
```

You want a `PROJCS[... UTM zone ...]` block and a pixel size in metres.

Download an NLCD land-cover TIFF from the
[MRLC Viewer](https://www.mrlc.gov/viewer/) using its Data Download tool. Select
the desired NLCD land-cover year/product. The NLCD raster does **not** need to
have the same bounds, resolution, or coordinate reference system as the cropped
DEM. It only needs to completely contain the cropped DEM's geographic extent;
using a much larger NLCD region is fine because Step 2 reprojects and clips it
to the DEM grid automatically.

If the cropped DEM still has a nodata border,
`utilities/crop_dem_valid_window.py` is available as a reference helper,
although its crop settings must be adapted to the data. The mesher can fill
nodata with the mean, but removing a large invalid border avoids creating
artificial flat terrain around the study area.

### 2. Reproject and align the land-cover raster

Run `align_land_cover.py`. It performs two categorical-raster
operations:

1. Reproject the source NLCD TIFF into the DEM coordinate reference system.
2. Resample it onto the DEM grid using nearest-neighbor resampling.

Expected result: `land_cover_aligned.tif` with the DEM's CRS, transform, width, and
height. Nearest-neighbor resampling is essential because NLCD values are class
codes, not continuous measurements.

Example:

```bash
python align_land_cover.py \
  --input-land-cover inputs/land_cover.tif \
  --reprojected-land-cover outputs/land_cover_reprojected.tif \
  --dem inputs/dem.tif \
  --output outputs/land_cover_aligned.tif
```

`--target-crs` is optional. When omitted, the script reads the target CRS from
the DEM. The other defaults are:

- source land cover: `land_cover.tif`
- intermediate projection: `land_cover_reprojected.tif`
- reference DEM: `dem.tif`
- aligned result: `land_cover_aligned.tif`

Use `utilities/check_raster_alignment.py` after adapting its filenames. Confirm
that DEM and aligned NLCD CRS/bounds are compatible and inspect the listed NLCD
classes and zero/nodata count.

### 3. Generate the terrain `.msh` and node attributes

Run `generate_mesh_and_attributes.py` in the FEniCSx environment.
All external and intermediate paths and all numerical tuning parameters are
flags.

Example:

```bash
python generate_mesh_and_attributes.py \
  --dem inputs/dem.tif \
  --land-cover outputs/land_cover_aligned.tif \
  --parameters parameters.json \
  --output-folder outputs \
  --downsample 3 \
  --sigma 3 \
  --hmax 150 \
  --isotropy 0.02 \
  --min-lc-squish 0.1
```

Use `--no-smooth` to disable Gaussian smoothing. The script validates the
numeric ranges before reading or writing mesh data. It also writes an effective
model parameter file containing any CLI overrides. Use that generated file for
the matching solver run so `isotropy` and the land-cover mappings cannot drift
between meshing and simulation. `--output-folder` supplies consistent default
locations for the entire bundle; any individual output-path flag can still
override its corresponding location.

The script carries out the full production path:

1. Reads and optionally smooths/downsamples the elevation TIFF.
2. Creates a triangulated surface from pixel-center coordinates.
3. Centers X, Y, and Z for numerical stability and saves the original X/Y
   center as `coord_offsets.npy`.
4. Samples aligned NLCD classes at initial vertices.
5. Builds a surface metric from slope and land-cover diffusivity contrast.
6. Remeshes with MMG.
7. Adds the `terrain_surface` physical group and writes Gmsh 2.2 `.msh`.
8. Reloads that **final** mesh through DOLFINx, samples NLCD again in final node
   order, and writes `land_cover_classes.npy` and
   `land_cover_diffusivity.npy`.

Important parameters:

- `downsample`: elevation grid stride before initial triangulation.
- `sigma`: Gaussian elevation smoothing strength.
- `hmax`: nominal/coarsest target edge scale in raster coordinate units.
- `isotropy`: minimum slope-direction scale ratio. 
  This must match the value used during mesh generation; its current value is `0.02`.
- `min_lc_squish`: extra refinement ratio across sharp land-cover boundaries.

The default production configuration reads `dem.tif` and
`land_cover_aligned.tif`, writes `terrain.msh`, and uses `isotropy=0.02`.

### 4. Keep and validate the generated bundle

Keep these outputs together:

```text
outputs/
├── terrain.msh
├── effective_parameters.json
├── land_cover_classes.npy
├── land_cover_diffusivity.npy
├── coord_offsets.npy
├── terrain_input.vtk
└── terrain_remeshed.mesh
```

The first four files are the solver bundle. `coord_offsets.npy` and the two
intermediate mesh files are useful for regenerating attributes or inspecting
the mesh-generation process, but the PDE solver does not read them.

Before running the PDE solver, check that:

- the `.msh` opens with DOLFINx as a 2-D surface with geometric dimension 3;
- each `.npy` array length equals the scalar CG1 function-space local array
  length for the same mesh;
- class codes are expected NLCD codes;
- diffusivity values are finite and within the intended map range;
- the folder contains exactly one `.msh` file, unless an explicit `--mesh`
  override will be supplied to the solver.

Generate the arrays and consume them with the same MPI layout. The present
workflow is safest when mesh generation and the solver are run serially. A
parallel solver partitions/reorders local degrees of freedom, while the plain
NumPy arrays contain no global node IDs with which to redistribute values.

### 5. Optional visualization export

`utilities/export_mesh_to_xdmf.py` writes an XDMF representation of a `.msh`.
Its default files are `terrain.msh` and `terrain.xdmf`. This is useful for
inspection in ParaView but is not required by the solver.

## Running the solver

Call `CWD_solver.py` with --mesh-folder pointing at a folder containing the mesh files described in the previous section.

```bash
python CWD_solver.py \
  --mesh-folder outputs \
  --output-folder solver_outputs
```

`--mesh-folder` discovers:

- the folder's single `*.msh` file;
- `land_cover_classes.npy`;
- `land_cover_diffusivity.npy`;
- `effective_parameters.json`, falling back to `parameters.json` if needed.

The solver stops with a descriptive error if a required file is absent, if the
folder contains more than one `.msh`, or if either nodal array length does not
match the loaded mesh. Individual `--mesh`, `--land-cover-classes`,
`--land-cover-diffusivity`, and `--parameters` flags remain available
and override folder discovery when needed. `--output-folder` selects where the solver writes
`CWD_S`, `CWD_I`, `CWD_D`, `CWD_E`, and land-cover visualization XDMF/HDF5
files; it defaults to `solver_outputs` and contains no user-specific path.

## Land-cover mappings

Both mappings live in `parameters.json` and are applied by
`utilities/shared_parameters.py`. The default diffusivity multipliers range from `0.0001`
for open water to `1.0` for unknown/nodata. Open water has a zero carrying-
capacity fraction; other class fractions multiply `carrying_capacity_base`.

Changing a carrying-capacity value requires only a new solver run. Changing a
diffusivity value should normally be followed by mesh regeneration because the
diffusivity contrast affects both the MMG refinement metric and the PDE tensor.

Class `0` currently defaults to full diffusivity and full carrying capacity.
That may be useful for avoiding accidental barriers, but it can also hide
missing NLCD coverage. Review the zero/nodata count before accepting a run.

## Folder contents

### Root configuration

- `parameters.json` — combined mesh, diffusion, land-cover, time, initial-condition, and
  reaction configuration, including human-readable NLCD class names.
- `align_land_cover.py` — flag-driven copy of `land_cover.py`; raster
  preparation.
- `generate_mesh_and_attributes.py` — flag-driven copy of `create_mesh.py`;
  current integrated TIFF-to-MSH and land-cover attachment pipeline.
- `land_cover_to_capacity.py` — copy of `LC_to_capacity.py`; solver-side
  compatibility wrapper around the shared mapping.
- `CWD_solver.py` — parameterized copy of `CWD_model.py` wired to the combined
  parameter file. Its normal interface is `--mesh-folder`; individual file
  flags are optional overrides. The copied solver's old user-specific output
  paths and stale wolf-model names have been removed.

Example invocation:

```bash
python CWD_solver.py \
  --mesh-folder outputs \
  --output-folder solver_outputs
```

### `utilities/`

- `shared_parameters.py` — validated loader and authoritative land-cover
  conversion functions.
- `verify_environment.py` and `smoke_test_fenics.py` — environment validation
  and distributed FEniCSx assembly checks.
- `reproject_dem.py` — step 1: reproject a DEM from geographic degrees to
  projected metres, cropping in the same pass. Auto-selects the UTM zone,
  defaults to bilinear resampling at 30 m, and warns when the result would
  produce a triangulation too large to remesh.
- `crop_dem_valid_window.py` — optional DEM crop helper.
- `check_raster_alignment.py` — raster metadata/class diagnostic.
- `resample_land_cover_on_existing_mesh.py` — regenerates the two `.npy` arrays
  for an already-created mesh using saved coordinate offsets.
- `export_mesh_to_xdmf.py` — optional ParaView/export helper.
