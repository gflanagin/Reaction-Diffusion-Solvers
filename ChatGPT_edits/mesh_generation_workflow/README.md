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

## Recommended workflow

The supported workflow scripts expose their paths and parameters as command-line
flags. Run either script with `--help` to see the complete interface. Defaults
preserve the filenames and numerical settings used by the original copies.

### 1. Prepare the DEM

Download an elevation GeoTIFF for the study area from the USGS
[National Map Downloader](https://www.usgs.gov/tools/download-data-maps-national-map).
The 3D Elevation Program (3DEP) products provide suitable DEMs; choose a
resolution appropriate for the size of the study area and the available
computing resources.

The downloaded DEM should usually be cropped to the intended simulation area
before running the mesher. In QGIS, load the DEM and use **Raster > Extraction
> Clip Raster by Extent** (or **Clip Raster by Mask Layer**) to save the study
area as a new GeoTIFF. See the QGIS
[raster extraction documentation](https://docs.qgis.org/latest/en/docs/user_manual/processing_algs/gdal/rasterextraction.html)
for details. Cropping keeps the mesh manageable and avoids including terrain
that is irrelevant to the simulation.

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

### 5. Run the solver

Pass the generated output directory to the solver; no mesh filenames need to be
edited in Python:

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

### 6. Optional visualization export

`utilities/export_mesh_to_xdmf.py` writes an XDMF representation of a `.msh`.
Its default files are `terrain.msh` and `terrain.xdmf`. This is useful for
inspection in ParaView but is not required by `CWD_model.py`, which reads the
Gmsh file directly.

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
- `crop_dem_valid_window.py` — optional DEM crop helper.
- `check_raster_alignment.py` — raster metadata/class diagnostic.
- `inspect_tif.py` and `inspect_msh.py` — older inspection aids with hard-coded
  filenames; adapt before use.
- `resample_land_cover_on_existing_mesh.py` — regenerates the two `.npy` arrays
  for an already-created mesh using saved coordinate offsets.
- `export_mesh_to_xdmf.py` — optional ParaView/export helper.

### `legacy_reference/`

These are copied for completeness, not part of the recommended path:

- `terrain_only_mmgs_mesher.py` — older terrain-only MMG route; it imports an
  `activation` module that is not present as a source file in the original
  folder.
- `simple_dem_to_xdmf.py` — direct structured DEM triangulation without the
  current MMG/land-cover pipeline.
- `simple_msh_to_xdmf.py` — older meshio conversion.
- `dem_to_stl.py` — early TIFF-to-STL route.
- `older_tif_stl_diagnostic.py` — diagnostic for that STL route.

## Should these scripts be combined?

The two production steps should remain separate internally. Raster alignment is
often done once, while mesh parameters are tuned and the mesh is regenerated
many times; keeping the aligned TIFF lets those mesh iterations avoid repeating
a potentially expensive reprojection.

A small **runner/orchestrator** that invokes both steps from one command would
make sense. It could offer `align`, `mesh`, and `all` subcommands and pass the
aligned raster directly into the mesher. That gives a one-command workflow
without turning the two reusable stages into one large monolithic file. I would
not merge `land_cover_to_capacity.py` into the mesher because `CWD_model.py`
imports that mapping at solver runtime.

## Known issues observed in the existing files

- `create_mesh.py` saves land-cover class codes as float64; the solver casts
  them back to int32, so this works but is unnecessarily indirect.
- `LC_diffusivity.py` reports water using `diffusivity == 0.0`, but the mapping
  assigns water `0.0001`; that printed count will therefore be misleading.
- `CWD_model.py` contains references later in the file to names not defined in
  the displayed solver (`S_func`, `I_func`, `D_func`, `dof_x`, `dof_y`,
  `xdmf_N`, `xdmf_W`, and `wolf_log`). These are solver issues rather than
  mesh-generation steps, but they will prevent the original file from
  completing as-is. They have been removed from the parameterized review copy;
  the original file remains untouched.
