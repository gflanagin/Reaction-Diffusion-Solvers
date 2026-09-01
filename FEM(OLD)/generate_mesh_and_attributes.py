from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI
import numpy as np
import rasterio
from rasterio.transform import rowcol

import mmgpy
import meshio
import pyvista as pv
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter
import argparse
import json
from pathlib import Path
import sys

WORKFLOW_ROOT = Path(__file__).resolve().parent
UTILITIES_DIR = WORKFLOW_ROOT / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import (  # noqa: E402
    DEFAULT_PARAMETERS,
    land_cover_to_diffusivity,
    load_parameters,
)

def compute_metric(verts, faces, hmax, isotropy, min_lc_squish,
                   activation_steepness, activation_cosine_threshold,
                   lc_grad=None, lc_diffusivity=None,
                   lc_neighbor_max=None, lc_neighbor_min=None):
    eps = 1e-6
    normals = np.zeros((len(verts), 3))
    counts = np.zeros(len(verts))

    for face in faces:
        pts = verts[face]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if normal[2]<0:
            normal = -normal

        if norm_len < 1e-10:
            continue
        normal /= norm_len

        for i in face:
            normals[i] += normal
            counts[i] += 1

    mask = counts > 0
    normals[mask] /= counts[mask, np.newaxis]
    # Renormalize after averaging
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms)

    g = np.array([0,0,-1])

    metrics = np.zeros((len(verts), 6))
    for i, normal in enumerate(normals):

        cos_theta = np.dot(g, normal)
        es = g-cos_theta*normal
        es_norm = np.linalg.norm(es)

        if es_norm < eps:
            M_slope = (np.eye(3) - np.outer(normal,normal)) 
        else:
            es = es / es_norm
            ec = np.cross(es,normal)

            activation = (
                1 + np.exp(activation_steepness * (activation_cosine_threshold - 1))
            ) / (
                1 + np.exp(
                    activation_steepness
                    * (activation_cosine_threshold - abs(cos_theta))
                )
            )

            h_s = isotropy + (1-isotropy)*activation

            M_slope = (1/h_s) * np.outer(es, es) + np.outer(ec, ec)

        if lc_grad is not None:
            g_lc = lc_grad[i]
            norm_glc = np.linalg.norm(g_lc)

            if norm_glc < eps:
                M_lc = np.eye(3) - np.outer(normal, normal)
            else:
                e_perp = g_lc / norm_glc  # across boundary
                e_par = np.cross(normal, e_perp)  # along boundary
                norm_par = np.linalg.norm(e_par)
                e_par = e_par/norm_par

                lc_low = min(lc_diffusivity[i], lc_neighbor_min[i])
                lc_high = max(lc_diffusivity[i], lc_neighbor_max[i])
                
                r = min_lc_squish + (1-min_lc_squish)*(lc_low / lc_high) if lc_high > eps else 1.0
                
                M_lc = (1/r) * np.outer(e_perp, e_perp) +  np.outer(e_par, e_par) 
        else:
            M_lc = np.eye(3) - np.outer(normal, normal)
                
        M = (1/hmax**2) * (M_slope @ M_lc @ M_lc @ M_slope + M_lc @ M_slope @ M_slope @ M_lc)/2
        metrics[i] = [M[0,0], M[0,1], M[0,2],
                            M[1,1], M[1,2],
                                    M[2,2]]
    return metrics

def dem_to_msh(input_tif, output_msh, nlcd_aligned_path,
               spatial_parameters,
               coord_offsets_path="coord_offsets.npy",
               terrain_input_path="terrain_input.vtk",
               remeshed_path="terrain_remeshed.mesh",
               downsample=3, smooth=True, sigma=2, hmax=400,
               isotropy=.1, min_lc_squish=.1):

    # --- Read DEM ---
    print(f"Reading {input_tif}...")
    with rasterio.open(input_tif) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform

    if nodata is not None:
        data[np.isclose(data, nodata)] = np.nan
    if np.isnan(data).any():
        print(f"Filling {np.isnan(data).sum()} nodata pixels...")
        data[np.isnan(data)] = np.nanmean(data)

    # Smooth before meshing
    if smooth:
        data = gaussian_filter(data, sigma=sigma)

    data = data[::downsample, ::downsample]
    rows, cols = data.shape
    print(f"Grid: {cols} x {rows}")

    col_indices = np.arange(0, cols * downsample, downsample)
    row_indices = np.arange(0, rows * downsample, downsample)
    xs, _ = xy(transform, np.zeros_like(col_indices), col_indices)
    _, ys = xy(transform, row_indices, np.zeros_like(row_indices))
    X, Y = np.meshgrid(np.array(xs), np.array(ys))
    Z = data.copy()
    
    x_offset = X.mean()  # before centering
    y_offset = Y.mean()
    np.save(coord_offsets_path, np.array([x_offset, y_offset]))
    X -= x_offset
    Y -= y_offset
    Z -= Z.mean()

    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    idx = np.arange(rows * cols).reshape(rows, cols)
    r, c = rows - 1, cols - 1
    i00 = idx[:r, :c].ravel()
    i10 = idx[1:, :c].ravel()
    i01 = idx[:r, 1:].ravel()
    i11 = idx[1:, 1:].ravel()
    faces = np.concatenate([
        np.stack([i00, i10, i11], axis=1),
        np.stack([i00, i11, i01], axis=1)
    ], axis=0).astype(np.int32)

    # Filter degenerate triangles
    v1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    areas = np.linalg.norm(np.cross(v1, v2), axis=1)
    faces = faces[areas > 1e-6]
    print(f"Triangles: {len(faces)}")

    # --- Write temp mesh file for mmgpy ---
    print("Writing temporary mesh...")
    tmp_mesh = meshio.Mesh(
        points=verts,
        cells=[("triangle", faces)]
    )
    meshio.write(terrain_input_path, tmp_mesh)

    # Sample land cover at mesh vertices
    print("Sampling land cover at mesh vertices...")
    x_offset, y_offset = np.load(coord_offsets_path)
    land_cover = sample_nlcd_at_nodes(nlcd_aligned_path, verts, x_offset, y_offset)
    lc_diffusivity = land_cover_to_diffusivity(land_cover, spatial_parameters)
    lc_grad = compute_lc_gradient(verts, faces, lc_diffusivity)
    lc_neighbor_min, lc_neighbor_max = compute_lc_neighbor_stats(verts, faces, lc_diffusivity)

    #Compute metric
    print("Computing anisotropic metric...")
    tensor_parameters = spatial_parameters["diffusion_tensor"]
    metrics = compute_metric(
        verts,
        faces,
        hmax,
        isotropy,
        min_lc_squish,
        tensor_parameters["activation_steepness"],
        tensor_parameters["activation_cosine_threshold"],
        lc_grad=lc_grad,
        lc_diffusivity=lc_diffusivity,
        lc_neighbor_max=lc_neighbor_max,
        lc_neighbor_min=lc_neighbor_min,
    )

    """
    print(f"Vertex coordinate ranges:")
    print(f"  X: {verts[:,0].min():.1f} to {verts[:,0].max():.1f}")
    print(f"  Y: {verts[:,1].min():.1f} to {verts[:,1].max():.1f}")
    print(f"  Z: {verts[:,2].min():.1f} to {verts[:,2].max():.1f}")
    print(f"hmin={hmin}, hmax={hmax}")
    print(f"Expected metric diagonal range: {1/hmax**2:.8f} to {1/hmin**2:.6f}")

    print(f"Metric stats:")
    print(f"  min: {metrics.min():.6f}")
    print(f"  max: {metrics.max():.6f}")
    print(f"  nan count: {np.isnan(metrics).sum()}")
    print(f"  inf count: {np.isinf(metrics).sum()}")
    """    
    print("Remeshing with mmgpy...")
    mesh = pv.read(terrain_input_path)
    mesh.point_data["metric"] = metrics  # shape (n_verts, 6)
    hmin = isotropy*min_lc_squish*hmax
    remeshed = mesh.mmg.remesh(
        hmin=hmin,
        hmax=hmax,
        hausd=hmax * 0.1,
        hgrad=1.3,
    )
    remeshed.save(remeshed_path)
    

    # --- Add physical groups via Gmsh and write final .msh ---
    print("Adding physical groups and writing .msh...")
    import gmsh
    gmsh.initialize()
    gmsh.open(remeshed_path)

    surfaces = gmsh.model.getEntities(2)
    tags = [s[1] for s in surfaces]
    gmsh.model.addPhysicalGroup(2, tags, tag=1)
    gmsh.model.setPhysicalName(2, 1, "terrain_surface")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_msh)
    gmsh.finalize()

    print(f"Mesh created! Written to {output_msh}")

def sample_nlcd_at_nodes(nlcd_aligned_path, mesh_verts, x_offset, y_offset):
    with rasterio.open(nlcd_aligned_path) as src:
        data = src.read(1)
        transform = src.transform

    xs = mesh_verts[:, 0] + x_offset
    ys = mesh_verts[:, 1] + y_offset

    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(rows, 0, data.shape[0] - 1)
    cols = np.clip(cols, 0, data.shape[1] - 1)

    land_cover = data[rows, cols].astype(np.int32)
    print(f"Unique land cover classes at mesh nodes: {np.unique(land_cover)}")
    return land_cover

def compute_lc_gradient(verts, faces, lc_diffusivity):
    """Compute gradient of lc_diffusivity at each vertex, normalized to max gradient in domain"""
    grad_lc = np.zeros((len(verts), 3))
    counts = np.zeros(len(verts))

    for face in faces:
        pts = verts[face]
        lc_vals = lc_diffusivity[face]
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        dlc1 = lc_vals[1] - lc_vals[0]
        dlc2 = lc_vals[2] - lc_vals[0]
        
        # Solve for gradient on triangle using least squares
        # grad_lc . v1 = dlc1, grad_lc . v2 = dlc2
        A = np.array([[v1[0], v1[1], v1[2]],
                      [v2[0], v2[1], v2[2]]])
        b = np.array([dlc1, dlc2])
        
        # Project onto triangle plane first
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        
        # Surface gradient via least squares
        try:
            gh, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        for i in face:
            grad_lc[i] += gh
            counts[i] += 1

    mask = counts > 0
    grad_lc[mask] /= counts[mask, np.newaxis]
    return grad_lc

def compute_lc_neighbor_stats(verts, faces, lc_diffusivity):
    n_verts = len(verts)
    adjacency = [set() for _ in range(n_verts)]
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i]].add(face[j])
    
    lc_neighbor_min = lc_diffusivity.copy()
    lc_neighbor_max = lc_diffusivity.copy()
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            lc_neighbor_min[i] = min(lc_neighbor_min[i], lc_diffusivity[j])
            lc_neighbor_max[i] = max(lc_neighbor_max[i], lc_diffusivity[j])
    
    return lc_neighbor_min, lc_neighbor_max

def build_parser():
    parser = argparse.ArgumentParser(
        description="Create an anisotropic terrain mesh and matching nodal land-cover arrays."
    )
    parser.add_argument("--dem", default="dem.tif",
                        help="Input elevation GeoTIFF.")
    parser.add_argument("--land-cover", default="land_cover_aligned.tif",
                        help="Land-cover raster aligned to the DEM CRS/grid.")
    parser.add_argument("--parameters", default=str(DEFAULT_PARAMETERS),
                        help="Combined model parameter JSON file.")
    parser.add_argument("--output-folder", default=None,
                        help="Place the complete mesh/attribute bundle in this folder.")
    parser.add_argument("--effective-parameters-output",
                        default=None,
                        help="Resolved model JSON written for the matching solver run.")
    parser.add_argument("--output-msh", default=None,
                        help="Final Gmsh mesh output.")
    parser.add_argument("--classes-output", default=None,
                        help="Output NLCD class array.")
    parser.add_argument("--diffusivity-output", default=None,
                        help="Output diffusivity multiplier array.")
    parser.add_argument("--coord-offsets-output", default=None,
                        help="Output X/Y coordinate offsets array.")
    parser.add_argument("--terrain-input", default=None,
                        help="Intermediate pre-remesh VTK file.")
    parser.add_argument("--remeshed-output", default=None,
                        help="Intermediate MMG mesh file.")
    parser.add_argument("--downsample", type=int, default=None,
                        help="Override mesh.downsample from the model parameter file.")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override mesh.smoothing_sigma.")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=None,
                        help="Override mesh.smooth with --smooth or --no-smooth.")
    parser.add_argument("--hmax", type=float, default=None,
                        help="Override mesh.hmax.")
    parser.add_argument("--isotropy", type=float, default=None,
                        help="Override diffusion_tensor.isotropy.")
    parser.add_argument("--min-lc-squish", type=float, default=None,
                        help="Override mesh.min_land_cover_squish.")
    return parser


def validate_args(args):
    if args.downsample < 1:
        raise ValueError("--downsample must be at least 1.")
    if args.sigma < 0:
        raise ValueError("--sigma cannot be negative.")
    if args.hmax <= 0:
        raise ValueError("--hmax must be positive.")
    if not 0 < args.isotropy <= 1:
        raise ValueError("--isotropy must be in (0, 1].")
    if not 0 < args.min_lc_squish <= 1:
        raise ValueError("--min-lc-squish must be in (0, 1].")


def main():
    args = build_parser().parse_args()
    output_folder = Path(args.output_folder or ".").expanduser()
    args.output_msh = args.output_msh or str(output_folder / "terrain.msh")
    args.classes_output = args.classes_output or str(output_folder / "land_cover_classes.npy")
    args.diffusivity_output = args.diffusivity_output or str(output_folder / "land_cover_diffusivity.npy")
    args.coord_offsets_output = args.coord_offsets_output or str(output_folder / "coord_offsets.npy")
    args.terrain_input = args.terrain_input or str(output_folder / "terrain_input.vtk")
    args.remeshed_output = args.remeshed_output or str(output_folder / "terrain_remeshed.mesh")
    args.effective_parameters_output = args.effective_parameters_output or str(
        output_folder / "effective_parameters.json"
    )

    spatial_parameters = load_parameters(args.parameters)
    mesh_parameters = spatial_parameters["mesh"]
    tensor_parameters = spatial_parameters["diffusion_tensor"]

    if args.downsample is None:
        args.downsample = int(mesh_parameters["downsample"])
    if args.sigma is None:
        args.sigma = float(mesh_parameters["smoothing_sigma"])
    if args.smooth is None:
        args.smooth = bool(mesh_parameters["smooth"])
    if args.hmax is None:
        args.hmax = float(mesh_parameters["hmax"])
    if args.isotropy is None:
        args.isotropy = float(tensor_parameters["isotropy"])
    if args.min_lc_squish is None:
        args.min_lc_squish = float(mesh_parameters["min_land_cover_squish"])

    validate_args(args)

    # Record CLI overrides in the effective configuration consumed by this run.
    mesh_parameters["downsample"] = args.downsample
    mesh_parameters["smooth"] = args.smooth
    mesh_parameters["smoothing_sigma"] = args.sigma
    mesh_parameters["hmax"] = args.hmax
    mesh_parameters["min_land_cover_squish"] = args.min_lc_squish
    tensor_parameters["isotropy"] = args.isotropy

    output_paths = (
        args.output_msh,
        args.classes_output,
        args.diffusivity_output,
        args.coord_offsets_output,
        args.terrain_input,
        args.remeshed_output,
        args.effective_parameters_output,
    )
    for output_path in output_paths:
        Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    with Path(args.effective_parameters_output).open("w", encoding="utf-8") as stream:
        json.dump(spatial_parameters, stream, indent=2)
        stream.write("\n")

    dem_to_msh(
        input_tif=args.dem,
        output_msh=args.output_msh,
        nlcd_aligned_path=args.land_cover,
        spatial_parameters=spatial_parameters,
        coord_offsets_path=args.coord_offsets_output,
        terrain_input_path=args.terrain_input,
        remeshed_path=args.remeshed_output,
        smooth=args.smooth,
        sigma=args.sigma,
        downsample=args.downsample,
        hmax=args.hmax,
        isotropy=args.isotropy,
        min_lc_squish=args.min_lc_squish,
    )

    mesh_data = read_from_msh(args.output_msh, MPI.COMM_WORLD, gdim=3)
    domain = mesh_data.mesh
    coords = domain.geometry.x
    x_offset, y_offset = np.load(args.coord_offsets_output)

    print("Computing land cover classes on final mesh")
    land_cover = sample_nlcd_at_nodes(
        args.land_cover, coords, x_offset, y_offset
    )
    diffusivity = land_cover_to_diffusivity(land_cover, spatial_parameters)

    np.save(args.diffusivity_output, diffusivity)
    np.save(args.classes_output, land_cover.astype(np.int32))

    print("Done!")


if __name__ == "__main__":
    main()
