from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI
import numpy as np
import rasterio
from rasterio.transform import rowcol

import mmgpy
import meshio
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter

def compute_metric(verts, faces, hmax, isotropy, min_lc_squish, lc_grad=None, lc_diffusivity=None, lc_neighbor_max=None, lc_neighbor_min=None):
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

            activation = (1+np.exp(50*(.96-1)))/(1+np.exp(50*(.96-abs(cos_theta))))

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

def dem_to_msh(input_tif, output_msh, downsample=3, smooth=True, sigma=2, hmax=400, isotropy=.1, min_lc_squish=.1):

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
    np.save("coord_offsets.npy", np.array([x_offset, y_offset]))
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
    meshio.write("terrain_input.vtk", tmp_mesh)

    # Sample land cover at mesh vertices
    print("Sampling land cover at mesh vertices...")
    x_offset, y_offset = np.load("coord_offsets.npy")
    land_cover = sample_nlcd_at_nodes("nlcd_aligned.tif", verts, x_offset, y_offset)
    lc_diffusivity = land_cover_to_diffusivity(land_cover)
    lc_grad = compute_lc_gradient(verts, faces, lc_diffusivity)
    lc_neighbor_min, lc_neighbor_max = compute_lc_neighbor_stats(verts, faces, lc_diffusivity)

    #Compute metric
    print("Computing anisotropic metric...")
    metrics = compute_metric(verts, faces, hmax, isotropy, min_lc_squish, lc_grad=lc_grad, lc_diffusivity=lc_diffusivity, lc_neighbor_max=lc_neighbor_max, lc_neighbor_min=lc_neighbor_min)

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
    mesh = mmgpy.read("terrain_input.vtk")

    mesh["metric"] = metrics  # shape (n_verts, 6)
    hmin = isotropy*min_lc_squish*hmax
    mesh.remesh(hmin=hmin, hmax=hmax, hausd=hmax*0.1, hgrad=1.3)
    mesh.save("terrain_remeshed.mesh")
    

    # --- Add physical groups via Gmsh and write final .msh ---
    print("Adding physical groups and writing .msh...")
    import gmsh
    gmsh.initialize()
    gmsh.open("terrain_remeshed.mesh")

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

def land_cover_to_diffusivity(land_cover_array):
    lc_map = {
        0:  1.0,   # unknown/nodata -- don't block
        11: 0.0001,   # Open water -- barrier
        12: 0.3,   # Perennial ice/snow
        21: 0.7,   # Developed open space
        22: 0.4,   # Developed low intensity
        23: 0.2,   # Developed medium intensity
        24: 0.05,  # Developed high intensity
        31: 0.9,   # Barren rock
        41: 0.5,   # Deciduous forest
        42: 0.4,   # Evergreen forest
        43: 0.45,  # Mixed forest
        52: 0.8,   # Shrub/scrub
        71: 0.95,  # Grassland/herbaceous
        81: 0.85,  # Pasture/hay
        82: 0.8,   # Cultivated crops
        90: 0.2,   # Woody wetlands
        95: 0.3,   # Emergent herbaceous wetlands
    }
    result = np.ones(len(land_cover_array))
    for code, value in lc_map.items():
        result[land_cover_array == code] = value
    return result

#Generate anisotropic mesh
dem_to_msh(
    input_tif  = "region3_cropped2.tif",
    output_msh = "terrain6.msh",
    smooth = True,
    sigma = 3,
    downsample = 3,
    hmax = 150,
    isotropy = .02,
    min_lc_squish=.1,
)


# Load mesh for terrain diffusion data
mesh_data = read_from_msh("terrain6.msh", MPI.COMM_WORLD, gdim=3)
domain = mesh_data.mesh
coords = domain.geometry.x
x_offset, y_offset = np.load("coord_offsets.npy")

# Sample and convert
print(f"Computing land cover classes on mesh")
land_cover = sample_nlcd_at_nodes("nlcd_aligned.tif", coords, x_offset, y_offset)
diffusivity = land_cover_to_diffusivity(land_cover)

np.save("land_cover_diffusivity.npy", diffusivity)
np.save("land_cover_classes.npy", land_cover.astype(np.float64))

print(f"Water pixels (barrier): {(diffusivity == 0.0001).sum()}")
print(f"Done!")