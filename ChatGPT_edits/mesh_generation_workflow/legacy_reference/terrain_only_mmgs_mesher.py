import mmgpy
import meshio
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter

from activation import activation

def compute_metric(verts, faces, hmin, hmax):
    eps = 1e-6
    normals = np.zeros((len(verts), 3))
    grad_z = np.zeros((len(verts), 3))
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
            M = (np.eye(3) - np.outer(normal,normal)) * (1/hmax**2) + np.outer(normal,normal) * (1/hmin**2)

        else:
            es = es / es_norm
            ec = np.cross(es,normal)

            activation = (1+np.exp(50*(.96-1)))/(1+np.exp(50*(.96-abs(cos_theta))))

            h_s = hmin + (hmax - hmin) * activation
            h_c = hmax
            h_n = hmin

            M = (1/h_s**2) * np.outer(es, es) \
                + (1/h_c**2) * np.outer(ec, ec) \
                + (1/h_n**2) * np.outer(normal, normal)

            # Upper triangle: m11, m12, m13, m22, m23, m33
        metrics[i] = [M[0,0], M[0,1], M[0,2],
                            M[1,1], M[1,2],
                                    M[2,2]]
    return metrics

def dem_to_msh(input_tif, output_msh, downsample=3, hmax=500, smooth=True, sigma=2, isotropy=.1):

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
    X -= x_offset
    Y -= y_offset
    Z -= Z.mean()
    np.save("coord_offsets.npy", np.array([x_offset, y_offset]))

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

    print("Computing anisotropic metric...")
    hmin = isotropy*hmax
    metrics = compute_metric(verts, faces, hmin, hmax)

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


