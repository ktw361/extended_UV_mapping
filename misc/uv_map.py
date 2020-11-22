import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import open3d as o3d

from . import fragment

""" Functions of surface mapping. """


def generate_uv_map(pts,
                    method,
                    bins=256,
                    to_256=False):
    """ Generate two-channel UV Mapping

    Args:
        pts: (n, 3)
        method: either 's' or 'spherical' for spherical projection
                or 'c' or 'cylindrical' for cylindrical projection
        bins: int, u/v values will be in {0, 1, ..., bins-1}
        to_256: map from {0, 1, ..., bins-1} to [0, 256),
                possibly with holes among numbers,
                useful for visualization.

    Returns: tuple of
        u_map: (n,) int [0, bins) or [0, 256)
        v_map: (n,) int [0, bins) or [0, 256)
    """
    SPHERICAL = {'s', 'spherical'}
    CYLINDRICAL = {'c', 'cylindrical'}
    assert method in (SPHERICAL | CYLINDRICAL)
    center = np.mean(pts, 0)
    length = np.linalg.norm(pts - center, axis=1)
    unit = (pts - center) / np.expand_dims(length, 1)  # [ux, uy, uz]

    u_map = 0.5 + np.arctan2(unit[:, 2], unit[:, 0]) / (2 * np.pi)
    if method in SPHERICAL:
        v_map = 0.5 - np.arcsin(unit[:, 1]) / np.pi
    if method in CYLINDRICAL:
        v_map = unit[:, 1]

    u_map = np.clip(u_map * bins, 0, bins-1).astype(int)  # [0,1]->[0,bins]->[0,bins-1]
    v_map = np.clip(v_map * bins, 0, bins-1).astype(int)
    if to_256:
        u_map = u_map * (256/bins)
        v_map = v_map * (256/bins)

    return u_map, v_map


def draw_uv_map_image(pts,
                      pts_ind,
                      img,
                      method,
                      bins=256):
    """ Draw UV Map (2 channels) as RGB image.
    color using (U, V, 0)

    Args:
        pts: (n, 3)
        pts_ind: (n, 2) int coordinate into img
        img: (h, w, 3) ndarray
        method: see generate_uv_map()
        bins: see generate_uv_map()

    Returns: (h, w, 3) img

    """
    h, w, _ = img.shape
    u_map, v_map = generate_uv_map(pts, method, bins=bins, to_256=True)
    _img = img.copy()
    uv = np.column_stack([u_map, v_map, np.zeros_like(u_map)])
    y_ind = np.clip(pts_ind[:, 1], 0, h-1)
    x_ind = np.clip(pts_ind[:, 0], 0, w-1)
    _img[y_ind, x_ind, :] = uv
    return _img


def generate_uv_frag_map(pts,
                         method,
                         uv_bins=256,
                         frag_bins=8,
                         to_256=False):
    """ Generate extended three-channel uv map with
        two channels for UV map + one channel fragment id

    Args:
        pts: (n,3)
        method: see generate_uv_map()
        uv_bins: see generate_uv_map()
        frag_bins: int, number of fragments
        to_256: map all three channels to [0, 256),
                possibly have holes among numbers.

    Returns: uvf_map (n, 3) int,
            uvf_map[:, 0:2] from 0 to uv_bins-1
            uvf_map[:, -1]  from 0 to frag_bins-1
    """
    u_map, v_map = generate_uv_map(pts, method=method, bins=uv_bins, to_256=to_256)
    _, vtx_frag_ids = fragment.fragmentation_fps(pts, num_frags=frag_bins)
    uvf = np.column_stack([u_map, v_map, vtx_frag_ids])
    if to_256:
        uvf[:, -1] = uvf[:, -1] * (256/(frag_bins-1))

    return uvf


def create_visualization_uvf_point_cloud(pts,
                                         method,
                                         uv_bins=256,
                                         frag_bins=8):
    """ Create a Open3D PointCloud with uvf map,
    as well as fragment centers colored with pink.

    Args:
        pts:
        method:
        uv_bins:
        frag_bins:

    Returns: A Open3D's PointCloud

    """
    u, v = generate_uv_map(pts, method, bins=uv_bins, to_256=True)
    frag_center, vtx_frag_ids = fragment.fragmentation_fps(pts, num_frags=frag_bins)
    vtx_frag_ids = vtx_frag_ids * (256/(frag_bins-1))
    uvf = np.column_stack([u, v, vtx_frag_ids])
    frag_clr = np.ones_like(uvf) * [255, 0, 255]  # Pink

    pts_show = np.concatenate([pts, frag_center], axis=0)
    clr_show = np.concatenate([uvf, frag_clr])

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(pts_show)
    pcd.colors = o3d.Vector3dVector(clr_show / 255)
    return pcd


def generate_nocs_map(pts,
                      cuboid,
                      verify_point_cloud=False):
    """ Generate three-channel NOCS Map,
    Normalize so that diagonal of tight bounding box has length of 1.

    Note setting `verify_point_cloud` to be True will make function SLOW!
    We can't use KDTree (O(NlgN)) for furthest points,
    hence we have to enumerate all pairs (O(N^2)).

    Args:
        pts: (n, 3)
        cuboid: (8, 3)
        verify_point_cloud: bool, verify that the max distance between
                            `pts` is smaller than largest diagonal of `cuboid`


    Returns: (n, 3) float, NOCS Map in [0, 1]

    """
    cuboid_dist_mat = euclidean_distances(cuboid, cuboid)
    if verify_point_cloud:
        # To avoid OOM, randomly samples some points
        n = pts.shape[0]
        ratio = 0.1
        rnd_ind_1 = np.random.choice(n, int(ratio * n))
        rnd_ind_2 = np.random.choice(n, int(ratio * n))
        pts_dist_mat = euclidean_distances(pts[rnd_ind_1, :], pts[rnd_ind_2, :])
        assert pts_dist_mat.max() < cuboid_dist_mat.max()
    i, j = np.unravel_index(cuboid_dist_mat.argmax(), cuboid_dist_mat.shape)
    diag_norm = np.linalg.norm(cuboid[i, :] - cuboid[j, :])

    # Move to 1st quadrant
    nocs = (pts - pts.min(0)) / diag_norm
    return nocs
