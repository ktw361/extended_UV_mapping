import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import open3d as o3d

from . import fragment
from utils import coor_utils, cv2_utils, visualize
from utils.class_names import fat_name2segid
from utils import data as datalib

""" Surface mapping utilities. """


class SurfMapping:
    """
    Forward phase: 3D => 2D
    Backward phase: 2D => 3D properties
    """
    def __init__(self):
        # Backward configs
        self.M_intrinsic = None
        self.M_offset = None
        self.side = None
        self.pnp_method = 'ransac'
        self.segid = 255

        self.map_to_xyz = None
        self.params = dict()

    def _compute_map_to_xyz(self, pts, **kwags):
        raise NotImplementedError

    def generate_map(self, pts, **kwargs):
        raise NotImplementedError

    def update_params(self, key, value):
        self.params[key] = value

    def update_backward_config(self,
                               M_intrinsic,
                               M_offset,
                               pnp_method):
        self.M_intrinsic = M_intrinsic
        self.M_offset = M_offset
        self.pnp_method = pnp_method

    def update_backward_cfg_from_side(self,
                                      side,
                                      segid=255,
                                      pnp_method='epnp'):
        """

        Args:
            side: Side object
            segid: int
            pnp_method: str
        """
        self.side = side
        self.segid = segid
        self.M_intrinsic = side.intrinsic_matrix
        self.M_offset = side.offset_matrix
        self.pnp_method = pnp_method
        self.params['mask'] = side.seg

    """ Backward functions. """

    def estimate_Tmw_from_map(self,
                              img,
                              mask,
                              segid,
                              map_to_xyz):
        """

        Args:
            img: (h, w, 3) with [u, v, 0] at masked region
            mask: (h, w, 3)
            segid: int, indicate masked region
            map_to_xyz: dict
            verbose: bool

        Returns: (4, 4) ndarray
        """
        verbose = self.params.get('verbose', True)
        M_intrinsic = self.M_intrinsic[:3, :3]
        M_offset = self.M_offset
        key_len = len(list(map_to_xyz.keys())[0])  # e.g. len((2,3)) = :1

        mask_y, mask_x, _ = np.where(mask == segid)
        n = len(mask_y)
        pts_2d, pts_3d = [], []
        for i in range(n):
            _i, _j = mask_y[i], mask_x[i]
            key = tuple(img[_i, _j, :key_len])
            if key in map_to_xyz:
                pts_3d.append(map_to_xyz[key])
                pts_2d.append([mask_x[i], mask_y[i]])

        pts_3d = np.row_stack(pts_3d)
        pts_2d = np.row_stack(pts_2d)
        pts_2d = coor_utils.reverse_offset(pts_2d, M_offset).astype(np.float32)
        if verbose:
            print("number of valid points: ", len(pts_2d))

        if self.pnp_method == 'epnp':
            Tmw = cv2_utils.transformation_4x4_EPnP(
                pts_3d, pts_2d, M_intrinsic)
        else:
            Tmw = cv2_utils.transformation_4x4_P3PRansac(
                pts_3d, pts_2d, M_intrinsic)
        return Tmw

    def fast_estimate_Tmw(self, img):
        assert self.side is not None
        assert self.map_to_xyz is not None
        if 'mask' not in self.params:
            self.params['mask'] = self.side.seg

        Tmw = self.estimate_Tmw_from_map(
            img, mask=self.params['mask'], segid=self.segid,
            map_to_xyz=self.map_to_xyz)
        return Tmw

    def fast_estimate_Tmw_and_compare(self,
                                      img,
                                      ind=0,
                                      color_gt='blue',
                                      color_est='red'):
        verbose = self.params.get('verbose', True)
        side = self.side
        Tmw_est = self.fast_estimate_Tmw(img)
        Tmw_true = side.Tmw_list[0]

        cuboid = side.canonical_cuboids[ind]
        proj_cuboid_est = coor_utils.world_to_camera_pipeline(
            cuboid, self.M_intrinsic, self.M_offset, Tmw=Tmw_est)
        proj_cuboid_true = side.get_params_from_key('projected_cuboid')[ind]
        _img = np.asarray(side.img).copy()
        _img = visualize.draw_proj_cuboid_image(
            _img, proj_cuboid_true, color=color_gt, thickness=12)
        _img = visualize.draw_proj_cuboid_image(
            _img, proj_cuboid_est, color=color_est, thickness=4)
        if verbose:
            print("Difference of Tmw_estimated and Tmw_true:\n",
                  Tmw_est - Tmw_true)
            print("L2 norm of difference: ",
                  np.linalg.norm(Tmw_est - Tmw_true, ord=1))
        return _img


class UVMapping(SurfMapping):
    """
    Forward config : {
        method: either 's' or 'spherical' for spherical projection
                or 'c' or 'cylindrical' for cylindrical projection
        bins: int, u/v values will be in {0, 1, ..., bins-1}
        to_256: map from {0, 1, ..., bins-1} to [0, 256),
                possibly with holes among numbers,
                useful for visualization.
    }

    Backward config : {

    }

    """
    def __init__(self,
                 method='s',
                 bins=256,
                 to_256=False,
                 M_intrinsic=None,
                 M_offset=None,
                 pnp_method=None):
        super(UVMapping, self).__init__()
        # Forward configs
        self.method = method
        self.bins = bins
        self.to_256 = to_256

        # Backwards configs
        self.M_intrinsic = M_intrinsic
        self.M_offset = M_offset
        self.pnp_method = pnp_method

        self.side = None
        self.segid = 255

        self.params = dict()
        self.map_to_xyz = None

    def _compute_map_to_xyz(self, pts, **kwags):
        u_map, v_map = self.generate_map(pts)
        map_to_xyz = {}
        for i in range(len(u_map)):
            key = (u_map[i], v_map[i])
            map_to_xyz[key] = pts[i]
        self.map_to_xyz = map_to_xyz
        return self.map_to_xyz

    """ Forward functions. """

    def generate_map(self, pts, **kwargs):
        """ Generate two-channel UV Mapping

        Args:
            pts: (n, 3)

        Returns: tuple of
            u_map: (n,) int [0, bins) or [0, 256)
            v_map: (n,) int [0, bins) or [0, 256)
        """
        method = self.method
        to_256 = self.to_256
        bins = self.bins

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

        u_map = np.clip(u_map * bins, 0, bins-1).astype(int)  # [0,1]->[0,bins]->[0,bins)
        v_map = np.clip(v_map * bins, 0, bins-1).astype(int)
        if to_256:
            u_map = u_map * (256//bins)
            v_map = v_map * (256//bins)

        return u_map, v_map


class UVFragMapping(SurfMapping):

    def __init__(self,
                 method='s',
                 uv_bins=256,
                 frag_bins=8,
                 to_256=False,
                 M_intrinsic=None,
                 M_offset=None,
                 pnp_method=None):
        super(UVFragMapping, self).__init__()
        # Forward configs
        self.method = method
        self.uv_bins = uv_bins
        self.frag_bins = frag_bins
        self.to_256 = to_256

        # Backwards configs
        self.M_intrinsic = M_intrinsic
        self.M_offset = M_offset
        self.pnp_method = pnp_method

        self.side = None
        self.segid = 255

    def generate_map(self, pts):
        """ Generate extended three-channel uv map with
            two channels for UV map + one channel fragment id

        Args:
            pts: (n,3)

        Returns: uvf_map (n, 3) int,
                uvf_map[:, 0:2] from 0 to uv_bins-1, or 0 to 255
                uvf_map[:, -1]  from 0 to frag_bins-1, or 0 to 255
        """
        u_map, v_map = generate_uv_map(
            pts, method=self.method, bins=self.uv_bins, to_256=self.to_256)
        _, vtx_frag_ids = fragment.fragmentation_fps(pts, num_frags=self.frag_bins)
        uvf = np.column_stack([u_map, v_map, vtx_frag_ids])
        if self.to_256:
            uvf[:, -1] = uvf[:, -1] * (256//self.frag_bins)

        return uvf

    def _compute_map_to_xyz(self, pts, **kwags):
        uvf_map = self.generate_map(pts)
        map_to_xyz = {}
        for i in range(uvf_map.shape[0]):
            key = tuple(uvf_map[i])
            map_to_xyz[key] = pts[i]
        self.map_to_xyz = map_to_xyz
        return self.map_to_xyz


""" Part I. Vanilla UV Mapping. """


def generate_uv_map(pts,
                    method='s',
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
        u_map = u_map * (256//bins)
        v_map = v_map * (256//bins)

    return u_map, v_map


def draw_uv_map_image(pts,
                      pts_ind,
                      img,
                      mask_config=None,
                      method='s',
                      bins=256):
    """ Draw one object's UV Map (2 channels) as RGB image.
    color using (U, V, 0)

    Args:
        pts: (n, 3)
        pts_ind: (n, 2) int coordinate into img
        img: (h, w, 3) ndarray
        mask_config: None or tuple of (mask, segid),
            where mask is a (h, w, 3) ndarray,
            and segid is a int indicating the masked region.
            For single object, segid == 255;
            for multiple objects, segid == (class_index+1) * 12, see class_names.py
        method: see generate_uv_map()
        bins: see generate_uv_map()

    Returns: (h, w, 3) array

    """
    h, w, _ = img.shape
    u_map, v_map = generate_uv_map(pts, method=method, bins=bins, to_256=True)
    _img = img.copy()
    uv = np.column_stack([u_map, v_map, np.zeros_like(u_map)])
    y_ind = np.clip(pts_ind[:, 1], 0, h-1)
    x_ind = np.clip(pts_ind[:, 0], 0, w-1)
    _img[y_ind, x_ind, :] = uv
    if mask_config is not None and isinstance(mask_config, tuple):
        mask, segid = mask_config
        if isinstance(mask, np.ndarray):
            _img[mask != segid] = img[mask != segid]
    return _img


def draw_uv_map_side(side, model_root, method='s', bins=256):
    """ Draw UV Map onto the RGB image of the Side class.

    Args:
        side: a Side class
        model_root: str, will be used to retrieve original 3d point cloud model
            using "/<model_root>/011_banana/google_512k/textured.obj"
        method: 's' or 'c'
        bins: int, default 256

    Returns: (h, w, 3) ndarray

    """
    _img = side.img
    for i, class_name in enumerate(side.class_list):
        model_path = f"{model_root}/{class_name}/google_512k/textured.obj"
        pts = datalib.load_obj(model_path)
        segid = fat_name2segid[class_name]

        store_dict = dict(depth_sorted_ind=None)
        pts_ind = side.transform_point_cloud(
            pts, ind=i, store_dict=store_dict).astype(int)
        pts_sorted = pts[store_dict['depth_sorted_ind']]
        _img = draw_uv_map_image(pts_sorted, pts_ind, _img,
                                 mask_config=(side.seg, segid),
                                 method=method, bins=bins)
    return _img


""" Part II. UV Mapping extended with fragments. """


def generate_uv_frag_map(pts,
                         method='s',
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
        uvf[:, -1] = uvf[:, -1] * (256//frag_bins)

    return uvf


def draw_uvf_map_image(pts,
                       pts_ind,
                       img,
                       mask_config=None,
                       method='s',
                       uv_bins=256,
                       frag_bins=8):
    """ Draw one object's UVF Map (3 channels) as RGB image.
    color using (U, V, Frag_id)

    Args:
        pts: (n, 3)
        pts_ind: (n, 2) int coordinate into img
        img: (h, w, 3) ndarray
        mask_config: None or tuple of (mask, segid),
            where mask is a (h, w, 3) ndarray,
            and segid is a int indicating the masked region.
            For single object, segid == 255;
            for multiple objects, segid == (class_index+1) * 12, see class_names.py
        method: see generate_uv_map()
        uv_bins: see generate_uv_map()
        frag_bins: int

    Returns: (h, w, 3) array

    """
    h, w, _ = img.shape
    uvf = generate_uv_frag_map(
        pts, method=method, uv_bins=uv_bins, frag_bins=frag_bins, to_256=True)
    _img = img.copy()
    y_ind = np.clip(pts_ind[:, 1], 0, h-1)
    x_ind = np.clip(pts_ind[:, 0], 0, w-1)
    _img[y_ind, x_ind, :] = uvf
    if mask_config is not None and isinstance(mask_config, tuple):
        mask, segid = mask_config
        if isinstance(mask, np.ndarray):
            _img[mask != segid] = img[mask != segid]
    return _img


def create_visualization_uvf_point_cloud(pts,
                                         method='s',
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
    vtx_frag_ids = vtx_frag_ids * (256//(frag_bins-1))
    uvf = np.column_stack([u, v, vtx_frag_ids])
    frag_clr = np.ones_like(uvf) * [255, 0, 255]  # Pink

    pts_show = np.concatenate([pts, frag_center], axis=0)
    clr_show = np.concatenate([uvf, frag_clr])

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(pts_show)
    pcd.colors = o3d.Vector3dVector(clr_show / 255)
    return pcd


""" Part III, others. E.g. NOCS map. """


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
