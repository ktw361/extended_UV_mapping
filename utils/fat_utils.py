import os.path as osp
import json
from types import SimpleNamespace
import numpy as np
import warnings

from scipy.spatial.transform import Rotation as R

from utils import coor_utils, visualize
from utils import data as datalib
import mmcv


class Side(object):
    # TODO: add `mixed` support
    """
    A Side is a set of images of 'depth' 'image' 'seg'
    and object annotations.
    """
    def __init__(self, base, ind, which, parse_json=False):
        """

        Args:
            base:
            ind:
            which: 'left' or 'right'
            parse_json: generate json namespace, used by jupyter
        """
        num = '%06d' % ind
        num = '.'.join([num, which])
        self.num = num
        self.which = which

        num = osp.join(base, num)
        depth = '.'.join([num, 'depth.png'])
        img = '.'.join([num, 'jpg'])
        json_file = '.'.join([num, 'json'])
        seg = '.'.join([num, 'seg.png'])

        self.depth = mmcv.imread(depth)
        self.img = mmcv.imread(img)
        self.seg = mmcv.imread(seg)

        self.json = None
        if parse_json:
            with open(json_file) as fp:
                self.json = json.load(
                    fp, object_hook=lambda d: SimpleNamespace(**d))
        with open(json_file) as fp:
            self.json_obj = json.load(fp)

        with open(osp.join(base, '_camera_settings.json')) as fp:
            self.camera_settings = json.load(fp)
        with open(osp.join(base, '_object_settings.json')) as fp:
            self.object_settings = json.load(fp)

        cam_params = self.camera_settings['camera_settings']
        if cam_params[0]['name'] == self.which:
            cam_params = cam_params[0]
        else:
            cam_params = cam_params[1]

        # Set intrinsic_matrix and projection_matrix
        self._setup_intrinsic_and_projection(cam_params)

    def __repr__(self):
        l2 = '\tdepth: %s' % str(self.depth.shape)
        l3 = '\tseg: %s' % str(self.seg.shape)
        r = str(self.num) + ':\n' + '\t' + str(self.img.shape) + \
            '\n' + l2 + '\n' + l3
        return r

    def _setup_intrinsic_and_projection(self, cam_params):
        self.cap_width = cam_params['captured_image_size']['width']
        self.cap_height = cam_params['captured_image_size']['height']

        intrinsics = cam_params['intrinsic_settings']
        self.fx, self.fy = intrinsics['fx'], intrinsics['fy']
        self.cx, self.cy = intrinsics['cx'], intrinsics['cy']
        s = intrinsics['s']
        # self._intrinsic_matrix = np.float32([
        #     [self.fx, s, self.cx],
        #     [0, self.fy, self.cy],
        #     [0,       0,       1],
        # ])  # (3, 3)

        # Calculate projection_matrix
        zfar = 1.0
        znear = 1.0
        zdiff = float(zfar - znear)
        a = (2.0 * self.fx) / float(self.cap_width)
        b = (2.0 * self.fy) / float(self.cap_height)
        c = -znear / zdiff if (zdiff > 0) else 0
        d = (znear * zfar) / zdiff if (zdiff > 0) else (-znear)
        c1 = 1.0 - (2.0 * self.cx) / self.cap_width
        c2 = (2.0 * self.cy) / self.cap_height - 1.0

        self._projection_matrix = np.float32([
            [a, 0, 0, 0],
            [0, b, 0, 0],
            [c1, c2, c, d],
            [0, 0, -1.0, 0]
        ])

        # The followings are used in transform_point_clouds()
        self._M_intrinsic = np.float32([
            [self.fx, 0, 0, 0],
            [0, self.fy, 0, 0],
            [0, 0, 1, 0],
        ])  # 3x4
        self._M_offset = np.float32([
            [1, 0, self.cap_width/2],
            [0, 1, self.cap_height/2],
            [0, 0, 1],
        ])

    @property
    def intrinsic_matrix(self):
        """
        Is:
            [fx 0  0]
            [0  fy 0]
            [0  0  1]

        Not:
            [fx 0 cx]
            [0 fy cy]
            [0  0  1]
        """
        return self._M_intrinsic

    @property
    def projection_matrix(self):
        return self._projection_matrix

    @property
    def offset_matrix(self):
        """ Equivalent to original self._M_ndc2win_then_flip """
        return self._M_offset

    def transform_point_cloud_with_transl_and_quat(self,
                                                   pts,
                                                   quat,
                                                   transl,
                                                   store_dict=None):
        """ Mimic OpenGL's 3D to 2D mapping using supplied translation and quaternion
        P_2d = flip() @ NDC2WIN() @ perspective_division
                 @ projection_matrix @ Tmw @ P_3d

            where Tmw is directly computed from (quat, transl)

        Args:
            pts: (n, 3) of (x, y, z)
            quat: (4, )
            transl: (3, )
            store_dict: None or a dict,
                if a dict is passed in, and has key 'depth_sorted_ind',
                The function will sort pts_2d by abs(depth), far away (from screen) points
                comes first, while closer points comes latter,
                which can overwrite far away points during numpy assignment.
                This is useful for displaying 2D points in correct order.
                store_dict['depth_sorted_ind'] will be overwritten to be sorted indexes.

        Returns: (n, 2) of [x, y]
        """
        DEPTH_SORTED_IND = 'depth_sorted_ind'

        rot = R.from_quat(quat).as_matrix()
        Tmw = coor_utils.concat_rot_transl_4x4(rot, transl)  # T_model->world, confront OpenGL's

        pts_h = coor_utils.to_homo_nx(pts).T  # pts_h: (4, n)
        M_transformations = self._M_intrinsic @ Tmw
        pts_h = M_transformations @ pts_h
        if isinstance(store_dict, dict) and DEPTH_SORTED_IND in store_dict:
            sorted_ind = np.argsort(abs(pts_h[-1, :]))[::-1]
            store_dict[DEPTH_SORTED_IND] = sorted_ind
            pts_h = pts_h[:, sorted_ind]
        pts_h = coor_utils.normalize_homo_xn(pts_h)
        pts_h = self._M_offset @ pts_h

        pts_2d = coor_utils.from_home_xn(pts_h).transpose()  # (3,n) -> (n,2)

        return pts_2d

    def transform_point_cloud(self, pts, ind, store_dict=None):
        """

        Args:
            pts: (n, 3)
            ind: index into self.objects
            store_dict: None or dict, see transform_point_cloud_with_transl_and_quat()

        Returns: (n, 2)
        """
        quat = self.get_params_from_key('quaternion_xyzw')[ind]
        transl = self.get_params_from_key('location')[ind]
        return self.transform_point_cloud_with_transl_and_quat(
            pts, quat, transl, store_dict=store_dict)

    @property
    def objects(self):
        """

        Returns: list of dict

        """
        return self.json_obj['objects']

    @property
    def class_list(self):
        """ Replace the _16k/_16K suffixes and returns the list

        Returns: list of str

        """
        return [v.replace('_16k', '').replace('_16K', '')
                for v in self.get_params_from_key('class')]

    @property
    def visibility_list(self):
        """

        Returns: list of float

        """
        return np.float32(self.get_params_from_key('visibility'))

    @property
    def Tmw_list(self):
        """ Rigid Transformation from model to world,
        the dataset generate the quaternions and translations
        using OpenCV's convention.

        Returns: list of 4x4 [R|t]

        """
        quat_list = self.get_params_from_key('quaternion_xyzw')
        transls_list = self.get_params_from_key('location')
        rot_list = [R.from_quat(q).as_matrix() for q in quat_list]
        Tmw_list = [coor_utils.concat_rot_transl_4x4(r, t)
                    for (r, t) in zip(rot_list, transls_list)]
        return Tmw_list

    def get_params_from_key(self, key):
        """

        Args:
            key: str

        Returns: list

        """
        ret = []
        for object in self.objects:
            ret.append(self._parse_with_key(object, key))
        return ret

    @staticmethod
    def _parse_with_key(d, key):
        if key in [
            'pose_transform_permuted',
            'location',
            'quaternion_xyzw',
            'cuboid_centroid',
            'projected_cuboid_centroid',
            'cuboid',
            'projected_cuboid',
        ]:
            return np.float32(d[key])
        else:
            return d[key]

    @property
    def camera_transformation(self):
        """

        Returns: T_camera->world, (4, 4)

        """
        warnings.warn("This property is used in NVDU.")
        loc = np.float32(
            self.json_obj['camera_data']['location_worldframe'])
        quat = np.float32(
            self.json_obj['camera_data']['quaternion_xyzw_worldframe'])
        rot = R.from_quat(quat).as_matrix()
        Tcw = coor_utils.concat_rot_transl_4x4(rot, loc)
        return Tcw

    @property
    def fixed_model_transform_list(self):
        """ If using `ycb_models_nvdu_aligned_cm`, then this one is not used. """
        ret = []
        for exp_obj in self.object_settings['exported_objects']:
            ret.append(np.float32(exp_obj['fixed_model_transform']))
        return ret

    def visualize_object(self,
                         ind,
                         model_root,
                         show_pivots=True,
                         show_cuboids=True,
                         show_projected_pcd=True,
                         color_array=(255, 0, 255)):
        """

        Args:
            ind: index into self.objects
            model_root: /<model_root>/<model_name>/google_512k/textured.obj'
                        example <model_name> like '037_scissors'
            show_pivots: bool, show transformed pivots axis
            show_cuboids: bool, show projected cuboids
            show_projected_pcd: bool, draw colored projected point cloud
                                onto the object
            color_array: tuple, default pink

        Returns: annotated ndarray (h, w, 3)

        """
        width, height, depth = self.object_settings['exported_objects'][ind]['cuboid_dimensions']
        pivots = np.float32([
            [0, 0, 0],       # center
            [height, 0, 0],  # X-axis
            [0, width, 0],   # Y-axis
            [0, 0, depth]
        ])

        # Get cuboids in 3D
        cx, cy, cz = 0, 0, 0
        right = cx + width / 2.0  # X axis point to the right
        _left = cx - width / 2.0
        top = cy + height / 2.0  # Y axis point upward
        bottom = cy + height / 2.0
        front = cz - depth / 2.0  # Z axis point inward
        rear = cz + depth / 2.0
        # List of 8 vertices of the box
        cuboid = np.float32([
            [_left, top, front],  # Front Top Left
            [right, top, front],  # Front Top Right
            [right, bottom, front],  # Front Bottom Right
            [_left, bottom, front],  # Front Bottom Left
            [_left, top, rear],  # Rear Top Left
            [right, top, rear],  # Rear Top Right
            [right, bottom, rear],  # Rear Bottom Right
            [_left, bottom, rear],  # Rear Bottom Left
        ])

        # Get model point cloud
        # The '006_mustart_bottle' is UPPER CASE 16K, others' are 16k?
        model_name = self.objects[0]['class'].replace('_16k', '').replace('_16K', '')
        model_pcd = datalib.load_obj(
            osp.join(model_root, model_name, 'google_512k/textured.obj'))

        # Visualize
        _img = self.img.copy()
        store_dict = dict(depth_sorted_ind=None)
        if show_pivots:
            pivots_2d = self.transform_point_cloud(pivots, ind, store_dict=store_dict)
            visualize.draw_pivots_image(_img, pivots_2d)
        if show_cuboids:
            cuboid_2d = self.transform_point_cloud(cuboid, ind, store_dict=store_dict)
            visualize.draw_proj_cuboid_image(_img, cuboid_2d)
        if show_projected_pcd:
            pcd_2d = self.transform_point_cloud(model_pcd, ind, store_dict=store_dict)
            pcd_ind = np.floor(pcd_2d).astype(int)
            y_ind = np.clip(pcd_ind[:, 1], 0, self.cap_height-1)
            x_ind = np.clip(pcd_ind[:, 0], 0, self.cap_width-1)
            _img[y_ind, x_ind, :] = color_array

        return _img


class Scene(object):
    """ Stores both left & right Sides.
    """
    def __init__(self, base, ind, parse_json=False):
        self.left = Side(base, ind, 'left', parse_json=parse_json)
        self.right = Side(base, ind, 'right', parse_json=parse_json)

    def __repr__(self):
        return '{\n' + repr(self.left) + '\n\n' + \
               repr(self.right) + '\n}'
