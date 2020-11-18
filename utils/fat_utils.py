import os.path as osp
import json
from types import SimpleNamespace
import numpy as np

from scipy.spatial.transform import Rotation as R

from utils import coor_utils
import mmcv


class Side(object):
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
        # In NVDU(OpenGL), only the projection matrix is used.
        self._setup_intrinsic_and_projection(cam_params)

    def __repr__(self):
        l2 = '\tdepth: %s' % str(self.depth.shape)
        l3 = '\tseg: %s' % str(self.seg.shape)
        r = str(self.num) + ':\n' + '\t' + str(self.img.shape) + \
            '\n' + l2 + '\n' + l3
        return r

    def _setup_intrinsic_and_projection(self, cam_params):
        """
        In NVDU(OpenGL), only the projection matrix is used.
        """

        self.cap_width = cam_params['captured_image_size']['width']
        self.cap_height = cam_params['captured_image_size']['height']

        intrinsics = cam_params['intrinsic_settings']
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        s = intrinsics['s']
        self.intrinsic_matrix = np.float32([
            [fx, s, cx],
            [0, fy, cy],
            [0,  0,  1],
        ])  # (3, 3)

        # Calculate projection_matrix
        zfar = 1.0
        znear = 1.0
        zdiff = float(zfar - znear)
        a = (2.0 * fx) / float(self.cap_width)
        b = (2.0 * fy) / float(self.cap_height)
        c = -znear / zdiff if (zdiff > 0) else 0
        d = (znear * zfar) / zdiff if (zdiff > 0) else (-znear)
        c1 = 1.0 - (2.0 * cx) / self.cap_width
        c2 = (2.0 * cy) / self.cap_height - 1.0

        self.projection_matrix = np.float32([
            [a, 0, 0, 0],
            [0, b, 0, 0],
            [c1, c2, c, d],
            [0, 0, -1.0, 0]
        ])

    @property
    def objects(self):
        """

        Returns: list of dict

        """
        return self.json_obj['objects']

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
        loc = np.float32(
            self.json_obj['camera_data']['location_worldframe'])
        quat = np.float32(
            self.json_obj['camera_data']['quaternion_xyzw_worldframe'])
        rot = R.from_quat(quat).as_matrix()
        Tcw = coor_utils.concat_rot_transl_4x4(rot, loc)
        return Tcw

    @property
    def fixed_model_transform_list(self):
        ret = []
        for exp_obj in self.object_settings['exported_objects']:
            ret.append(np.float32(exp_obj['fixed_model_transform']))
        return ret


class Scene(object):
    """ Stores both left & right Sides.
    """
    def __init__(self, base, ind, parse_json=False):
        self.left = Side(base, ind, 'left', parse_json=parse_json)
        self.right = Side(base, ind, 'right', parse_json=parse_json)

    def __repr__(self):
        return '{\n' + repr(self.left) + '\n\n' + \
               repr(self.right) + '\n}'
