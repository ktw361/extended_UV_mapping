import os
import os.path as osp
from types import SimpleNamespace

import numpy as np
import json
import mmcv


class Side(object):
    """
    A Side is a set of images of 'depth' 'image' 'seg'
    and object annotations.
    """
    def __init__(self, base, ind, which):
        num = '%06d' % ind
        num = '.'.join([num, which])
        self.num = num

        num = osp.join(base, num)
        depth = '.'.join([num, 'depth.png'])
        img = '.'.join([num, 'jpg'])
        json_file = '.'.join([num, 'json'])
        seg = '.'.join([num, 'seg.png'])

        self.depth = mmcv.imread(depth)
        self.img = mmcv.imread(img)
        self.seg = mmcv.imread(seg)
        with open(json_file) as fp:
            self.json = json.load(
                fp, object_hook=lambda d: SimpleNamespace(**d))
        with open(json_file) as fp:
            self.json_obj = json.load(fp)

    def __repr__(self):
        l2 = '\tdepth: %s' % str(self.depth.shape)
        l3 = '\tseg: %s' % str(self.seg.shape)
        r = str(self.num) + ':\n' + '\t' + str(self.img.shape) + '\n' + l2 + '\n' + l3
        return r


class Scene(object):
    """ Stores both left & right Sides.
    """
    def __init__(self, base, ind):
        self.left = Side(base, ind, 'left')
        self.right = Side(base, ind, 'right')

    def __repr__(self):
        return '{\n' + repr(self.left) + '\n\n' + repr(self.right) + '\n}'

