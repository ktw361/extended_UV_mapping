import numpy as np


def load_obj(fname):
    """

    Args:
        fname: e.g. '037_scissor/textured_simple.obj'

    Returns: ndarray (n, 3)

    """
    with open(fname, 'r') as fp:
        lines = fp.readlines()
        # char = None
        # while char != 'v':
        #     char = fp.readline().split(' ')[0]
        lines = [l for l in lines if l.split(' ')[0] == 'v']
        # v x y z r g b
        lines = np.float32([
            list(map(float, l.split(' ')[1:4])) for l in lines
        ])
    return lines
