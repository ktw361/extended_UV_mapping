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


def _parse_ply_header(f):
    """ @author Pavel Rojtberg <https://github.com/paroj/linemode_dadtaset> """
    assert f.readline().strip() == "ply"
    vtx_count = 0
    idx_count = 0

    for l in f:
        if l.strip() == "end_header":
            break
        elif l.startswith("element vertex"):
            vtx_count = int(l.split()[-1])
        elif l.startswith("element face"):
            idx_count = int(l.split()[-1])

    return vtx_count, idx_count


def ply_vtx(path):
    """
    @author Pavel Rojtberg <https://github.com/paroj/linemode_dadtaset>

    read all vertices from a ply file
    """
    f = open(path)
    vtx_count = _parse_ply_header(f)[0]

    pts = []

    for _ in range(vtx_count):
        pts.append(f.readline().split()[:3])

    return np.array(pts, dtype=np.float32)
