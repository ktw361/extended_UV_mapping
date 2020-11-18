""" Utility function for coordinate system. """
import numpy as np

""" Misc """


def to_homo_xn(pts):
    """ assume [x, n], output [x+1, n]"""
    n = pts.shape[1]
    return np.vstack((pts, np.ones([1, n])))


def to_homo_nx(pts):
    """ [n,x] -> [n,x+1] """
    return to_homo_xn(pts.T).T


def from_home_xn(pts):
    """ [x+1, n] -> [x, n] """
    return pts[:-1, :]


def from_home_nx(pts):
    """ [n, x+1] -> [n, x] """
    return from_home_xn(pts.T).T


def transform_nx3(transform_matrix, x):
    """

    Args:
        transform_matrix: (4, 4)
        x: (n, 3)

    Returns: (n, 3)

    """
    return transform_3xn(transform_matrix, x.T).T


def transform_3xn(transform_matrix, x):
    """

    Args:
        transform_matrix: (4, 4)
        x: (3, n)

    Returns: (3, n)

    """
    x2 = transform_matrix @ to_homo_xn(x)
    return from_home_xn(x2)


def concat_rot_transl_3x4(rot, transl):
    """
    Args:
        rot: (3, 3)
        transl: (3, 1) or (3, )

    Returns: (3, 4)

    """
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = rot
    Rt[0:3, -1] = transl.squeeze()
    return Rt


def concat_rot_transl_4x4(rot, transl):
    """
    Args:
        rot: (3, 3)
        transl: (3, 1) or (3, )

    Returns: (4, 4)

    """
    Rt = np.zeros([4, 4])
    Rt[0:3, 0:3] = rot
    Rt[0:3, -1] = transl.squeeze()
    Rt[-1, -1] = 1.0
    return Rt


def rotation_epfl(alpha, beta, gamma):
    R = np.zeros([3, 3])
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    R[0, 0] = cos_a * cos_g - cos_b * sin_a * sin_g
    R[1, 0] = cos_g * sin_a + cos_a * cos_b * sin_g
    R[2, 0] = sin_b * sin_g

    R[0, 1] = -cos_b * cos_g * sin_a - cos_a * sin_g
    R[1, 1] = cos_a * cos_b * cos_g - sin_a * sin_g
    R[2, 1] = cos_g * sin_b

    R[0, 2] = sin_a * sin_b
    R[1, 2] = -cos_a * sin_b
    R[2, 2] = cos_b

    return R


# def rotation_xyz_from_euler(x_rot, y_rot, z_rot):
#     rz = np.float32([
#         [np.cos(z_rot), np.sin(z_rot), 0],
#         [-np.sin(z_rot), np.cos(z_rot), 0],
#         [0, 0, 1]
#     ])
#     ry = np.float32([
#         [np.cos(y_rot), 0, -np.sin(y_rot)],
#         [0, 1, 0],
#         [np.sin(y_rot), 0, np.cos(y_rot)],
#     ])
#     rx = np.float32([
#         [1, 0, 0],
#         [0, np.cos(x_rot), np.sin(x_rot)],
#         [0, -np.sin(x_rot), np.cos(x_rot)],
#     ])
#     return rz @ ry @ rx


""" Camera """


def project_3d_2d(A, Xcam):
    """

    :param A: [3, 3]
    :param Xcam:  [3, n]
    :return:  [2, n]
    """
    Ximg_h = A @ Xcam
    return extract_pixel_homo_xn(Ximg_h)


def extract_pixel_homo_xn(x2d_h):
    """ x2d_h: [3, n] -> [2, n] """
    x2d = x2d_h / x2d_h[-1, :]
    return from_home_xn(x2d)


def extract_pixel_homo_nx(x2d_h):
    """ x2d_h: [n, 3] -> [n, 2] """
    return extract_pixel_homo_xn(x2d_h.T).T


