import cv2
from . import coor_utils


def transformation_4x4_EPnP(pts_3d,
                            pts_2d,
                            M_intrinsic):
    """ Helper function to get 4x4 transformation matrix
    using solvePnP() of cv2.

    Args:
        pts_3d: (n, 3) ndarray
        pts_2d: (n, 2) ndarray, must be corresponding order in pts_3d
        M_intrinsic: (3, 3) intrinsic matrix

    Returns: (4, 4) ndarray of [R|t]

    """
    _, rot_exp, t = cv2.solvePnP(
        pts_3d, pts_2d, cameraMatrix=M_intrinsic, distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP)
    rot, _ = cv2.Rodrigues(rot_exp)

    return coor_utils.concat_rot_transl_4x4(rot, t)


def transformation_4x4_P3PRansac(pts_3d,
                                 pts_2d,
                                 M_intrinsic,
                                 iterationsCount=150,
                                 reprojectionError=1.0,
                                 return_inliers=False,
                                 verbose=False):
    """ Helper function to get 4x4 transformation matrix
    using P3P & Ransac of cv2.

    Args:
        pts_3d: (n, 3) ndarray
        pts_2d: (n, 2) ndarray, must be corresponding order in pts_3d
        M_intrinsic: (3, 3) intrinsic matrix
        verbose: bool

    Returns: (4, 4) ndarray of [R|t]

    """
    _, rvec, tvecs, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, M_intrinsic, distCoeffs=None,
        iterationsCount=iterationsCount, reprojectionError=reprojectionError,
        flags=cv2.SOLVEPNP_P3P)
    if verbose:
        print(f"{len(inliers)} inliers out of {len(pts_3d).shape} points")
    rot, _ = cv2.Rodrigues(rvec, jacobian=None)
    transformation = coor_utils.concat_rot_transl_4x4(rot, tvecs)
    if return_inliers:
        return transformation, inliers
    else:
        return transformation
