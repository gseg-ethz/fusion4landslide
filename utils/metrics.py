
from typing import Optional
import numpy as np
import torch
from utils.o3d_tools import array2tensor

def apply_transform(point_array, transformation, normals: Optional = None):
    """ apply a transformation on a point array """
    rot, trans = transformation[:3, :3], transformation[:3, 3]
    point_array_transformed = (rot @ point_array.T).T + trans
    if normals is not None:
        normals_transformed = (normals @ rot.T).T
        return point_array_transformed, normals_transformed
    else:
        return point_array_transformed

def compute_inlier_ratio(src_corr_points, tgt_corr_points, transform, positive_radius=0.05):
    """ computing the inlier ratio between a set of point correspondences using a given transformation """
    src_corr_points_2 = apply_transform(src_corr_points, transform)
    # residuals = torch.sqrt(((ref_corr_points - src_corr_points_2) ** 2).sum(1))
    res = torch.linalg.norm(tgt_corr_points - src_corr_points_2, axis=1)
    mask = res < positive_radius
    inlier_ratio = np.mean(array2tensor((res < positive_radius).cpu(), invert=True))
    return inlier_ratio, mask


