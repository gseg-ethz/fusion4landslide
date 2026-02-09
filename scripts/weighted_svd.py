import numpy as np
import os.path as osp
import torch
from utils.common import dir_exist, read_coord_without_name, rot_tran2transform
import open3d as o3d
from utils.o3d_tools import ransac_registration, tensor2pcd

# Compute rigid transformation between a list of 3D-3D correspondences using weighted SVD.
# Modified from https://github.com/qinzheng93/GeoTransformer/blob/main/geotransformer/modules/registration/procrustes.py
def weighted_svd(src_pts, tgt_pts, eps=1e-6, weights=None, weight_thresh=0.0, return_transform=True):
    if src_pts.ndim == 2:
        src_pts = src_pts.unsqueeze(0)
        tgt_pts = tgt_pts.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_pts.shape[0]
    if weights is None:
        weights = torch.ones_like(src_pts[:, :, None])
    weights[weights < weight_thresh] = 0.0
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    # weights = weights.unsqueeze(2)  # (B, N, 1)

    # get the centering coordinates
    centroid_src = torch.sum(src_pts * weights, dim=1, keepdim=True)  # (B, 1, 3)
    centroid_tgt = torch.sum(tgt_pts * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_pts - centroid_src  # (B, N, 3)
    tgt_points_centered = tgt_pts - centroid_tgt  # (B, N, 3)

    # compute rotation
    H = src_points_centered.permute(0, 2, 1) @ (weights * tgt_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = centroid_tgt.permute(0, 2, 1) - R @ centroid_src.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-7,
    return_transform=True,
    return_rmse=True,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    # U, _, V = torch.svd(H.cpu())  # H = USV^T
    # Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    # eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    U, _, V = torch.svd(H)  # H = USV^T
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.dtype).to(src_points.device)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


def refine_local_rigid_correspondences(corr_neigh_2, refine_type='SVD', weights=None):
    if refine_type == 'SVD':
        rot, tra = weighted_procrustes(
            corr_neigh_2[:, :3],
            corr_neigh_2[:, 3:6],
            weights=weights,
            weight_thresh=0.0,
            eps=1e-6,
            return_transform=False,
            return_rmse=True
        )
        res = (rot @ corr_neigh_2[:, :3].T).T + tra - corr_neigh_2[:, 3:6]
        # point 3D rmse
        max_res = 1
        res_pts = torch.linalg.norm(res, axis=1, keepdims=True)
        corr_neigh_2 = corr_neigh_2[res_pts.squeeze() < max_res, :]
        refine_transform_img = torch.eye(4).cuda()
        # refine_transform_img = torch.from_numpy(refine_transform_img).float().cuda()
        refine_transform_img[:3, :3] = rot
        refine_transform_img[:3, 3] = tra
    elif refine_type == 'RANSAC':
        src_pcd = tensor2pcd(corr_neigh_2[:, :3].cpu())
        tgt_pcd = tensor2pcd(corr_neigh_2[:, 3:6].cpu())
        corrs = np.arange(corr_neigh_2.shape[0]).repeat(2).reshape(corr_neigh_2.shape[0], 2)
        corrs = o3d.utility.Vector2iVector(corrs)
        # corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        ransac_registration(src_pcd, tgt_pcd, corrs)
    return corr_neigh_2, refine_transform_img


def main():
    data_path = '../utils/result_transform/'
    dir_exist(data_path)
    src_pts = read_coord_without_name(osp.join(data_path, 'Target_Coord_Manual_Select_BH22_2209.txt'), delimiter=', ')
    tgt_pts = read_coord_without_name(osp.join(data_path, 'Target_Coord_Manual_Select_BH22_2107.txt'), delimiter=', ')

    # remove T4
    src_pts = np.delete(src_pts, 5, axis=0)
    tgt_pts = np.delete(tgt_pts, 5, axis=0)
    src_pts = torch.from_numpy(src_pts).float().cuda()
    tgt_pts = torch.from_numpy(tgt_pts).float().cuda()

    transform_R, transform_t = weighted_procrustes(
        src_pts,
        tgt_pts,
        weights=None,
        weight_thresh=0.0,
        eps=1e-6,
        return_transform=False,
        return_rmse=True
    )
    # transform_svd = weighted_svd(src_pts, tgt_pts, eps=1e-6, weights=None, weight_thresh=0.0, return_transform=True)
    # transform_svd = torch.round(transform_svd, decimals=9)
    transform_R = torch.round(transform_R, decimals=9)
    transform_t = torch.round(transform_t, decimals=9)
    transform_svd = rot_tran2transform(transform_R, transform_t)

    res = (transform_R @ src_pts.T).T + transform_t - tgt_pts

    res_pts = torch.linalg.norm(res, axis=1, keepdims=True)

    np.savetxt(osp.join(data_path, 'transform_svd_BH22.txt'), transform_svd.cpu(), delimiter=' ', fmt='%s')


if __name__ == '__main__':
    main()
