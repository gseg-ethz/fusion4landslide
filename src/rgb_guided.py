import os
import os.path as osp
import gc
import shutil
import copy
import numpy as np
import torch
import open3d as o3d
import matplotlib
import matplotlib.cm as cm
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
from collections import Counter

from easydict import EasyDict as edict
from utils.o3d_tools import (
    array2tensor, icp_registration, tensor2pcd, pcd2tensor,
    pcd2array, array2pcd, ransac_registration
)
from utils.common import dir_exist
from cpp_core.supervoxel_segmentation.build import supervoxel as svl_segmentation


def weighted_procrustes(src_points, ref_points, weights=None, weight_thresh=0.0, eps=1e-7,
                        return_transform=True, return_rmse=True):
    """
    Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.
    Modified from PointDSC.
    """
    if src_points.ndim == 2:
        src_points, ref_points = src_points.unsqueeze(0), ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(weights < weight_thresh, torch.zeros_like(weights), weights)
    weights /= (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)
    src_centered = src_points - src_centroid
    ref_centered = ref_points - ref_centroid

    H = src_centered.permute(0, 2, 1) @ (weights * ref_centered)
    U, _, V = torch.svd(H)
    Ut = U.transpose(1, 2)
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.device)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut
    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(R.device)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        return transform.squeeze(0) if squeeze_first else transform
    else:
        return (R.squeeze(0), t.squeeze(0)) if squeeze_first else (R, t)


def refine_local_rigid_correspondences(corr_neigh_2, refine_type='SVD'):
    """
    Refine local rigid correspondences using either SVD or RANSAC.
    """
    if refine_type == 'SVD':
        rot, tra = weighted_procrustes(
            corr_neigh_2[:, :3], corr_neigh_2[:, 3:6],
            weights=None, weight_thresh=0.0, eps=1e-6,
            return_transform=False, return_rmse=True
        )
        res = (rot @ corr_neigh_2[:, :3].T).T + tra - corr_neigh_2[:, 3:6]
        res_pts = torch.linalg.norm(res, axis=1, keepdims=True)
        mask = res_pts.squeeze() < 2.5 * torch.median(res_pts)
        mask_quality = torch.sum(mask) / res_pts.shape[0] >= 0.70

        corr_neigh_2 = corr_neigh_2[mask, :]
        refine_transform = torch.eye(4).to(corr_neigh_2.device)
        refine_transform[:3, :3] = rot
        refine_transform[:3, 3] = tra
        return corr_neigh_2, refine_transform, mask, mask_quality

    elif refine_type == 'RANSAC':
        src_pcd = tensor2pcd(corr_neigh_2[:, :3].cpu())
        tgt_pcd = tensor2pcd(corr_neigh_2[:, 3:6].cpu())
        corrs = np.arange(corr_neigh_2.shape[0]).repeat(2).reshape(corr_neigh_2.shape[0], 2)
        corrs = o3d.utility.Vector2iVector(corrs)
        ransac_registration(src_pcd, tgt_pcd, corrs)
        return corr_neigh_2


def crop_and_save(img, img_path, start_x, start_y, crop_size, output_path):
    """Crop and save image patches for visualization or patch-wise processing."""
    img_cropped = img.crop((start_x, start_y, start_x + crop_size[0], start_y + crop_size[1]))
    save_dir = osp.join(output_path, 'cropped_images', img_path)
    dir_exist(save_dir)
    img_cropped.save(osp.join(save_dir, f'{start_x}_{start_y}.jpg'))


def quaternion_to_rotation_matrix(q):
    """Convert a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])


def read_2d_matches(path):
    """Read and concatenate 2D matches from a directory."""
    if not osp.exists(path):
        return np.asarray([])

    corr_all = [
        np.loadtxt(osp.join(path, corr_path))
        for corr_path in sorted(os.listdir(path))
    ]
    corr_all_non_empty = [arr for arr in corr_all if arr.size > 0]
    return np.vstack(corr_all_non_empty)


def map_corr_2d_to_3d_filter(corres_2d, src_pixel, tgt_pixel, pixel_thres):
    """
    Construct filtered 3D correspondences from 2D pixel correspondences
    using nearest neighbors and RANSAC outlier rejection.
    """
    num_closest_src = 5
    num_closest_tgt = 5

    s_tree = cKDTree(corres_2d[:, :2])
    _, indices = s_tree.query(src_pixel.cpu(), k=num_closest_src)
    src_tgt_corr_temp = corres_2d[indices]
    src_tgt_corr_avg = src_tgt_corr_temp.mean(axis=1)
    corres_2d_tgt_corrected = src_tgt_corr_avg[:, 2:4]

    try:
        model, inliers = ransac(
            (src_pixel.numpy(), src_tgt_corr_avg[:, :2]),
            AffineTransform,
            min_samples=3,
            residual_threshold=5,
            max_trials=100
        )
    except ValueError:
        inliers = np.zeros(src_pixel.shape[0], dtype=bool)

    mask_corr = np.zeros(src_pixel.shape[0], dtype=bool)
    mask_corr[inliers] = True

    corres_2d_tgt_corrected_filtered = np.zeros_like(corres_2d_tgt_corrected)
    corres_2d_tgt_corrected_filtered[inliers] = corres_2d_tgt_corrected[inliers]

    s_tree_2 = cKDTree(tgt_pixel.cpu())
    indices_2 = np.full(src_pixel.shape[0], fill_value=-1, dtype=int)
    if np.any(inliers):
        distances_2, matched_indices = s_tree_2.query(corres_2d_tgt_corrected[inliers], k=1)
        indices_2[inliers] = matched_indices

    return indices_2, mask_corr, corres_2d_tgt_corrected_filtered


def map_corr_2d_to_3d(corres_2d, src_pixel, tgt_pixel, pixel_thres):
    """
    :param corres_2d:
    :param src_pixel:
    :param tgt_pixel:
    :return: the closest tgt_pixel indices for all src_pixel, validity of all matches
    """
    # find the closest corres_2d_src for src_pixel
    # then get the corres_2d that links to src_pixel
    # then find the closest tgt_pixel for corres_2d_tgt
    # finally, find the src_pixel - tgt_pixel through the mid corres_2d

    # limitation: always consider src > tgt
    # currently only consider closest 1 point
    num_closest_0 = 2
    num_closest = 2
    # pixel_thres = 5
    s_tree = cKDTree(corres_2d[:, :2])
    # search the nearest points from the tree for src_pixel
    distance, indices = s_tree.query(src_pixel.cpu(), k=num_closest_0)
    mask_corr = []
    corr_2d_to_src_tgt = []
    # for i in range(num_closest_0):
    for i in range(1):
        mask_corr_1 = distance[:, i] < pixel_thres

        src_tgt_corr_temp = corres_2d[indices[:, 0], :]

        #################
        # TODO:
        # correct the matches using residuals
        # delta_src = np.asarray(src_pixel) - src_tgt_corr_temp[:, :2]
        # src_tgt_corr_temp[:, 2:4] = src_tgt_corr_temp[:, 2:4] + delta_src
        #################

        s_tree_2 = cKDTree(tgt_pixel.cpu())
        distance_2, indices_2 = s_tree_2.query(src_tgt_corr_temp[:, 2:4], k=num_closest)
        for j in range(num_closest):
            mask_corr_2 = distance_2[:, j] < pixel_thres
            mask_corr_temp = mask_corr_1 & mask_corr_2
            mask_corr.append(mask_corr_temp)
            corr_2d_to_src_tgt.append(src_tgt_corr_temp)
        a = None
    return indices_2[:, 0], mask_corr[0], corr_2d_to_src_tgt[0]

    # for i in range(src_pixel.shape[0]):
    #     curr_src_pixel = src_pixel[i, :]
    #     s_tree = cKDTree(corres_2d[:, :2])
    #     distance, indices = s_tree.query(curr_src_pixel, k=5)


def map_corr_2d_to_3d_tgt2src(corres_2d, src_pixel, tgt_pixel, pixel_thres):
    """
    :param corres_2d:
    :param src_pixel:
    :param tgt_pixel:
    :return: the closest tgt_pixel indices for all src_pixel, validity of all matches
    """
    #### reverse, tgt_pixel --> src_pixel
    # find the closest corres_2d_src for src_pixel
    # then get the corres_2d that links to src_pixel
    # then find the closest tgt_pixel for corres_2d_tgt
    # finally, find the src_pixel - tgt_pixel through the mid corres_2d

    # limitation: always consider src > tgt
    # currently only consider closest 1 point
    num_closest_0 = 2
    num_closest = 2
    # pixel_thres = 5
    t_tree = cKDTree(corres_2d[:, 2:4])
    # search the nearest points from the tree for src_pixel
    distance, indices = t_tree.query(tgt_pixel.cpu(), k=num_closest_0)
    mask_corr = []
    corr_2d_to_src_tgt = []
    # for i in range(num_closest_0):
    for i in range(1):
        mask_corr_1 = distance[:, i] < pixel_thres

        tgt_src_corr_temp = corres_2d[indices[:, 0], :]

        s_tree = cKDTree(src_pixel.cpu())
        distance_2, indices_2 = s_tree.query(tgt_src_corr_temp[:, :2], k=num_closest)
        for j in range(num_closest):
            mask_corr_2 = distance_2[:, j] < pixel_thres
            mask_corr_temp = mask_corr_1 & mask_corr_2
            mask_corr.append(mask_corr_temp)
            corr_2d_to_src_tgt.append(tgt_src_corr_temp)
        a = None
    return indices_2[:, 0], mask_corr[0], corr_2d_to_src_tgt[0]

    # for i in range(src_pixel.shape[0]):
    #     curr_src_pixel = src_pixel[i, :]
    #     s_tree = cKDTree(corres_2d[:, :2])
    #     distance, indices = s_tree.query(curr_src_pixel, k=5)


class Image_DVFs(abc.ABC):
    def __init__(self, config):

        self.config = config
        self.logging = config.logging
        self.verbose = config.verbose
        self.save_interim = config.save_interim
        # self.voxel_size = config.voxel_size
        # self.points_per_batch = config.points_per_batch
        self.device = self.config.device
        # self.batch_size = config.batch_size
        # self.num_workers = config.num_workers

        # self.dicts = edict(dict())
        # save variables to separate dicts
        # self.dicts.path, self.dicts.visualize, self.dicts.method, self.dicts.hypara, self.dicts.debug = (
        #     config.path_name, config.visualization, config.method, config.parameter_setting, config.debugging)

        # initialize some useful variables
        self._initialize()

        self._read_data()

    def implement_segmentation(self):
        if self.verbose:
            self.logging.info(f'Start clustering with {self.method.clustering_type}...')
        if self.method.clustering_type in ['supervoxel', 'hdbscan']:

            svl_radius = np.max((np.sqrt(3) * (10 * self._compute_median_resolution()), self.method.voxel_size))
            # svl_radius = svl_radius * 3
            # svl_radius = 0.1
            if self.data.dataset == 'rockfall_simulator':
                svl_radius = 0.1

            segment_path = osp.join(self.output_root, f'{self.method.clustering_type}_segmentation')
            dir_exist(segment_path)

            if self.method.clustering_type == 'supervoxel':
                segment_id_src_pts_input = svl_segmentation.computeSupervoxel(
                    self.src_pcd_path, self.method.n_normals, svl_radius,
                    osp.join(segment_path, f'segment_of_input_src_tile_{self.config.tile_id}.txt'))

                # type for each pts in input src pts
                segment_id_src_pts_input = np.asarray(segment_id_src_pts_input).reshape(-1, 1)
            elif self.method.clustering_type == 'hdbscan':
                scaler = StandardScaler()
                points_scaled = scaler.fit_transform(self.data_input.src_pts.cpu().numpy()[:, :])
                clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1000, gen_min_span_tree=True)
                clusterer.fit(points_scaled)
                segment_id_src_pts_input = clusterer.labels_

                # get colors for each labels, then save to path
                import matplotlib.pyplot as plt
                unique_labels = np.unique(segment_id_src_pts_input)

                # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

                # colors = generate_distinct_colors(len(unique_labels))
                # colors = (colors[:, :3] * 255).astype(int)
                src_colors = np.ones((len(segment_id_src_pts_input), 3))

                CLASS_COLORS = np.asarray([
                    [243, 214, 171],  # sunset
                    [70, 115, 66],  # fern green
                    [233, 50, 239],
                    [243, 238, 0],
                    [190, 153, 153],
                    [0, 233, 11],
                    [239, 114, 0],
                    [214, 66, 54],  # vermillon
                    [0, 8, 116]])
                colors = CLASS_COLORS

                for label, color in zip(unique_labels, colors):
                    if label == -1:
                        color = [0, 0, 0]
                    else:
                        color = colors[label % len(CLASS_COLORS)]
                    mask = segment_id_src_pts_input == label
                    src_colors[mask, :] = color

                src_pts_colors_labels = np.hstack((self.data_input.src_pts.cpu().numpy(), src_colors, segment_id_src_pts_input[:, None]))
                # save to pathc
                np.savetxt(osp.join(segment_path, 'segment_of_input_src.txt'), src_pts_colors_labels)

                segment_id_src_pts_input = segment_id_src_pts_input.reshape(-1, 1)


            # svl_idx = array2tensor(svl_idx).to(self.device)
            # only select valid pts for segmentation
            segment_id_src_pts_in_dvfs = segment_id_src_pts_input[self.data_output.idx_valid_src.cpu()]

            # Extract the indices of individual supervoxels
            segment_patch_id_full = np.unique(segment_id_src_pts_in_dvfs)

            if self.verbose:
                # self.logging.info('----------------------------------------------------------------------')
                self.logging.info(f'Remove supervoxels that contain less than 10 points')
            # Count the occurrences of each index
            segment_id_src_pts_in_dvfs = segment_id_src_pts_in_dvfs.flatten()
            counts = Counter(segment_id_src_pts_in_dvfs)
            # Only keep indices that occur more than 10 times
            # 120 for landslides, 10 for rockfall simulator
            # segment_patch_id_valid = {idx for idx, count in counts.items() if count > 120 and idx != -1}
            segment_patch_id_valid = {idx for idx, count in counts.items() if count > 10 and idx != -1}
            # Find the valid indices and store them
            supervoxel_list_valid = []
            supervoxel_list_invalid = []
            supervoxel_list_invalid_local = []
            for idx in tqdm(segment_patch_id_full, position=0, leave=True):
                if idx in segment_patch_id_valid:
                    supervoxel_list_valid.append(self.data_output.idx_valid_src
                                                 [np.where(segment_id_src_pts_in_dvfs == idx)[0]])
                else:
                    supervoxel_list_invalid.append(self.data_output.idx_valid_src
                                                   [np.where(segment_id_src_pts_in_dvfs == idx)[0]])
                    supervoxel_list_invalid_local.append(np.where(segment_id_src_pts_in_dvfs == idx)[0])

            # update the output indices of src_pts, remove src pts indices in invalid supervoxel
            # tgt indices share the same indices
            if supervoxel_list_invalid_local == []:
                supervoxel_list_invalid_local = []
            else:
                supervoxel_list_invalid_local = np.hstack(supervoxel_list_invalid_local)
            mask_pts_valid = torch.ones(len(self.data_output.idx_valid_src), dtype=bool)
            mask_pts_valid[supervoxel_list_invalid_local] = False

            self.data_output.idx_valid_src_refine = self.data_output.idx_valid_src[mask_pts_valid]
            self.data_output.idx_valid_tgt_refine = self.data_output.idx_valid_tgt[mask_pts_valid]
            self.data_output.corres_3d_refine = self.data_output.corres_3d[mask_pts_valid, :]
            self.data_output.corres_3d_magnitude_refine = self.data_output.corres_3d_magnitude
            [mask_pts_valid]

            self.data_interim.segment_patches = supervoxel_list_valid
        # elif self.config.segment_type == 'dbscan':
        #     eps = 0.5  # Adjust based on your data's scale
        #     min_samples = 10  # Adjust based on your expected minimum cluster size
        #
        #     segment_path = osp.join(self.output_root, f'{self.config.segment_type}_segmentation')
        #     dir_exist(segment_path)
        #
        #     # dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree', n_jobs=-1)
        #     # dbscan.fit_predict(self.data_input.src_pts.cpu().numpy())
        #
        #     # dbscan.fit_predict(Y)
        #     # labels = dbscan.labels_
        #     # print(labels.size)
        #     # clusters = labels.tolist()
        #
        #     import hdbscan
        #     from sklearn.preprocessing import StandardScaler
        #
        #     scaler = StandardScaler()
        #     points_scaled = scaler.fit_transform(self.data_input.src_pts.cpu().numpy())
        #     clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10, gen_min_span_tree=True)
        #     clusterer.fit(points_scaled)
        #     segment_id_src_pts_input = clusterer.labels_

    def local_rigid_refinement(self):
        if self.verbose:
            self.logging.info(f'Start rigid refinement...')
        # refine matches for each segmentation type
        mask_valid_local = []
        corres_3d_refine_apply_icp = []
        for patch_i in tqdm(self.data_interim.segment_patches[:], position=0, leave=True):
            # 1351302, segment_patches wrong
            # select idx in
            idx = [torch.where(self.data_output.idx_valid_src_refine == value)[0] for value in patch_i]
            idx = torch.hstack(idx)
            temp_corr = self.data_output.corres_3d_refine[idx, :]

            temp_corr_pruning, temp_transform_initial, temp_mask_valid, mask_robust = refine_local_rigid_correspondences(
                temp_corr,
                refine_type='SVD')
            # remove poor estimation patches
            # if self.config.binary_filtering:
            #     temp_corr_scaled = torch.divide(temp_corr, torch.max(torch.abs(temp_corr)))
            #     filtering_output = self.outlier_removal_nn.filter_input(
            #         temp_corr_scaled.unsqueeze(0).unsqueeze(0).float(),
            #         temp_corr.unsqueeze(0).float())
            #     if not filtering_output['robust_estimate']:
            #         mask = filtering_output['scores'].squeeze() > 0.999
            #         temp_corr = temp_corr[mask]
            # mask not used
            mask_valid_local.append(idx[np.asarray(temp_mask_valid.cpu())])

            if self.method.icp_refine and temp_corr.shape[0] > 0:
                temp_transform_region_initial = temp_transform_initial.cpu()
                # only do it for temp_corr_pruning, not for all pts in original pcd?
                # temp_src_pcd = tensor2pcd(temp_corr_pruning[:, :3])
                # temp_tgt_pcd = tensor2pcd(temp_corr_pruning[:, 3:6])
                # but now do it for all pts
                temp_src_pcd = tensor2pcd(temp_corr[:, :3])
                temp_tgt_pcd = tensor2pcd(temp_corr[:, 3:6])

                # refine img matches tgt using icp based on reliable img matches
                icp_result = icp_registration(temp_src_pcd, temp_tgt_pcd, threshold=self.method.icp_thres,
                                              initial_transform=temp_transform_region_initial, icp_type='point2point')

                icp_transform_refine = np.eye(4)
                icp_transform_refine[:3, :3] = icp_result['est_transform'][:3, :3]
                icp_transform_refine[:3, 3] = icp_result['est_transform'][:3, 3]
                icp_transform_refine = torch.tensor(icp_transform_refine, dtype=torch.float32).cuda()

                # temp_dvf_region_refine_src = temp_dvf_region_pruning[:, :3]
                temp_corr_src = temp_corr[:, :3]
                temp_corr_tgt_apply_icp = ((icp_transform_refine[:3, :3] @ temp_corr_src.T).T +
                                              icp_transform_refine[:3, 3])

                ##############
                # check how large of the residuals
                # temp_corr_tgt = temp_corr[:, 3:6]
                # residual_icp = torch.linalg.norm(temp_corr_tgt_apply_icp - temp_corr_tgt, axis=1)
                # inlier = residual_icp <= 2.5 * torch.mean(residual_icp)

                # if torch.sum(inlier) / len(inlier) >= 0.9:
                ##############

                temp_corr_apply_icp = torch.hstack([temp_corr_src, temp_corr_tgt_apply_icp])
                # temp_corr_apply_icp = torch.hstack([temp_corr_src, temp_corr_tgt_apply_icp])[inlier]

                # if mask_robust:
                #     corres_3d_refine_apply_icp.append(temp_corr_apply_icp)

                corres_3d_refine_apply_icp.append(temp_corr_apply_icp)

            # temp_corr = temp_corr.unsqueeze(0)
            # x1, x2 = temp_corr[:, :, :3], temp_corr[:, :, 3:6]
            # rotation_est, translation_est, residuals, _ = kabsch_transformation_estimation(
            #     x1, x2)

                a = 0

            a = 0

        mask_valid_local = torch.hstack(mask_valid_local)
        self.data_output.idx_valid_src_refine = self.data_output.idx_valid_src_refine[mask_valid_local]
        self.data_output.idx_valid_tgt_refine = self.data_output.idx_valid_tgt_refine[mask_valid_local]

        # ?????? what's the mask? only used for refine, but no icp replace
        self.data_output.corres_3d_refine = self.data_output.corres_3d_refine[mask_valid_local, :]
        self.data_output.corres_3d_magnitude_refine = self.data_output.corres_3d_magnitude_refine[mask_valid_local]

        if self.method.icp_refine:
            self.data_output.corres_3d_refine_apply_icp = torch.vstack(corres_3d_refine_apply_icp)
            self.data_output.corres_3d_magnitude_refine_apply_icp = torch.linalg.norm(
                self.data_output.corres_3d_refine_apply_icp[:, 3:6] -
                self.data_output.corres_3d_refine_apply_icp[:, :3], dim=1)[:, None]

    def implement_rgb_guided_estimation(self):
        # project 3d points to 2d image pixels
        # can be uniformed
        # if self.config.dataset == 'rockfall_simulator':
        #     self.pcd2img_rockfall_simulator()
        # elif self.config.dataset == 'brienz':
        #     self.pcd2img_brienz()
        if self.method.num_sub_img == 1:
            self.pcd2img_with_different_types()

            if self.data.multiple_case:
                self._get_2d_matches_online()

                if self.verbose:
                    self.logging.info('Implementing 2d matching online is done!')

            # load image matches from Efficient LoftR
            if self.data.multiple_case:
                corres_2d_initial = read_2d_matches(
                    osp.join(self.output_root, self.config.path_name.img_matching_result_dir, f'tile_{self.config.tile_id}', 'corres_2d'))
            else:
                corres_2d_initial = read_2d_matches(osp.join(self.output_root, self.config.path_name.img_matching_result_dir, 'corres_2d'))

            # for better visualization in CloudCompare
            # only for visualization, not for real computation; can be commented for faster computation
            if self.data.dataset == 'rockfall_simulator':
                max_mag_visualize = 0.06
            elif self.data.dataset == 'brienz':
                max_mag_visualize = 5
            elif self.data.dataset == 'mattertal':
                max_mag_visualize = 10
            else:
                max_mag_visualize = 10

            if corres_2d_initial.shape[0] > 0:
                if self.method.lifting_type == 'nn_search':
                    # find the corresponding indices of src_pixel from tgt_pixel
                    # idx all, valid indices,
                    idx_proj_src2tgt, mask_in_pixel_thres_src, _ = map_corr_2d_to_3d(
                        corres_2d_initial, self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt, self.method.pixel_thres)

                elif self.method.lifting_type == 'nn_search_filter':
                    # find the corresponding indices of src_pixel from tgt_pixel
                    # idx all, valid indices,
                    idx_proj_src2tgt, mask_in_pixel_thres_src, _ = map_corr_2d_to_3d_filter(
                        corres_2d_initial, self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt, self.method.pixel_thres)
                elif self.method.lifting_type == 'interpolation':
                    points_3d_matches, point_3d_matches_mask = self.lift_2d_to_3d_with_interpolation(
                        torch.from_numpy(corres_2d_initial[:, :4]).to(torch.float32).to(self.device), batch_size=1000000)
                    a = 0
                else:
                    NotImplementedError

                ########
                # save interpolation-based 2d lifted matches
                # points_3d_matches_valid = points_3d_matches[point_3d_matches_mask]
                # magnitude_visualize = torch.linalg.norm(points_3d_matches_valid[:, :3] - points_3d_matches_valid[:, 3:6], axis=1).to(self.device)
                # magnitude_visualize[0] = 0
                # magnitude_visualize[1] = max_mag_visualize
                # dvfs_3d_from_2d_vector_mag_visualize = torch.hstack(
                #     (points_3d_matches_valid[:, :3], magnitude_visualize[:, None]))
                # dir_exist(osp.join(self.output_root, 'results'))
                # np.savetxt(
                #     osp.join(self.output_root, 'results', 'c2f_dvfms_from_2d_interpolation_src2tgt_without_pruning_visualize_0_5.txt'),
                #     dvfs_3d_from_2d_vector_mag_visualize.cpu())
                ########

                if self.method.save_initial_matches_from_2d and np.sum(mask_in_pixel_thres_src) > 0:
                    ###############
                    # get the interim result for comparison
                    corres_3d_visualize = torch.zeros(
                        (len(self.data_interim.idx_valid_src[mask_in_pixel_thres_src]), 6)).to(self.device)
                    corres_3d_visualize[:, :3] = self.data_input_3d.src_pts[self.data_interim.idx_valid_src][
                                                 mask_in_pixel_thres_src, :3]
                    corres_3d_visualize[:, 3:6] = self.data_input_3d.tgt_pts[self.data_interim.idx_valid_tgt][
                                                  idx_proj_src2tgt[mask_in_pixel_thres_src], :3]

                    magnitude_visualize = torch.linalg.norm(corres_3d_visualize[:, :3] - corres_3d_visualize[:, 3:6],
                                                            axis=1).to(self.device)
                    magnitude_visualize[0] = 0
                    magnitude_visualize[1] = max_mag_visualize
                    # magnitude = magnitude[mask_in_pixel_thres_tgt]
                    dvfs_3d_from_2d_vector_mag_visualize = torch.hstack(
                        (corres_3d_visualize[:, :3], magnitude_visualize[:, None]))
                    dir_exist(osp.join(self.output_root, 'results'))
                    if self.data.multiple_case:
                        np.savetxt(osp.join(self.output_root, 'results',
                                            f'rgb_guided_raw_wo_pruning_dvfms_src2tgt_visualize_tile_{self.config.tile_id}.txt'),
                                   dvfs_3d_from_2d_vector_mag_visualize.cpu())
                    else:
                        np.savetxt(osp.join(self.output_root, 'results',
                                            'rgb_guided_raw_wo_pruning_dvfms_src2tgt_visualize.txt'),
                                   dvfs_3d_from_2d_vector_mag_visualize.cpu())
                    ###############

                if self.method.matches_from_2d_type in ['nn_mutual', 'nn_union', 'nn_src_with_tgt_for_visualize']:
                    # find the corresponding indices of tgt_pixel from src_pixel
                    # tgt_pixel --> src_pixel
                    idx_proj_tgt2src, mask_in_pixel_thres_tgt, _ = map_corr_2d_to_3d_tgt2src(
                        corres_2d_initial, self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt,
                        self.method.pixel_thres)

                    ###############
                    # tgt2src based on src visualization, therefore many overlapped points result in very sparse DVFs
                    if self.method.save_initial_matches_from_2d and np.sum(mask_in_pixel_thres_tgt) > 0:
                        # get the interim result for comparison
                        corres_3d_visualize_2 = torch.zeros(
                            (len(self.data_interim.idx_valid_tgt[mask_in_pixel_thres_tgt]), 6)).to(self.device)
                        corres_3d_visualize_2[:, :3] = self.data_input_3d.src_pts[self.data_interim.idx_valid_src][
                                                       idx_proj_tgt2src[mask_in_pixel_thres_tgt], :3]
                        corres_3d_visualize_2[:, 3:6] = self.data_input_3d.tgt_pts[self.data_interim.idx_valid_tgt][
                                                        mask_in_pixel_thres_tgt, :3]
                        magnitude_visualize_2 = torch.linalg.norm(
                            corres_3d_visualize_2[:, :3] - corres_3d_visualize_2[:, 3:6],
                            axis=1).to(self.device)
                        magnitude_visualize_2[0] = 0
                        magnitude_visualize_2[1] = max_mag_visualize
                        # magnitude = magnitude[mask_in_pixel_thres_tgt]
                        dvfs_3d_from_2d_vector_mag_visualize_2 = torch.hstack(
                            (corres_3d_visualize_2[:, :3], magnitude_visualize_2[:, None]))
                        dir_exist(osp.join(self.output_root, 'results'))
                        dir_exist(osp.join(self.output_root, 'results'))
                        if self.data.multiple_case:
                            np.savetxt(
                                osp.join(self.output_root, 'results',
                                         f'rgb_guided_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5_tile_{self.config.tile_id}.txt'),
                                dvfs_3d_from_2d_vector_mag_visualize_2.cpu())
                        else:
                            np.savetxt(
                                osp.join(self.output_root, 'results',
                                         'rgb_guided_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5.txt'),
                                dvfs_3d_from_2d_vector_mag_visualize_2.cpu())
                    ###############

                if self.method.matches_from_2d_type == 'nn_mutual':
                    #################
                    # only select mutual intersection matches
                    mask_intersection = torch.zeros(len(mask_in_pixel_thres_src), dtype=torch.bool)
                    for n, i in enumerate(idx_proj_src2tgt):
                        if mask_in_pixel_thres_src[n] and mask_in_pixel_thres_tgt[i]:
                            # only the src_1 = src_2, and tgt_1 == tgt_2 then select it
                            mask_temp = idx_proj_tgt2src[i] == n
                            if mask_temp:
                                mask_intersection[n] = True
                elif self.method.matches_from_2d_type == 'nn_union':
                    # check, only limit the pts to src? doesn't work unless considering the tgt pts or even new pts
                    # select union intersection  matches
                    mask_intersection = torch.zeros(len(mask_in_pixel_thres_src), dtype=torch.bool)
                    for n, i in enumerate(idx_proj_src2tgt):
                        if mask_in_pixel_thres_src[n] or mask_in_pixel_thres_tgt[i]:
                            mask_temp = idx_proj_tgt2src[i] == n
                            if mask_temp:
                                mask_intersection[n] = True
                elif self.method.matches_from_2d_type in ['nn_src_only', 'nn_src_with_tgt_for_visualize']:
                    # only use src source
                    mask_intersection = mask_in_pixel_thres_src

                # get intial corrs_3d, from 2d matches, within the given pixel threshold
                corres_3d = torch.zeros((len(self.data_interim.idx_valid_src[mask_intersection]), 6)).to(self.device)
                corres_3d[:, :3] = self.data_input_3d.src_pts[self.data_interim.idx_valid_src][mask_intersection, :3]
                # tgt_pts_valid is wrong?
                corres_3d[:, 3:6] = self.data_input_3d.tgt_pts[self.data_interim.idx_valid_tgt][
                                    idx_proj_src2tgt[mask_intersection], :3]

                # remove indices that doesn't meet the pixel threshold; these indices will be further updated
                idx_valid_src = self.data_interim.idx_valid_src[mask_intersection]
                idx_valid_tgt = self.data_interim.idx_valid_tgt[idx_proj_src2tgt[mask_intersection]]
                #################

                # outlier removal
                magnitude = torch.linalg.norm(corres_3d[:, :3] - corres_3d[:, 3:6], axis=1).to(self.device)
                mask_within_max_mag = magnitude <= self.method.max_magnitude

                #################
                # # TODO:
                # np.savetxt(osp.join(self.output_root, 'results',
                #                     f'rgb_guided_dvfs_wo_refinement_{self.config.tile_id}.txt'),
                #            corres_3d_visualize.cpu(), delimiter=' ', fmt='%.6f')
                # labels = np.array([f's{i}' for i in range(len(corres_3d[:, 0]))])[:, None]
                # output_data_src = np.column_stack((labels, corres_3d[:, :3].cpu().numpy()))
                # output_data_tgt = np.column_stack((labels, corres_3d[:, 3:6].cpu().numpy()))
                # np.savetxt(osp.join(self.output_root, 'results',
                #                     f'rgb_guided_dvfs_wo_refinement_src_{self.config.tile_id}.txt'),
                #            output_data_src, fmt='%s', delimiter=' ')
                # np.savetxt(osp.join(self.output_root, 'results',
                #                     f'rgb_guided_dvfs_wo_refinement_tgt_{self.config.tile_id}.txt'),
                #            output_data_tgt, fmt='%s', delimiter=' ')
                #################

                # not really needed if the idx (with unmatched -1) is saved
                # self.data_interim.corres_3d_from_2d = corres_3d[mask_within_max_mag, :]
                corres_3d_from_2d = corres_3d[mask_within_max_mag, :]

                magnitude_visualize = magnitude[mask_within_max_mag]
                # TODO: comment unnecessary results for faster computation
                # if magnitude_visualize.numel() < 2:
                #     # min: 2
                #     if self.verbose:
                #         self.logging.info(f'Current tile_{self.config.tile_id} has no valid corres_3d_from_2d')
                # else:
                #     magnitude_visualize[0] = 0
                #     magnitude_visualize[1] = max_mag_visualize
                #     dvfs_3d_from_2d_vector_mag = torch.hstack((corres_3d_from_2d[:, :3], magnitude_visualize[:, None]))
                #     if self.data.multiple_case:
                #         np.savetxt(osp.join(self.output_root, 'results',
                #                             f'rgb_guided_dvfms_from_2d_with_max_mag_removal_visualize_0_5_tile_{self.config.tile_id}.txt'),
                #                    dvfs_3d_from_2d_vector_mag.cpu())
                #     else:
                #         np.savetxt(osp.join(self.output_root, 'results',
                #                             'rgb_guided_dvfms_from_2d_with_max_mag_removal_visualize_0_5.txt'),
                #                    dvfs_3d_from_2d_vector_mag.cpu())

                # remove indices that has more than max. mag. these indices will be further updated until the final output
                idx_valid_src = idx_valid_src[mask_within_max_mag]
                idx_valid_tgt = idx_valid_tgt[mask_within_max_mag]

                self.data_interim.idx_valid_src_from_2d = idx_valid_src
                self.data_interim.idx_valid_tgt_from_2d = idx_valid_tgt

                # keep the indices of unmatched pts as -1
                corres_3d_from_2d_idx = - torch.ones(self.data_input_3d.idx_initial_src.shape[0], 2).to(torch.int64).to(
                    self.device)
                corres_3d_from_2d_idx[:, 0] = torch.arange(self.data_input_3d.idx_initial_src.shape[0])
                corres_3d_from_2d_idx[idx_valid_src, 1] = idx_valid_tgt

                self.data_interim.corres_3d_from_2d_idx = corres_3d_from_2d_idx

                # clear not used variables, related to self
                del self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt, (
                    self.data_interim.idx_valid_src), self.data_interim.idx_valid_tgt, (
                    self.data_interim.idx_valid_src_from_2d), self.data_interim.idx_valid_tgt_from_2d

            else:
                self.logging.info('No initial pixel matches for the current tile')

                # no valid 3d matches from 2d, even in the very initial image matching step, before outlier removal
                corres_3d_from_2d_idx = - torch.ones(self.data_input_3d.idx_initial_src.shape[0], 2).to(torch.int64).to(
                    self.device)
                corres_3d_from_2d_idx[:, 0] = torch.arange(self.data_input_3d.idx_initial_src.shape[0])

                self.data_interim.corres_3d_from_2d_idx = corres_3d_from_2d_idx
                a = 0

        else:
            corres_3d_from_2d_idx_list = []
            for num_src in np.arange(self.method.num_sub_img):
                for num_tgt in np.arange(self.method.num_sub_img):
                    self.data_input_2d.src_image = self.data_input_2d.src_image_list[num_src]
                    self.data_input_2d.tgt_image = self.data_input_2d.tgt_image_list[num_tgt]
                    self.data_input_2d.src_extrinsic = self.data_input_2d.src_extrinsic_list[num_src]
                    self.data_input_2d.tgt_extrinsic = self.data_input_2d.tgt_extrinsic_list[num_tgt]

                    self.pcd2img_with_different_types()

                    if self.data.multiple_case:
                        self._get_2d_matches_online()

                        if self.verbose:
                            self.logging.info('Implementing 2d matching online is done!')

                    # load image matches from Efficient LoftR
                    if self.data.multiple_case:
                        corres_2d_initial = read_2d_matches(
                            osp.join(self.output_root, self.config.path_name.img_matching_result_dir,
                                     f'tile_{self.config.tile_id}', 'corres_2d'))
                    else:
                        corres_2d_initial = read_2d_matches(
                            osp.join(self.output_root, self.config.path_name.img_matching_result_dir, 'corres_2d'))

                    # for better visualization in CloudCompare
                    # only for visualization, not for real computation; can be commented for faster computation
                    if self.data.dataset == 'rockfall_simulator':
                        max_mag_visualize = 0.06
                    elif self.data.dataset == 'brienz':
                        max_mag_visualize = 5
                    elif self.data.dataset == 'mattertal':
                        max_mag_visualize = 10
                    else:
                        max_mag_visualize = 10

                    if corres_2d_initial.shape[0] > 0:
                        if self.method.lifting_type == 'nn_search':
                            # find the corresponding indices of src_pixel from tgt_pixel
                            # idx all, valid indices,
                            idx_proj_src2tgt, mask_in_pixel_thres_src, _ = map_corr_2d_to_3d(
                                corres_2d_initial, self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt, self.method.pixel_thres)
                        elif self.method.lifting_type == 'interpolation':
                            points_3d_matcheColors, point_3d_matches_mask = self.lift_2d_to_3d_with_interpolation(
                                torch.from_numpy(corres_2d_initial[:, :4]).to(torch.float32).to(self.device), batch_size=1000000)
                            a = 0
                        else:
                            NotImplementedError

                        ########
                        # save interpolation-based 2d lifted matches
                        # points_3d_matches_valid = points_3d_matches[point_3d_matches_mask]
                        # magnitude_visualize = torch.linalg.norm(points_3d_matches_valid[:, :3] - points_3d_matches_valid[:, 3:6], axis=1).to(self.device)
                        # magnitude_visualize[0] = 0
                        # magnitude_visualize[1] = max_mag_visualize
                        # dvfs_3d_from_2d_vector_mag_visualize = torch.hstack(
                        #     (points_3d_matches_valid[:, :3], magnitude_visualize[:, None]))
                        # dir_exist(osp.join(self.output_root, 'results'))
                        # np.savetxt(
                        #     osp.join(self.output_root, 'results', 'rgb_guided_dvfms_from_2d_interpolation_src2tgt_without_pruning_visualize_0_5.txt'),
                        #     dvfs_3d_from_2d_vector_mag_visualize.cpu())
                        ########

                        if self.method.save_initial_matches_from_2d and np.sum(mask_in_pixel_thres_src) > 0:
                            ###############
                            # get the interim result for comparison
                            corres_3d_visualize = torch.zeros(
                                (len(self.data_interim.idx_valid_src[mask_in_pixel_thres_src]), 6)).to(self.device)
                            corres_3d_visualize[:, :3] = self.data_input_3d.src_pts[self.data_interim.idx_valid_src][
                                                         mask_in_pixel_thres_src, :3]
                            corres_3d_visualize[:, 3:6] = self.data_input_3d.tgt_pts[self.data_interim.idx_valid_tgt][
                                                          idx_proj_src2tgt[mask_in_pixel_thres_src], :3]
                            # magnitude_visualize = torch.linalg.norm(corres_3d_visualize[:, :3] - corres_3d_visualize[:, 3:6],
                            #                                         axis=1).to(self.device)
                            # magnitude_visualize[0] = 0
                            # magnitude_visualize[1] = max_mag_visualize
                            # # magnitude = magnitude[mask_in_pixel_thres_tgt]
                            # dvfs_3d_from_2d_vector_mag_visualize = torch.hstack(
                            #     (corres_3d_visualize[:, :3], magnitude_visualize[:, None]))
                            # dir_exist(osp.join(self.output_root, 'results'))
                            # if 'full' in self.input_root:
                            #     np.savetxt(osp.join(self.output_root, 'results',
                            #                         f'rgb_guided_dvfms_from_2d_src2tgt_without_pruning_visualize_0_5_tile_{self.config.tile_id}.txt'),
                            #                dvfs_3d_from_2d_vector_mag_visualize.cpu())
                            # else:
                            #     np.savetxt(osp.join(self.output_root, 'results',
                            #                         'rgb_guided_dvfms_from_2d_src2tgt_without_pruning_visualize_0_5.txt'),
                            #                dvfs_3d_from_2d_vector_mag_visualize.cpu())
                            ###############

                        if self.method.matches_from_2d_type in ['nn_mutual', 'nn_union', 'nn_src_with_tgt_for_visualize']:
                            # find the corresponding indices of tgt_pixel from src_pixel
                            # tgt_pixel --> src_pixel
                            idx_proj_tgt2src, mask_in_pixel_thres_tgt, _ = map_corr_2d_to_3d_tgt2src(
                                corres_2d_initial, self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt,
                                self.method.pixel_thres)

                            ###############
                            # tgt2src based on src visualization, therefore many overlapped points result in very sparse DVFs
                            if self.method.save_initial_matches_from_2d and np.sum(mask_in_pixel_thres_tgt) > 0:
                                # get the interim result for comparison
                                corres_3d_visualize_2 = torch.zeros(
                                    (len(self.data_interim.idx_valid_tgt[mask_in_pixel_thres_tgt]), 6)).to(self.device)
                                corres_3d_visualize_2[:, :3] = self.data_input_3d.src_pts[self.data_interim.idx_valid_src][
                                                               idx_proj_tgt2src[mask_in_pixel_thres_tgt], :3]
                                corres_3d_visualize_2[:, 3:6] = self.data_input_3d.tgt_pts[self.data_interim.idx_valid_tgt][
                                                                mask_in_pixel_thres_tgt, :3]
                                magnitude_visualize_2 = torch.linalg.norm(
                                    corres_3d_visualize_2[:, :3] - corres_3d_visualize_2[:, 3:6],
                                    axis=1).to(self.device)
                                magnitude_visualize_2[0] = 0
                                magnitude_visualize_2[1] = max_mag_visualize
                                # magnitude = magnitude[mask_in_pixel_thres_tgt]
                                dvfs_3d_from_2d_vector_mag_visualize_2 = torch.hstack(
                                    (corres_3d_visualize_2[:, :3], magnitude_visualize_2[:, None]))
                                dir_exist(osp.join(self.output_root, 'results'))
                                dir_exist(osp.join(self.output_root, 'results'))
                                if self.data.multiple_case:
                                    np.savetxt(
                                        osp.join(self.output_root, 'results',
                                                 f'rgb_guided_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5_tile_{self.config.tile_id}.txt'),
                                        dvfs_3d_from_2d_vector_mag_visualize_2.cpu())
                                else:
                                    np.savetxt(
                                        osp.join(self.output_root, 'results',
                                                 'rgb_guided_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5.txt'),
                                        dvfs_3d_from_2d_vector_mag_visualize_2.cpu())
                            ###############

                        if self.method.matches_from_2d_type == 'nn_mutual':
                            #################
                            # only select mutual intersection matches
                            mask_intersection = torch.zeros(len(mask_in_pixel_thres_src), dtype=torch.bool)
                            for n, i in enumerate(idx_proj_src2tgt):
                                if mask_in_pixel_thres_src[n] and mask_in_pixel_thres_tgt[i]:
                                    # only the src_1 = src_2, and tgt_1 == tgt_2 then select it
                                    mask_temp = idx_proj_tgt2src[i] == n
                                    if mask_temp:
                                        mask_intersection[n] = True
                        elif self.method.matches_from_2d_type == 'nn_union':
                            # check, only limit the pts to src? doesn't work unless considering the tgt pts or even new pts
                            # select union intersection  matches
                            mask_intersection = torch.zeros(len(mask_in_pixel_thres_src), dtype=torch.bool)
                            for n, i in enumerate(idx_proj_src2tgt):
                                if mask_in_pixel_thres_src[n] or mask_in_pixel_thres_tgt[i]:
                                    mask_temp = idx_proj_tgt2src[i] == n
                                    if mask_temp:
                                        mask_intersection[n] = True
                        elif self.method.matches_from_2d_type in ['nn_src_only', 'nn_src_with_tgt_for_visualize']:
                            # only use src source
                            mask_intersection = mask_in_pixel_thres_src

                        # get intial corrs_3d, from 2d matches, within the given pixel threshold
                        corres_3d = torch.zeros((len(self.data_interim.idx_valid_src[mask_intersection]), 6)).to(self.device)
                        corres_3d[:, :3] = self.data_input_3d.src_pts[self.data_interim.idx_valid_src][mask_intersection, :3]
                        # tgt_pts_valid is wrong?
                        corres_3d[:, 3:6] = self.data_input_3d.tgt_pts[self.data_interim.idx_valid_tgt][
                                            idx_proj_src2tgt[mask_intersection], :3]

                        # remove indices that doesn't meet the pixel threshold; these indices will be further updated
                        idx_valid_src = self.data_interim.idx_valid_src[mask_intersection]
                        idx_valid_tgt = self.data_interim.idx_valid_tgt[idx_proj_src2tgt[mask_intersection]]
                        #################

                        # outlier removal
                        magnitude = torch.linalg.norm(corres_3d[:, :3] - corres_3d[:, 3:6], axis=1).to(self.device)
                        mask_within_max_mag = magnitude <= self.method.max_magnitude

                        # not really needed if the idx (with unmatched -1) is saved
                        # self.data_interim.corres_3d_from_2d = corres_3d[mask_within_max_mag, :]
                        corres_3d_from_2d = corres_3d[mask_within_max_mag, :]

                        magnitude_visualize = magnitude[mask_within_max_mag]
                        if magnitude_visualize.numel() < 2:
                            # min: 2
                            if self.verbose:
                                self.logging.info(f'Current tile_{self.config.tile_id} has no valid corres_3d_from_2d for sub_img [{num_src}, {num_tgt}]')
                        else:
                            magnitude_visualize[0] = 0
                            magnitude_visualize[1] = max_mag_visualize
                            dvfs_3d_from_2d_vector_mag = torch.hstack((corres_3d_from_2d[:, :3], magnitude_visualize[:, None]))
                            # if 'full' in self.input_root:
                            #     np.savetxt(osp.join(self.output_root, 'results',
                            #                         f'rgb_guided_dvfms_from_2d_with_max_mag_removal_visualize_0_5_tile_{self.config.tile_id}.txt'),
                            #                dvfs_3d_from_2d_vector_mag.cpu())
                            # else:
                            #     np.savetxt(osp.join(self.output_root, 'results',
                            #                         'rgb_guided_dvfms_from_2d_with_max_mag_removal_visualize_0_5.txt'),
                            #                dvfs_3d_from_2d_vector_mag.cpu())

                        # remove indices that has more than max. mag. these indices will be further updated until the final output
                        idx_valid_src = idx_valid_src[mask_within_max_mag]
                        idx_valid_tgt = idx_valid_tgt[mask_within_max_mag]

                        self.data_interim.idx_valid_src_from_2d = idx_valid_src
                        self.data_interim.idx_valid_tgt_from_2d = idx_valid_tgt

                        # keep the indices of unmatched pts as -1
                        corres_3d_from_2d_idx = - torch.ones(self.data_input_3d.idx_initial_src.shape[0], 2).to(torch.int64).to(
                            self.device)
                        corres_3d_from_2d_idx[:, 0] = torch.arange(self.data_input_3d.idx_initial_src.shape[0])
                        corres_3d_from_2d_idx[idx_valid_src, 1] = idx_valid_tgt

                        # self.data_interim.corres_3d_from_2d_idx = corres_3d_from_2d_idx

                        # clear not used variables, related to self
                        del self.data_interim.proj_uv_src, self.data_interim.proj_uv_tgt, (
                            self.data_interim.idx_valid_src), self.data_interim.idx_valid_tgt, (
                            self.data_interim.idx_valid_src_from_2d), self.data_interim.idx_valid_tgt_from_2d

                    else:
                        self.logging.info(f'No initial pixel matches sub_img [{num_src}, {num_tgt}] for the current tile')

                        # no valid 3d matches from 2d, even in the very initial image matching step, before outlier removal
                        corres_3d_from_2d_idx = - torch.ones(self.data_input_3d.idx_initial_src.shape[0], 2).to(torch.int64).to(
                            self.device)
                        corres_3d_from_2d_idx[:, 0] = torch.arange(self.data_input_3d.idx_initial_src.shape[0])

                        # self.data_interim.corres_3d_from_2d_idx = corres_3d_from_2d_idx
                        a = 0

                    corres_3d_from_2d_idx_list.append(corres_3d_from_2d_idx)

            # slow version
            # for i in range(corres_3d_from_2d_idx_list[0].size(0)):
            #     if corres_3d_from_2d_idx_list[0][i, 1] == -1:  # Check if the second column of the main tensor is -1
            #         # Iterate through the other tensors in the list
            #         for tensor in corres_3d_from_2d_idx_list[1:]:
            #             if tensor[i, 1] != -1:  # If the second column is not -1 in the backup tensor
            #                 corres_3d_from_2d_idx_list[0][i, 1] = tensor[i, 1]  # Update the main tensor with the non -1 value
            #                 break  # Once we find a valid match, break out of the inner loop

            # switch the order
            # if self.method.num_sub_img == 2:
            #     corres_3d_from_2d_idx_list = [corres_3d_from_2d_idx_list[0], corres_3d_from_2d_idx_list[3], corres_3d_from_2d_idx_list[1], corres_3d_from_2d_idx_list[2]]
            # else:
            #     desired_order = [0, 4, 8]
            #     remaining_indices = [i for i in range(len(corres_3d_from_2d_idx_list)) if i not in desired_order]
            #
            #     # Create the reordered list
            #     reordered_list = (
            #             [corres_3d_from_2d_idx_list[i] for i in desired_order] +
            #             [corres_3d_from_2d_idx_list[i] for i in remaining_indices]
            #     )
            #     corres_3d_from_2d_idx_list = reordered_list

            # Main tensor
            main_tensor = corres_3d_from_2d_idx_list[0]
            # Create a mask where the second column is -1
            mask = main_tensor[:, 1] == -1
            # Iterate over backup tensors (from tensor_list[1:] onwards)
            for tensor in corres_3d_from_2d_idx_list[1:]:
                # For each backup tensor, find where its second column is not -1
                valid_mask = tensor[:, 1] != -1
                # Use the mask to update the main tensor with valid entries from the backup tensor
                # Only update where the main tensor's second column is -1
                mask_both = mask & valid_mask
                main_tensor[mask_both, 1] = tensor[mask_both, 1]

            self.data_interim.corres_3d_from_2d_idx = main_tensor
            # avoid unknown errors, only the most matched img and pose are used, e.g., for compute img patches
            self.data_input_2d.src_image = self.data_input_2d.src_image_list[0]
            self.data_input_2d.tgt_image = self.data_input_2d.tgt_image_list[0]
            self.data_input_2d.src_extrinsic = self.data_input_2d.src_extrinsic_list[0]
            self.data_input_2d.tgt_extrinsic = self.data_input_2d.tgt_extrinsic_list[0]

            a = 0

        idx_valid = torch.where(self.data_interim.corres_3d_from_2d_idx[:, 1] > -1)[0]

        self.data_output.idx_valid_src = self.data_interim.corres_3d_from_2d_idx[idx_valid, 0]
        self.data_output.idx_valid_tgt = self.data_interim.corres_3d_from_2d_idx[idx_valid, 1]

        corres_3d_src = self.data_input_3d.src_pts[self.data_output.idx_valid_src, :]
        corres_3d_tgt = self.data_input_3d.tgt_pts[self.data_output.idx_valid_tgt, :]
        corres_magnitude = torch.linalg.norm(corres_3d_src - corres_3d_tgt, axis=1).to(self.device)
        self.data_output.corres_3d = torch.cat([corres_3d_src, corres_3d_tgt], dim=1)
        self.data_output.corres_3d_magnitude = corres_magnitude[:, None]

        # self.pcd2img_with_different_types()

        self.implement_segmentation()

        self.local_rigid_refinement()

        self._save_process_dvf()

        torch.cuda.empty_cache()
        gc.collect()

    def _save_process_dvf(self):
        dir_exist(osp.join(self.output_root, 'results'))

        # pure output
        # np.savetxt(osp.join(self.output_root, 'results', f'src_id.txt'),
        #            self.data_output.idx_valid_src.cpu(), delimiter=' ', fmt='%.0f')
        # np.savetxt(osp.join(self.output_root, 'results', f'tgt_id.txt'),
        #            self.data_output.idx_valid_tgt.cpu(), delimiter=' ', fmt='%.0f')

        # TODO: save the magnitude, for debugging
        # np.savetxt(osp.join(self.output_root, 'results', f'rgb_guided_dvfs_wo_refinement_{self.config.tile_id}.txt'),
        #            self.data_output.corres_3d.cpu(), delimiter=' ', fmt='%.6f')
        np.savetxt(osp.join(self.output_root, 'results', f'rgb_guided_wo_refinement_dvfms_tile_{self.config.tile_id}.txt'),
                   np.hstack((self.data_output.corres_3d[:, :3].cpu(), self.data_output.corres_3d_magnitude.cpu())),
                   delimiter=' ', fmt='%.6f')

        # # save ids, also for wo_refinement
        # np.savetxt(osp.join(self.output_root, 'results', f'src_id_refine.txt'),
        #            self.data_output.idx_valid_src_refine.cpu(), delimiter=' ', fmt='%.0f')
        # np.savetxt(osp.join(self.output_root, 'results', f'tgt_id_refine.txt'),
        #            self.data_output.idx_valid_tgt_refine.cpu(), delimiter=' ', fmt='%.0f')
        # np.savetxt(osp.join(self.output_root, 'results', f'pure_2d_displace_vector.txt'),
        #                self.data_output.corres_3d_refine.cpu(), delimiter=' ', fmt='%.6f')
        # np.savetxt(osp.join(self.output_root, 'results', f'pure_2d_displace_magnitude.txt'),
        #            np.hstack((self.data_output.corres_3d_refine[:, :3].cpu(),
        #                       self.data_output.corres_3d_magnitude_refine.cpu())),
        #            delimiter=' ', fmt='%.6f')

        if self.method.icp_refine:
            np.savetxt(osp.join(self.output_root, 'results', f'rgb_guided_w_refinement_dvfs_src2tgt_tile_{self.config.tile_id}.txt'),
                       self.data_output.corres_3d_refine_apply_icp.cpu(), delimiter=' ', fmt='%.6f')
            np.savetxt(
                osp.join(self.output_root, 'results', f'rgb_guided_w_refinement_dvfms_src2tgt_tile_{self.config.tile_id}.txt'),
                np.hstack((self.data_output.corres_3d_refine_apply_icp[:, :3].cpu(),
                           self.data_output.corres_3d_magnitude_refine_apply_icp.cpu())),
                delimiter=' ', fmt='%.6f')

            if self.data_output.corres_3d_magnitude_refine_apply_icp.shape[0] > 2:
                corres_3d_magnitude_refine_apply_icp_visualize = copy.deepcopy(self.data_output.corres_3d_magnitude_refine_apply_icp)

                # for better visualization in CloudCompare
                # only for visualization, not for real computation; can be commented for faster computation
                if self.data.dataset == 'rockfall_simulator':
                    max_mag_visualize = 0.06
                elif self.data.dataset == 'brienz':
                    max_mag_visualize = 5
                elif self.data.dataset == 'mattertal':
                    max_mag_visualize = 10
                else:
                    max_mag_visualize = 10

                corres_3d_magnitude_refine_apply_icp_visualize[0] = 0
                corres_3d_magnitude_refine_apply_icp_visualize[1] = max_mag_visualize
                np.savetxt(
                    osp.join(self.output_root, 'results', f'rgb_guided_w_refinement_dvfms_src2tgt_visualize_tile_{self.config.tile_id}.txt'),
                    np.hstack((self.data_output.corres_3d_refine_apply_icp[:, :3].cpu(),
                               corres_3d_magnitude_refine_apply_icp_visualize.cpu())),
                    delimiter=' ', fmt='%.6f')

    def _initialize(self):
        # self.data_input, self.data_interim, self.data_output = edict(dict()), edict(dict()), edict(dict())
        # self.temp_dict = edict(dict())

        # data_interim only saves important intermediate data, e.g., debug data
        self.data_input_2d, self.data_input_3d, self.data_interim, self.data_output = (
            edict(dict()), edict(dict()), edict(dict()), edict(dict()))

        self.backbones = edict(dict())

        self.input_root = self.config.path_name.input_root
        self.output_root = self.config.path_name.output_root
        self.data = self.config.data
        self.method = self.config.method

    # def _pcd_tiling(self):
    #     a = 0

    def lift_2d_to_3d_with_interpolation(self, pixel_matches, batch_size=10000):
        depth_map_src = torch.from_numpy(self.data_interim.depth_map_src).to(torch.float32).to(self.device)
        depth_map_tgt = torch.from_numpy(self.data_interim.depth_map_tgt).to(torch.float32).to(self.device)
        camera_intrinsic = self.data_input_2d.camera_intrinsic

        # intrinsic_inv = torch.inverse(camera_intrinsic)
        points_3d_matches = []
        point_3d_matches_mask = []

        for start in range(0, pixel_matches.shape[0], batch_size):
            end = min(start + batch_size, pixel_matches.shape[0])
            batch = pixel_matches[start:end]

            # Separate source and target pixel coordinates in the batch
            u_src, v_src, u_tgt, v_tgt = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
            # should flip over u and v, strange but valid for Rigel scans, not RTC360 scans
            # v_src, u_src, v_tgt, u_tgt = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]

            # Interpolated depths for batch
            depth_src, mask_valid_depth_src = bilinear_interpolate(depth_map_src, u_src, v_src)
            depth_tgt, mask_valid_depth_tgt = bilinear_interpolate(depth_map_tgt, u_tgt, v_tgt)

            # Homogeneous 2D coordinates for source and target in the batch
            uv_src_homogeneous = torch.stack([u_src, v_src, torch.ones_like(u_src)], dim=1)
            uv_tgt_homogeneous = torch.stack([u_tgt, v_tgt, torch.ones_like(u_tgt)], dim=1)

            # Lift to 3D for batch source and target pixels
            uv_src_homogeneous = uv_src_homogeneous[:, [1, 0, 2]]
            uv_tgt_homogeneous = uv_tgt_homogeneous[:, [1, 0, 2]]

            if self.data.dataset != 'rockfall_simulator':
                uv_src_homogeneous[:, 1] = depth_map_src.shape[0] - uv_src_homogeneous[:, 1]
                uv_tgt_homogeneous[:, 1] = depth_map_tgt.shape[0] - uv_tgt_homogeneous[:, 1]
            uv_src_homogeneous = uv_src_homogeneous * depth_src
            uv_tgt_homogeneous = uv_tgt_homogeneous * depth_tgt

            intrinsic_inv = torch.linalg.inv(camera_intrinsic)

            src_3d_camera = intrinsic_inv @ uv_src_homogeneous.T
            tgt_3d_camera = intrinsic_inv @ uv_tgt_homogeneous.T

            rotation_matrix_src = self.data_input_2d.src_extrinsic[:3, :3]
            translation_vector_src = self.data_input_2d.src_extrinsic[:3, 3]
            rotation_inv_src = rotation_matrix_src.T
            translation_inv_src = -rotation_inv_src @ translation_vector_src
            src_3d_world = rotation_inv_src @ src_3d_camera + translation_inv_src[:, None]
            src_3d_world = src_3d_world.T
            src_3d_world_valid = src_3d_world[mask_valid_depth_src]

            rotation_matrix_tgt = self.data_input_2d.tgt_extrinsic[:3, :3]
            translation_vector_tgt = self.data_input_2d.tgt_extrinsic[:3, 3]
            rotation_inv_tgt = rotation_matrix_tgt.T
            translation_inv_tgt = -rotation_inv_tgt @ translation_vector_tgt
            tgt_3d_world = rotation_inv_tgt @ tgt_3d_camera + translation_inv_tgt[:, None]
            tgt_3d_world = tgt_3d_world.T
            tgt_3d_world_valid = tgt_3d_world[mask_valid_depth_tgt]

            # Concatenate and add to results
            points_3d_matches.append(torch.cat([src_3d_world, tgt_3d_world], dim=1))

            mask = mask_valid_depth_src & mask_valid_depth_tgt
            # print(torch.sum(mask))
            point_3d_matches_mask.append(mask)

        return torch.cat(points_3d_matches, dim=0), torch.cat(point_3d_matches_mask)

    def _get_the_most_matched_idx(self, transformation, pts_coord):
        # src part, use the correct format of pose to project, and find count the number of points
        transformation_src = torch.stack(transformation, dim=0)
        src_pts_expanded = pts_coord.unsqueeze(0).expand(transformation_src.shape[0], -1, -1)

        rotations = transformation_src[:, :3, :3]  # Shape: (100, 3, 3)
        translations = transformation_src[:, :3, 3]  # Shape: (100, 3)
        # Compute transformed points in camera coordinates
        src_pts_camera = torch.bmm(rotations, src_pts_expanded.transpose(1, 2)) + translations.unsqueeze(
            2)  # Shape: (100, 3, N)

        # Expand the intrinsic matrix to match the number of transformations
        camera_intrinsic_expanded = self.data_input_2d.camera_intrinsic.expand(transformation_src.shape[0], -1,
                                                                               -1)  # Shape: (100, 3, 3)

        # Project to 2D pixel coordinates
        src_pixel = torch.bmm(camera_intrinsic_expanded, src_pts_camera)  # Shape: (100, 3, N)
        src_pixel_normalized = src_pixel / src_pixel[:, 2:3, :]  # Normalize by the third row

        # Extract u, v pixel coordinates and adjust for image boundaries
        src_pts_u = src_pixel_normalized[:, 0, :].cpu().numpy()
        src_pts_v = (self.method.image_size[0] - src_pixel_normalized[:, 1, :]).cpu().numpy()

        # Generate masks for points within image bounds for each transformation
        mask_valid_pts_uv_src = ((src_pts_u > 0) & (src_pts_u < self.method.image_size[1]) &
                                 (src_pts_v > 0) & (src_pts_v < self.method.image_size[0]))

        a = np.sum(mask_valid_pts_uv_src, axis=1)
        return a

    def _find_the_most_matched_image(self):
        # fine the most matched image and image extrinsic

        # get all extrinsics of src and tgt images
        camera_extrinsic_path = osp.join(self.input_root, 'image', 'transformations')

        # Initialize lists to store image names and camera poses
        img_names_src, img_names_tgt = [], []
        transformation_src, transformation_tgt = [], []

        # get epoch id
        src_id = self.data.src_pcd.split('_')[0]
        tgt_id = self.data.tgt_pcd.split('_')[0]

        # Read the file
        with open(osp.join(camera_extrinsic_path, 'Images_used.txt'), 'r') as file:
            while True:
                # Read the image name
                image_name = file.readline().strip()
                if not image_name:  # Break if the end of the file is reached
                    break
                # if image_name == '220706_133229_Image0_46ms_15-6.jpg':
                #     a = 0
                # image_names.append(image_name)

                # Parse translation and rotation
                translation = np.fromstring(file.readline().strip(), sep=' ')
                # considering an offset between image and point cloud for tgt epoch
                # if '220706' in image_name:
                #     translation[1] -= 2
                rotation = np.array([np.fromstring(file.readline().strip(), sep=' ') for _ in range(3)])

                # Construct the 4x4 transformation matrix
                transformation_matrix = torch.eye(4, dtype=torch.float32).to(self.device)
                transformation_matrix[:3, :3] = torch.from_numpy(rotation).to(self.device)
                transformation_matrix[:3, 3] = torch.from_numpy(translation).to(self.device)

                transformation_matrix = torch.linalg.inv(transformation_matrix)

                # Append to source or target lists based on the year in the image name
                # src: Brienz or Mattertal
                ## TODO: should work for other epochs or other datasets
                if src_id in image_name:
                    img_names_src.append(image_name)
                    transformation_src.append(transformation_matrix)
                # tgt: Brienz or Mattertal
                elif tgt_id in image_name:
                    img_names_tgt.append(image_name)
                    transformation_tgt.append(transformation_matrix)
                # transformation_matrices.append(transformation_matrix)

                a = 0

        mask_sum_src = self._get_the_most_matched_idx(transformation_src, self.data_input_3d.src_pts)
        mask_sum_tgt = self._get_the_most_matched_idx(transformation_tgt, self.data_input_3d.tgt_pts)

        if self.method.save_img_matching_visualization:
            # save the matched images for debugging, sanity check
            save_path = osp.join(self.output_root, self.config.path_name.img_matching_result_dir, f'tile_{self.config.tile_id}')

            # if self.method.img_matching_type == 'ELoFTR':
            #     save_path = osp.join(self.output_root, 'img_matching_results_EloFTR', f'tile_{self.config.tile_id}')
            # elif self.method.img_matching_type == 'RoMA':
            #     save_path = osp.join(self.output_root, 'img_matching_results_RoMA', f'tile_{self.config.tile_id}')
            dir_exist(save_path)

        if self.method.num_sub_img == 1:
            img_name_src = img_names_src[np.argmax(mask_sum_src)]
            img_name_tgt = img_names_tgt[np.argmax(mask_sum_tgt)]

            src_image_path = osp.join(self.input_root, 'image', 'raw_images', 'src_images', img_name_src)
            tgt_image_path = osp.join(self.input_root, 'image', 'raw_images', 'tgt_images', img_name_tgt)

            shutil.copy(src_image_path, osp.join(save_path, img_name_src))
            shutil.copy(tgt_image_path, osp.join(save_path, img_name_tgt))

            # Read the images
            self.data_input_2d.src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
            self.data_input_2d.tgt_image = cv2.imread(tgt_image_path, cv2.IMREAD_COLOR)

            # Extract the corresponding transformation (extrinsics)
            self.data_input_2d.src_extrinsic = transformation_src[np.argmax(mask_sum_src)]
            self.data_input_2d.tgt_extrinsic = transformation_tgt[np.argmax(mask_sum_tgt)]
        else:
            # Get indices of the top n maximum values
            top_n_indices_src = np.argsort(mask_sum_src)[-self.method.num_sub_img:][::-1]
            top_n_indices_tgt = np.argsort(mask_sum_tgt)[-self.method.num_sub_img:][::-1]

            # Select the corresponding image names
            img_names_src_top_n = np.array(img_names_src)[top_n_indices_src]
            img_names_tgt_top_n = np.array(img_names_tgt)[top_n_indices_tgt]

            # Create a list of image paths for the top n images
            src_image_paths = [osp.join(self.input_root, 'image', 'raw_images', 'src_images', img_name) for img_name in
                               img_names_src_top_n]
            tgt_image_paths = [osp.join(self.input_root, 'image', 'raw_images', 'tgt_images', img_name) for img_name in
                               img_names_tgt_top_n]

            # Loop over the top n images and copy them
            for src_img_path, tgt_img_path, src_name, tgt_name in zip(src_image_paths, tgt_image_paths,
                                                                      img_names_src_top_n, img_names_tgt_top_n):
                shutil.copy(src_img_path, osp.join(save_path, src_name))
                shutil.copy(tgt_img_path, osp.join(save_path, tgt_name))

            # Load images and extrinsics for the top n images
            src_images = [
                cv2.imread(osp.join(self.input_root, 'image', 'raw_images', 'src_images', img_name), cv2.IMREAD_COLOR)
                for img_name in img_names_src_top_n]
            tgt_images = [
                cv2.imread(osp.join(self.input_root, 'image', 'raw_images', 'tgt_images', img_name), cv2.IMREAD_COLOR)
                for img_name in img_names_tgt_top_n]

            src_extrinsics = [transformation_src[idx] for idx in top_n_indices_src]
            tgt_extrinsics = [transformation_tgt[idx] for idx in top_n_indices_tgt]

            # Store them in the data_input_2d object (assuming it can handle lists or sequences of images)
            self.data_input_2d.src_image_list = src_images
            self.data_input_2d.tgt_image_list = tgt_images
            self.data_input_2d.src_extrinsic_list = src_extrinsics
            self.data_input_2d.tgt_extrinsic_list = tgt_extrinsics

        # self.data_input_2d.src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
        # self.data_input_2d.tgt_image = cv2.imread(tgt_image_path, cv2.IMREAD_COLOR)
        # self.data_input_2d.src_extrinsic = transformation_src[np.argmax(mask_sum_src)]
        # self.data_input_2d.tgt_extrinsic = transformation_tgt[np.argmax(mask_sum_tgt)]

        a = 0
        a = 0

    def _read_data(self):
        # full pcd for clustering, subsampled pcd for feature computation
        # 3d
        if self.data.multiple_case:
            self.src_pcd_path = self.config.src_tile_overlap_path
            self.tgt_pcd_path = self.config.tgt_tile_overlap_path
        else:
            self.src_pcd_path = osp.join(self.input_root, 'raw_pcd', self.data.src_pcd)
            self.tgt_pcd_path = osp.join(self.input_root, 'raw_pcd', self.data.tgt_pcd)
        self.data_input_3d.src_pcd = o3d.io.read_point_cloud(self.src_pcd_path)
        self.data_input_3d.tgt_pcd = o3d.io.read_point_cloud(self.tgt_pcd_path)

        self.data_input_3d.src_pts, self.data_input_3d.src_clr = pcd2tensor(self.data_input_3d.src_pcd,
                                                                            return_colors=True)
        self.data_input_3d.tgt_pts, self.data_input_3d.tgt_clr = pcd2tensor(self.data_input_3d.tgt_pcd,
                                                                            return_colors=True)

        # record point indices of input src pcd, tgt pcd
        self.data_input_3d.idx_initial_src = torch.arange(self.data_input_3d.src_pts.shape[0]).to(self.device)
        self.data_input_3d.idx_initial_tgt = torch.arange(self.data_input_3d.tgt_pts.shape[0]).to(self.device)

        # 2d
        camera_intrinsic = np.loadtxt(osp.join(self.input_root, 'image', 'camera_intrinsic.txt'), delimiter=' ')
        self.data_input_2d.camera_intrinsic = array2tensor(camera_intrinsic).to(torch.float32).to(self.device)

        # deal with full datasets, multiple cases
        if self.data.multiple_case:
            self._find_the_most_matched_image()

            # output
            # self.data_input_2d.src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
            # self.data_input_2d.tgt_image = cv2.imread(tgt_image_path, cv2.IMREAD_COLOR)
            # self.data_input_2d.src_extrinsic = array2tensor(src_extrinsic).to(torch.float32).to(
            #     self.device)
            # self.data_input_2d.tgt_extrinsic = array2tensor(tgt_extrinsic).to(torch.float32).to(
            #     self.device)
        # deal with one ROIs and one image case
        else:
            src_image_path = osp.join(self.input_root, 'image', 'raw_images', self.data.src_image)
            tgt_image_path = osp.join(self.input_root, 'image', 'raw_images', self.data.tgt_image)
            self.data_input_2d.src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
            self.data_input_2d.tgt_image = cv2.imread(tgt_image_path, cv2.IMREAD_COLOR)

            src_extrinsic = np.eye(4)
            tgt_extrinsic = np.eye(4)
            if self.data.dataset == 'rockfall_simulator':
                camera_extrinsic_path = osp.join(self.input_root, 'image', 'transformations')
                src_extrinsic_quaternion = np.loadtxt(os.path.join(camera_extrinsic_path, 'camera_extrinsic_epoch_1.txt'))
                tgt_extrinsic_quaternion = np.loadtxt(os.path.join(camera_extrinsic_path, 'camera_extrinsic_epoch_2.txt'))

                src_extrinsic[:3, :3] = quaternion_to_rotation_matrix(src_extrinsic_quaternion[:4])
                src_extrinsic[:3, 3] = src_extrinsic_quaternion[4:7]
                tgt_extrinsic[:3, :3] = quaternion_to_rotation_matrix(tgt_extrinsic_quaternion[:4])
                tgt_extrinsic[:3, 3] = tgt_extrinsic_quaternion[4:7]

                src_extrinsic = np.linalg.inv(src_extrinsic)
                tgt_extrinsic = np.linalg.inv(tgt_extrinsic)

            elif self.data.dataset == 'brienz':
                camera_extrinsic_path = osp.join(self.input_root, 'image', 'transformations')
                transform_sop_src = np.loadtxt(os.path.join(camera_extrinsic_path, 'sop_transformation_200221.txt'),
                                               delimiter=' ')
                transform_cop_src = np.loadtxt(os.path.join(camera_extrinsic_path, 'cop_transformation_10_1_200221.txt'),
                                               delimiter=' ')
                transform_sop_tgt = np.loadtxt(os.path.join(camera_extrinsic_path, 'sop_transformation_201130.txt'),
                                               delimiter=' ')
                transform_cop_tgt = np.loadtxt(os.path.join(camera_extrinsic_path, 'cop_transformation_9_1_201130.txt'),
                                               delimiter=' ')
                transform_mounting = np.loadtxt(os.path.join(camera_extrinsic_path, 'mounting_transformation.txt'),
                                                delimiter=',')

                if self.config.coord_type == 'PRCS':
                    transform_ext_src = (transform_mounting @ np.linalg.inv(transform_cop_src) @
                                         np.linalg.inv(transform_sop_src))
                    transform_ext_tgt = (transform_mounting @ np.linalg.inv(transform_cop_tgt) @
                                         np.linalg.inv(transform_sop_tgt))
                elif self.config.coord_type == 'SOCS':
                    transform_ext_src = transform_cop_src @ np.linalg.inv(transform_mounting)
                    transform_ext_tgt = transform_cop_tgt @ np.linalg.inv(transform_mounting)
                else:
                    raise NotImplementedError

                src_extrinsic = transform_ext_src
                tgt_extrinsic = transform_ext_tgt

            elif self.data.dataset in ['brienz_tls', 'mattertal']:
                camera_extrinsic_path = osp.join(self.input_root, 'image', 'transformations')

                # camera extrinsic
                src_transform_total = np.loadtxt(osp.join(camera_extrinsic_path, f'{self.data.src_pose}'),
                                                 delimiter=' ')
                tgt_transform_total = np.loadtxt(osp.join(camera_extrinsic_path, f'{self.data.tgt_pose}'),
                                                 delimiter=' ')
                src_extrinsic = np.linalg.inv(src_transform_total)
                tgt_extrinsic = np.linalg.inv(tgt_transform_total)

            self.data_input_2d.src_extrinsic = array2tensor(src_extrinsic).to(torch.float32).to(
                self.device)
            self.data_input_2d.tgt_extrinsic = array2tensor(tgt_extrinsic).to(torch.float32).to(
                self.device)

    @property
    def _feat_path(self) -> str:
        if self.data.multiple_case:
            return osp.join(self.output_root, 'features', f'features_tile_{self.config.tile_id}.npz')
        else:
            return osp.join(self.output_root, 'features', f'features.npz')

    def _feat_desc_nn(self, input):
        if self.method.point_feat_compute:
            return self.config.feat_desc_nn(input)

    def _voxel_subsampling(self):
        self.data_interim.src_pcd_sub = self.data_input_3d.src_pcd.voxel_down_sample(voxel_size=self.method.voxel_size)
        self.data_interim.tgt_pcd_sub = self.data_input_3d.tgt_pcd.voxel_down_sample(voxel_size=self.method.voxel_size)
        self.data_interim.src_pts_sub, self.data_interim.src_clr_sub = (
            pcd2tensor(self.data_interim.src_pcd_sub, return_colors=True))
        self.data_interim.tgt_pts_sub, self.data_interim.tgt_clr_sub = (
            pcd2tensor(self.data_interim.tgt_pcd_sub, return_colors=True))

        # get the indices of voxels in original pts
        s_tree = cKDTree(self.data_input_3d.src_pts.cpu())
        distance, src_voxel2pts = s_tree.query(self.data_interim.src_pts_sub.cpu(), k=1)

        t_tree = cKDTree(self.data_input_3d.tgt_pts.cpu())
        distance, tgt_voxel2pts = t_tree.query(self.data_interim.tgt_pts_sub.cpu(), k=1)

        self.data_interim.idx_voxel2pts_src = src_voxel2pts
        self.data_interim.idx_voxel2pts_tgt = tgt_voxel2pts

        # get the indices of pts to voxels
        idx_pts2voxel_src = - torch.ones(len(self.data_input_3d.idx_initial_src),).to(torch.int64).to(
            self.device)
        idx_pts2voxel_tgt = - torch.ones(len(self.data_input_3d.idx_initial_tgt),).to(torch.int64).to(
            self.device)

        idx_pts2voxel_src[src_voxel2pts] = torch.arange(len(src_voxel2pts)).to(self.device)
        idx_pts2voxel_tgt[tgt_voxel2pts] = torch.arange(len(tgt_voxel2pts)).to(self.device)

        self.data_interim.idx_pts2voxel_src = idx_pts2voxel_src
        self.data_interim.idx_pts2voxel_tgt = idx_pts2voxel_tgt

    def _get_2d_matches_online(self):
        if not hasattr(self.backbones, 'model_img_matching'):
            # initialize the model, but only happens to the first time. Should apply to other backbones as well
            if self.method.img_matching_type == 'ELoFTR':
                # reference: [Efficient LoFTR]
                from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
                from EfficientLoFTR.src.utils.plotting import make_matching_figure

                # You can choose model type in ['full', 'opt']
                model_type = 'full'  # 'full' for best quality, 'opt' for best efficiency
                # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
                precision = 'fp32'  # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

                if model_type == 'full':
                    _default_cfg = deepcopy(full_default_cfg)
                elif model_type == 'opt':
                    _default_cfg = deepcopy(opt_default_cfg)

                if precision == 'mp':
                    _default_cfg['mp'] = True
                elif precision == 'fp16':
                    _default_cfg['half'] = True

                img_matcher = LoFTR(config=_default_cfg)
                img_matcher.load_state_dict(torch.load("./weights/eloftr_outdoor.ckpt")['state_dict'])
                img_matcher = reparameter(img_matcher)  # no reparameterization will lead to low performance

                if precision == 'fp16':
                    img_matcher = img_matcher.half()

                img_matcher = img_matcher.eval().cuda()

                self.backbones.img_matcher = img_matcher
            elif self.method.img_matching_type == 'RoMA':
                from RoMa.romatch import roma_outdoor
                from EfficientLoFTR.src.utils.plotting import make_matching_figure

                roma_model = roma_outdoor(device=self.device)
                self.backbones.img_matcher = roma_model

            else:
                NotImplementedError
            a = 0

        ##############
        # cropping before matching
        src_img = Image.fromarray(self.data_input_2d.src_image)
        tgt_img = Image.fromarray(self.data_input_2d.tgt_image)

        # Calculate the starting coordinates for cropping
        start_x = 0
        while start_x + self.method.crop_size[0] <= src_img.width:
            start_y = 0
            while start_y + self.method.crop_size[1] <= src_img.height:
                img_save_path = osp.join(f'tile_{self.config.tile_id}')
                crop_and_save(src_img, osp.join(img_save_path, 'src_images'), start_x, start_y, self.method.crop_size, self.output_root)
                crop_and_save(tgt_img, osp.join(img_save_path, 'tgt_images'), start_x, start_y, self.method.crop_size, self.output_root)

                # Move the cropping window
                start_y += self.method.crop_size[1] - self.method.overlap_size[1]
            start_x += self.method.crop_size[0] - self.method.overlap_size[0]
        ##############
        src_img_path = osp.join(self.output_root, 'cropped_images', img_save_path, 'src_images')
        tgt_img_path = osp.join(self.output_root, 'cropped_images', img_save_path, 'tgt_images')

        src_img_list = os.listdir(src_img_path)
        tgt_img_list = os.listdir(tgt_img_path)
        # for _, i in enumerate(tqdm(src_img_list, position=0, leave=True)):
        for _, i in enumerate(src_img_list):
            src_img_path_temp = osp.join(src_img_path, i)
            src_name = osp.splitext(i)[0]
            src_base_u, src_base_v = int(src_name.split('_')[0]), int(src_name.split('_')[1])

            for _, j in enumerate(tgt_img_list):
                tgt_img_path_temp = osp.join(tgt_img_path, j)
                tgt_name = osp.splitext(j)[0]
                tgt_base_u, tgt_base_v = int(tgt_name.split('_')[0]), int(tgt_name.split('_')[1])

                # #################
                # # TODO:
                # if src_base_u != 0 or src_base_v != 0 or tgt_base_u != 0 or tgt_base_v != 640:
                #     continue
                # #################

                if self.method.img_matching_type == 'ELoFTR':
                    img0_raw = cv2.imread(src_img_path_temp, cv2.IMREAD_GRAYSCALE)
                    img1_raw = cv2.imread(tgt_img_path_temp, cv2.IMREAD_GRAYSCALE)
                    img0_raw = cv2.resize(img0_raw, (
                    img0_raw.shape[1] // 32 * 32, img0_raw.shape[0] // 32 * 32))  # input size shuold be divisible by 32
                    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1] // 32 * 32, img1_raw.shape[0] // 32 * 32))

                    if precision == 'fp16':
                        img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
                        img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
                    else:
                        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
                        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
                    batch = {'image0': img0, 'image1': img1}

                    # Inference with EfficientLoFTR and get prediction
                    with torch.no_grad():
                        if precision == 'mp':
                            with torch.autocast(enabled=True, device_type='cuda'):
                                self.backbones.img_matcher(batch)
                        else:
                            self.backbones.img_matcher(batch)
                        mkpts0 = batch['mkpts0_f'].cpu().numpy()
                        mkpts1 = batch['mkpts1_f'].cpu().numpy()
                        mconf = batch['mconf'].cpu().numpy()

                        # #################
                        # TODO: select only 15 matches for debugging
                        # import random
                        # indices = list(range(len(mconf)))
                        # random.shuffle(indices)
                        # mkpts0 = mkpts0[indices[:10]]
                        # mkpts1 = mkpts1[indices[:10]]
                        # mconf = mconf[indices[:10]]
                        #
                        # a = 0
                        # #################

                    # Draw
                    if model_type == 'opt':
                        print(mconf.max())
                        mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

                    color = cm.jet(mconf)
                    text = [
                        'Efficient LoFTR',
                        'Matches: {}'.format(len(mkpts0)),
                    ]
                elif self.method.img_matching_type == 'RoMA':
                    W_A, H_A = Image.open(src_img_path_temp).size
                    W_B, H_B = Image.open(tgt_img_path_temp).size

                    # Match
                    warp, certainty = roma_model.match(src_img_path_temp, tgt_img_path_temp, device=self.device)
                    # Sample matches for estimation
                    matches, certainty = roma_model.sample(warp, certainty, num=10000)

                    if matches.ndim == 1:
                        # self.logging.info('No initial pixel matches for the current tile')
                        continue
                    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

                    mkpts0 = np.asarray(kpts1.cpu())
                    mkpts1 = np.asarray(kpts2.cpu())
                    mconf = np.asarray(certainty.cpu())

                    # #################
                    # TODO: select only 15 matches for debugging
                    # import random
                    # indices = list(range(len(mconf)))
                    # random.shuffle(indices)
                    # mkpts0 = mkpts0[indices[:10]]
                    # mkpts1 = mkpts1[indices[:10]]
                    # mconf = mconf[indices[:10]]
                    #
                    # a = 0
                    # #################

                    text = [
                        'RoMA',
                        'Matches: {}'.format(len(kpts1)),
                    ]

                # convert cropped pixel to a single image coord.
                src_matches_global = mkpts0 + [src_base_u, src_base_v]
                tgt_matches_global = mkpts1 + [tgt_base_u, tgt_base_v]
                matches = np.concatenate((src_matches_global, tgt_matches_global, mconf[:, None]), axis=1)
                # src pixel coordinates, tgt pixel coordinates, confidence score

                corres_2d_dir = osp.join(self.output_root, self.config.path_name.img_matching_result_dir, img_save_path, 'corres_2d')

                # if self.method.img_matching_type == 'ELoFTR':
                #     corres_2d_dir = osp.join(self.output_root, 'img_matching_results_EloFTR', img_save_path, 'corres_2d')
                # elif self.method.img_matching_type == 'RoMA':
                #     corres_2d_dir = osp.join(self.output_root, 'img_matching_results_RoMA', img_save_path, 'corres_2d')
                dir_exist(corres_2d_dir)

                save_matches = osp.join(corres_2d_dir, f'src_{src_name}_tgt_{tgt_name}.txt')
                np.savetxt(save_matches, matches)

                if self.method.save_img_matching_visualization:
                    visualization_dir = corres_2d_dir.replace('corres_2d', 'visualization')
                    dir_exist(visualization_dir)

                    if self.method.img_matching_type == 'RoMA':
                        img0_raw = cv2.imread(src_img_path_temp, cv2.IMREAD_GRAYSCALE)
                        img1_raw = cv2.imread(tgt_img_path_temp, cv2.IMREAD_GRAYSCALE)

                    # didnt save the visualization result due to the runtime consideration
                    save_img_path = osp.join(visualization_dir, f'src_{src_name}_tgt_{tgt_name}.jpg')
                    make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, path=save_img_path, color=None, dpi=30, text=text, save_individual=True)

                    # num_matches.append(len(mkpts0))
                    # save_img = osp.join(visualization_dir, f'src_{src_name}_tgt_{tgt_name}.jpg')
                    # plt.savefig(str(save_img), bbox_inches='tight', pad_inches=0)
                    # plt.close()


                    a = 0

        a = 0
    # major functions
    ##################################

    def pcd2img_with_different_types(self):
        """
        Converts 3D point cloud to 2D image, generates a depth map with RGB image dimensions,
        and finds corresponding correspondences.
        """
        # Project source points to camera space and then to image plane
        src_pts_camera = (self.data_input_2d.src_extrinsic[:3, :3] @ self.data_input_3d.src_pts.T +
                          self.data_input_2d.src_extrinsic[:3, 3][:, None])
        src_pixel = self.data_input_2d.camera_intrinsic @ src_pts_camera[:3, :]
        src_pixel_normalized = src_pixel / src_pixel[2, :]

        src_pts_u = src_pixel_normalized[0, :].cpu().numpy()
        src_pts_v = (self.method.image_size[0] - src_pixel_normalized[1, :]).cpu().numpy()

        # Mask to ensure points are within image bounds
        mask_valid_pts_uv_src = ((src_pts_u > 0) & (src_pts_u < self.method.image_size[1]) &
                                 (src_pts_v > 0) & (src_pts_v < self.method.image_size[0]))
        src_pts_u = src_pts_u[mask_valid_pts_uv_src]
        src_pts_v = src_pts_v[mask_valid_pts_uv_src]

        ############
        # return back for checking
        # test_v = self.method.image_size[0]  - src_pts_v
        # test_u = src_pts_u
        # # test_v = test_v * src_pixel[2, :]
        # # test_u = test_u * src_pixel[2, :]
        # test_uv = np.vstack((test_u, test_v)).T
        # test_uv = torch.from_numpy(test_uv).to(self.device)
        # ones = torch.ones_like(test_uv[:, :1])
        # test_uv = torch.cat((test_uv, ones), dim=1)
        # test_uv = test_uv * src_pixel[2, :][mask_valid_pts_uv_src].unsqueeze(0).T
        #
        # intrinsic_inv = torch.linalg.inv(self.data_input_2d.camera_intrinsic)
        # # intrinsic_inv = copy.deepcopy(self.data_input_2d.camera_intrinsic)
        # # intrinsic_inv[0, 0] = 1 / intrinsic_inv[0, 0]
        # # intrinsic_inv[1, 1] = 1 / intrinsic_inv[1, 1]
        # # intrinsic_inv[0, 2] = - intrinsic_inv[0, 2]
        # # intrinsic_inv[1, 2] = - intrinsic_inv[1, 2]
        #
        # src_3d_camera = intrinsic_inv @ test_uv.T
        # # tgt_3d_points = intrinsic_inv @ test_uv.T
        #
        # rotation_matrix = self.data_input_2d.src_extrinsic[:3, :3]
        # translation_vector = self.data_input_2d.src_extrinsic[:3, 3]
        # rotation_inv = rotation_matrix.T
        # translation_inv = -rotation_inv @ translation_vector
        # src_3d_world = rotation_inv @ src_3d_camera + translation_inv[:, None]
        # src_3d_world = src_3d_world.T
        # src_3d_world_orignal = self.data_input_3d.src_pts[mask_valid_pts_uv_src]
        ##############

        # Save projected points and indices
        self.data_interim.proj_uv_src = torch.cat((torch.tensor(src_pts_u)[:, None], torch.tensor(src_pts_v)[:, None]),
                                                  dim=1)
        # revert for Riegl scans
        if self.data.dataset != 'rockfall_simulator':
            self.data_interim.proj_uv_src = self.data_interim.proj_uv_src[:, [1, 0]]

        # Handle source indices within image
        self.data_interim.idx_valid_src = self.data_input_3d.idx_initial_src[mask_valid_pts_uv_src]

        # Repeat similar projection steps for target points
        tgt_pts_camera = (self.data_input_2d.tgt_extrinsic[:3, :3] @ self.data_input_3d.tgt_pts.T +
                          self.data_input_2d.tgt_extrinsic[:3, 3][:, None])
        tgt_pixel = self.data_input_2d.camera_intrinsic @ tgt_pts_camera[:3, :]
        tgt_pixel_normalized = tgt_pixel / tgt_pixel[2, :]

        tgt_pts_u = tgt_pixel_normalized[0, :].cpu().numpy()
        tgt_pts_v = (self.method.image_size[0] - tgt_pixel_normalized[1, :]).cpu().numpy()

        mask_valid_pts_uv_tgt = ((tgt_pts_u > 0) & (tgt_pts_u < self.method.image_size[1]) &
                                 (tgt_pts_v > 0) & (tgt_pts_v < self.method.image_size[0]))
        tgt_pts_u = tgt_pts_u[mask_valid_pts_uv_tgt]
        tgt_pts_v = tgt_pts_v[mask_valid_pts_uv_tgt]

        if self.method.lifting_type == 'interpolation':
            src_depth = src_pts_camera[2, :]  # Capture depth values from Z-coordinate
            src_depth = src_depth[mask_valid_pts_uv_src]  # Keep valid depth values only

            tgt_depth = tgt_pts_camera[2, :]
            tgt_depth = tgt_depth[mask_valid_pts_uv_tgt]

            # Initialize depth map
            depth_map_src = - np.ones(self.method.image_size, dtype=np.float32)
            depth_map_src[src_pts_v.astype(int), src_pts_u.astype(int)] = src_depth.cpu().numpy()

            # depth_map_src = densify_depth_map(depth_map_src, src_pts_u, src_pts_v, src_depth, self.method.image_size)

            # Initialize target depth map
            depth_map_tgt = - np.ones(self.method.image_size, dtype=np.float32)
            depth_map_tgt[tgt_pts_v.astype(int), tgt_pts_u.astype(int)] = tgt_depth.cpu().numpy()

            # depth_map_tgt = densify_depth_map(depth_map_tgt, tgt_pts_u, tgt_pts_v, tgt_depth, self.method.image_size)

            # Store depth maps in the interim data for future lifting of 2D matches to 3D
            self.data_interim.depth_map_src = depth_map_src
            self.data_interim.depth_map_tgt = depth_map_tgt

            # import matplotlib.pyplot as plt
            # # Display the depth map with a colormap for better visualization
            # plt.figure(figsize=(10, 8))
            # plt.imshow(np.transpose(depth_map_src), cmap='viridis')
            # plt.colorbar(label='Depth')
            # # plt.title(title)
            # plt.axis('on')  # Hide axis for clarity
            # plt.show()

            # visualize src rgb as well
            # plt.figure(figsize=(10, 8))
            # plt.imshow(self.data_input_2d.src_image[:,:,0], cmap='viridis')
            # plt.colorbar(label='rgb')
            # # plt.title(title)
            # plt.axis('on')  # Hide axis for clarity
            # plt.show()

            a = 0

        # Save target points and indices
        self.data_interim.proj_uv_tgt = torch.cat((torch.tensor(tgt_pts_u)[:, None], torch.tensor(tgt_pts_v)[:, None]),
                                                  dim=1)
        if self.data.dataset != 'rockfall_simulator':
            self.data_interim.proj_uv_tgt = self.data_interim.proj_uv_tgt[:, [1, 0]]

        self.data_interim.idx_valid_tgt = self.data_input_3d.idx_initial_tgt[mask_valid_pts_uv_tgt]

        a = 0

    def _compute_median_resolution(self):
        """
        Computes the median point cloud resolution of the tiles, where point cloud resolution is defined as the distance to the closes point
        in teh same point cloud.
        source: [F2S3](https://github.com/gseg-ethz/F2S3_pc_deformation_monitoring)
        :param self:
        :return: pc_resolution (float): min resolution across the source and target point cloud (median distance to the closet point)
        """

        # start_time = time.time()
        # Compute the point cloud resolution of the source point cloud (k=2 as the closest point is the point itself)
        neigh = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        neigh.fit(np.asarray(self.data_input_3d.src_pts.cpu()))
        dist01, _ = neigh.kneighbors(np.asarray(self.data_input_3d.src_pts.cpu()), return_distance=True)

        # Resolution of the source point cloud
        resolution_s = np.median(dist01[:, -1])

        # Compute the point cloud resolution of the target point cloud (k=2 as the closest point is the point itself)
        neigh.fit(np.asarray(self.data_input_3d.tgt_pts.cpu()))
        dist01, _ = neigh.kneighbors(np.asarray(self.data_input_3d.tgt_pts.cpu()), return_distance=True)

        # Resolution of the target point cloud
        resolution_t = np.median(dist01[:, -1])

        # end_time = time.time()

        # if args.verbose:
        #     args.logging.info(
        #         '2.3 Median point cloud resolution of the tiles computed in: {:.2f} s'.format(end_time - start_time))

        # Lower of the both point cloud resolutions (i.e. larger median distance to the closest points)

        tile_st_resolution = max(resolution_s, resolution_t)

        return tile_st_resolution
