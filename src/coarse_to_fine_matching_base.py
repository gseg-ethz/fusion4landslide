# ---------------------------------------------------------------------------- #
# Main functions of the proposed fusion method
# ---------------------------------------------------------------------------- #
import abc
import os
from easydict import EasyDict as edict
import os.path as osp
import numpy as np
from utils.o3d_tools import array2tensor, visualize_patch_match, icp_registration, tensor2pcd, pcd2tensor, pcd2array, array2pcd
import torch
import gc
import shutil
import matplotlib
from scipy.spatial import KDTree


# # Check if we're in a headless environment (like a server or remote machine with no display)
# if os.environ.get("DISPLAY", "") == "":
#     print("No display found. Using non-interactive 'Agg' backend.")
#     matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive, headless environments
# else:
#     print("Display found. Using interactive 'TkAgg' backend.")
#     matplotlib.use('TkAgg')  # Use 'TkAgg' backend for interactive environments with a display

import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.common import dir_exist
# import copy
from scripts.weighted_svd import refine_local_rigid_correspondences
from cpp_core.supervoxel_segmentation.build import supervoxel as supervoxel_partition
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# import hdbscan
from collections import Counter
from src.data_loader import Preprocess_Dataset
import copy
import matplotlib.cm as cm

import open3d as o3d
import cv2
from scipy.spatial import cKDTree
import hnswlib
from PIL import Image
from copy import deepcopy
from src.superpoint_partition import superpoint_partition


def refine_dvfs_with_threshold(src_pts, transformed_src_pts, tgt_pts, distance_threshold=0.1, batch_size=1024):
    """
    Refine matches by finding nearest tgt_pts to transformed_src_pts with a threshold.

    Args:
        src_pts (torch.Tensor): (N, 3) original source points.
        transformed_src_pts (torch.Tensor): (N, 3) transformed source points after coarse transformation.
        tgt_pts (torch.Tensor): (M, 3) target points.
        distance_threshold (float): Maximum distance for a valid match.
        batch_size (int): Batch size for efficient processing.

    Returns:
        torch.Tensor: (K, 6) refined matches [src_pts, matched_tgt_pts], only valid matches retained.
    """
    device = src_pts.device

    # Convert target points to Open3D point cloud
    tgt_o3d = o3d.geometry.PointCloud()
    tgt_o3d.points = o3d.utility.Vector3dVector(tgt_pts.cpu().numpy())  # Open3D requires numpy
    tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_o3d)

    # Prepare storage for refined matches
    refined_matches = []

    # Batch processing to avoid memory issues
    for i in range(0, transformed_src_pts.shape[0], batch_size):
        batch_transformed = transformed_src_pts[i:i + batch_size].cpu().numpy()

        # Batch nearest neighbor search
        nearest_tgt = []
        valid_mask = []

        for pt in batch_transformed:
            [_, idx, dists] = tgt_kdtree.search_knn_vector_3d(pt, 1)  # Get nearest point
            if len(idx) > 0 and dists[0] < distance_threshold ** 2:  # Open3D returns squared distance
                nearest_tgt.append(tgt_pts[idx[0]].unsqueeze(0))
                valid_mask.append(True)
            else:
                nearest_tgt.append(torch.zeros((1, 3), device=device))  # Placeholder for invalid matches
                valid_mask.append(False)

        # Convert list to tensor
        nearest_tgt = torch.cat(nearest_tgt, dim=0).to(device)
        valid_mask = torch.tensor(valid_mask, device=device)

        # Store valid matches
        refined_matches.append(torch.cat([src_pts[i:i + batch_size], nearest_tgt], dim=1)[valid_mask])

    # Concatenate all batches
    return torch.cat(refined_matches, dim=0) if refined_matches else torch.empty((0, 6), device=device)


def crop_and_save(img, img_path, start_x, start_y, crop_size, output_path):
    # Crop the image
    img_cropped = img.crop((start_x, start_y, start_x + crop_size[0], start_y + crop_size[1]))

    # Save the cropped image
    save_dir = osp.join(output_path, 'cropped_images', img_path)
    dir_exist(save_dir)
    img_cropped.save(osp.join(save_dir, f'{start_x}_{start_y}.jpg'))
    # img_cropped.save(osp.join(save_dir, f'{start_x}_{start_y}.jpg'))


def extract_img_patch_brienz_tls(color, w, h, img_size=64, enlarge_patch_size=False):
    if enlarge_patch_size:
        # enlarge the region
        size = 50
        h[0] = np.max((0, h[0] - size))
        h[1] = np.min((h[1] + size, color.shape[0]))
        w[0] = np.max((0, w[0] - size))
        w[1] = np.min((w[1] + size, color.shape[0]))

        image = Image.fromarray(color[h[0]: h[1], w[0]: w[1]])
        # resample=Image.Resampling.NEAREST
    else:
        ############
        # original
        image = Image.fromarray(color[h[0]: h[1], w[0]: w[1]])
        ############

    ###############
    # visualize
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    #
    # # Create a plot with both the original image and a highlighted rectangle
    # fig, ax = plt.subplots(1)
    # # Display the original image
    # ax.imshow(color)
    # # Create a rectangle patch to highlight the cropped area
    # rect = patches.Rectangle((w[0], h[0]), w[1] - w[0], h[1] - h[0], linewidth=2, edgecolor='r', facecolor='none')
    # # Add the rectangle to the plot
    # ax.add_patch(rect)
    # # Optionally show the cropped image in another figure or subplot
    # plt.figure()
    # plt.imshow(image)
    # # Show both plots
    # plt.show()
    ###############

    # plt.imshow(image)
    # plt.show()
    image = image.resize((img_size, img_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def load_state_with_same_shape(model, weights):
    # print("Loading weights:" + ', '.join(weights.keys()))
    model_state = model.state_dict()
    filtered_weights = {
        k[9:]: v for k, v in weights.items() if k[9:] in model_state and v.size() == model_state[k[9:]].size()
    }
    # print("Loaded weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights

def load_pretrain_model(model_name, dim, checkpoint_dir):
    # model_name = 'Res50UNet'
    backbone2d = build_backbone(model_name, dim, weights='DEFAULT')
    backbone2d = resume_checkpoint(backbone2d, checkpoint_dir)
    return backbone2d

def resume_checkpoint(backbone2d, checkpoint_dir='models/checkpoint.pth'):
    import os
    from torch.serialization import default_restore_location
    if os.path.isfile(checkpoint_dir):
        # print('Loading 2D pretrained checkpoint for pixel-wise image feature extraction...')
        state = torch.load(checkpoint_dir, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        # load weights
        model = backbone2d
        matched_weights = load_state_with_same_shape(model, state['model'])
        # print("matched weight: ",matched_weights)
        model.load_state_dict(matched_weights, strict=False)
        del state
        return model


def adjust_intrinsic(intrinsic_before, image_dim_before, image_dim_after):
    # adjust camera intrinsics if resize the image

    if image_dim_before == image_dim_after:
        return intrinsic_before

    intrinsic_return = np.copy(intrinsic_before)

    height_before = image_dim_before[0]
    height_after = image_dim_after[0]
    height_ratio = height_after / height_before

    width_after = image_dim_after[1]
    width_before = image_dim_before[1]
    width_ratio = width_after / width_before

    if width_ratio >= height_ratio:
        resize_height = height_after
        resize_width = height_ratio * width_before
    else:
        resize_width = width_after
        resize_height = width_ratio * height_before

    intrinsic_return[0, 0] *= float(resize_width) / float(width_before)
    intrinsic_return[1, 1] *= float(resize_height) / float(height_before)
    # account for cropping/padding here
    intrinsic_return[0, 2] *= float(resize_width - 1) / float(width_before - 1)
    intrinsic_return[1, 2] *= float(resize_height - 1) / float(height_before - 1)

    return intrinsic_return


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Args:
    q (tuple or list): Quaternion (w, x, y, z)

    Returns:
    np.ndarray: Corresponding 3x3 rotation matrix
    """
    w, x, y, z = q

    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

    return R


def get_intersect_tensor(tensor1, tensor2, device='cpu'):
    """ Get intersection of two given tensors """
    # Flatten the tensors to 1D to perform set operations
    tensor1_flat = tensor1.view(-1, tensor1.size(-1)).to(device)
    tensor2_flat = tensor2.view(-1, tensor2.size(-1)).to(device)

    # Use set operations to find common rows
    set_tensor1 = {tuple(row.tolist()) for row in tensor1_flat}
    set_tensor2 = {tuple(row.tolist()) for row in tensor2_flat}

    common_elements_set = set_tensor1.intersection(set_tensor2)

    if common_elements_set:
        common_elements = torch.tensor([list(elem) for elem in common_elements_set], dtype=tensor1.dtype, device=device)
    else:
        common_elements = torch.empty((0, tensor1.size(1)), dtype=tensor1.dtype, device=device)

    # Find the indices in the original tensors
    indices_tensor1 = torch.nonzero((tensor1_flat[:, None] == common_elements).all(-1), as_tuple=True)[0]
    indices_tensor2 = torch.nonzero((tensor2_flat[:, None] == common_elements).all(-1), as_tuple=True)[1]

    return common_elements, indices_tensor1, indices_tensor2


def get_union_tensor(tensor1, tensor2, device='cpu'):
    """ Get union of two given tensors """
    # Flatten the tensors to 1D to perform set operations
    tensor1_flat = tensor1.view(-1, tensor1.size(-1)).to(device)
    tensor2_flat = tensor2.view(-1, tensor2.size(-1)).to(device)

    # Use set operations to find unique rows from both tensors
    set_tensor1 = {tuple(row.tolist()) for row in tensor1_flat}
    set_tensor2 = {tuple(row.tolist()) for row in tensor2_flat}

    union_elements_set = set_tensor1.union(set_tensor2)

    if union_elements_set:
        union_elements = torch.tensor([list(elem) for elem in union_elements_set], dtype=tensor1.dtype, device=device)
    else:
        union_elements = torch.empty((0, tensor1.size(1)), dtype=tensor1.dtype, device=device)

    # Finding indices without exceeding INT_MAX
    indices_tensor1 = []
    indices_tensor2 = []

    for i, row in enumerate(tensor1_flat):
        if tuple(row.tolist()) in union_elements_set:
            indices_tensor1.append(i)

    for i, row in enumerate(tensor2_flat):
        if tuple(row.tolist()) in union_elements_set:
            indices_tensor2.append(i)

    indices_tensor1 = torch.tensor(indices_tensor1, device=device)
    indices_tensor2 = torch.tensor(indices_tensor2, device=device)

    return union_elements, indices_tensor1, indices_tensor2


def read_2d_matches(path):
    if not osp.exists(path):
        return np.asarray([])
    corr_list = sorted(os.listdir(path))
    single_check = False
    if single_check:
        # src_960_1536_tgt_960_1024.txt
        # src_320_0_tgt_0_0.txt, corr_list[160]
        #########
        corr_list = corr_list[377]
        corr = np.loadtxt(osp.join(path, corr_list))
        return corr
    else:
        #########
        corr_all = []
        # for corr_path in tqdm(corr_list, position=0, leave=True):
        for corr_path in corr_list:
            corr = np.loadtxt(osp.join(path, corr_path))
            corr_all.append(corr)
        corr_all_non_empty = [arr for arr in corr_all if arr.size > 0]
        return np.vstack(corr_all_non_empty)


def bilinear_interpolate(depth_map, u, v, type='single_closest'):
    """
    Performs bilinear interpolation on GPU for depth values.

    Args:
        depth_map (torch.Tensor): Depth map of the image on GPU (height, width).
        u, v (torch.Tensor): Tensors of non-integer pixel coordinates on GPU.

    Returns:
        torch.Tensor: Interpolated depth values at (u, v) with shape (batch_size, 1).
    """

    # v should be depth_map.shape[0] - v
    # v = depth_map.shape[0] - v

    if type == 'single_closest':
        u0 = u.floor().long()
        v0 = v.floor().long()
        u0 = torch.clamp(u0, 0, depth_map.shape[0] - 1)
        v0 = torch.clamp(v0, 0, depth_map.shape[1] - 1)
        # u -- 1920, v -- 2560; depth_map [1920, 2560]
        depth00 = depth_map[u0, v0]
        mask_valid_depth = (depth00 >= 0)
        return depth00.unsqueeze(1), mask_valid_depth  # Ensure output shape is (batch_size, 1)
    else:
        u0 = u.floor().long()
        v0 = v.floor().long()
        u1 = u0 + 1
        v1 = v0 + 1

        # Clip coordinates to be within depth_map bounds
        u0 = torch.clamp(u0, 0, depth_map.shape[1] - 1)
        v0 = torch.clamp(v0, 0, depth_map.shape[0] - 1)
        u1 = torch.clamp(u1, 0, depth_map.shape[1] - 1)
        v1 = torch.clamp(v1, 0, depth_map.shape[0] - 1)

        import matplotlib.pyplot as plt
        # # Display the depth map with a colormap for better visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(np.transpose(np.asarray(depth_map.cpu())), cmap='viridis')
        plt.colorbar(label='Depth')
        # plt.title(title)
        plt.axis('on')  # Hide axis for clarity
        plt.show()

        # Depth values at the four surrounding pixels
        depth00 = depth_map[v0, u0]
        depth10 = depth_map[v0, u1]
        depth01 = depth_map[v1, u0]
        depth11 = depth_map[v1, u1]

        # mask_valid_depth = torch.zeros(len(depth00), dtype=torch.bool)
        mask_valid_depth = (depth00 >= 0) & (depth10 >= 0) & (depth01 >= 0) & (depth11 >= 0)

        # Fractional parts of the coordinates
        du = u - u0.float()
        dv = v - v0.float()

        # Bilinear interpolation formula
        depth = (depth00 * (1 - du) * (1 - dv) +
                 depth10 * du * (1 - dv) +
                 depth01 * (1 - du) * dv +
                 depth11 * du * dv)

        return depth.unsqueeze(1), mask_valid_depth  # Ensure output shape is (batch_size, 1)


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


def compute_img_range_brienz_tls(K, sample_tmp, color, neighbor_radius=None):
    # sample_tmp = compute_bounding_box(single_pts, radius=neighbor_radius)
    # x, y --> width, height
    # sample_tmp[:, 0] = np.round(sample_tmp[:, 0] * K[0, 0] / sample_tmp[:, 2] + K[0, 2])
    # sample_tmp[:, 1] = np.round(sample_tmp[:, 1] * K[1, 1] / sample_tmp[:, 2] + K[1, 2])

    src_pixel = K @ sample_tmp[:, :3].T
    src_pixel_normalized = src_pixel / src_pixel[2, :]
    src_pts_u = src_pixel_normalized[0, :]
    src_pts_v = 1920 - src_pixel_normalized[1, :]

    proj_uv_src = torch.cat((src_pts_v[:, None], src_pts_u[:, None]), dim=1).cpu()

    # [-4.0574e-01, 1.5852e+00, -1.2538e-05]

    width = np.array([np.clip(torch.min(proj_uv_src[:, 0]), 0, color.shape[1]),
                      np.clip(torch.max(proj_uv_src[:, 0]), 0, color.shape[1])], dtype=np.int32)
    height = np.array([np.clip(torch.min(proj_uv_src[:, 1]), 0, color.shape[0]),
                       np.clip(torch.max(proj_uv_src[:, 1]), 0, color.shape[0])], dtype=np.int32)
    return width, height


def construct_cluster_img_patch(idx_spt2pts_src, src_pts_camera, color_src_1, camera_intrinsic, enlarge_patch_size):
    src_mask_valid_img_patch = torch.ones(len(idx_spt2pts_src), dtype=torch.bool)
    # 0.3 construct img patch for each cluster: N X 64 X 64 X 3
    img_patch_src = []
    for idx_i, pts_idx_src in enumerate(idx_spt2pts_src):
        pts_coord_in_spt_src = src_pts_camera[pts_idx_src, :]

        # save pts to path for check
        # np.savetxt(osp.join('/scratch2/zhawang/projects/deformation/DeformHD_local/output', 'src_0_pts.txt'), pts_coord_in_spt_src)

        img_size = 64
        # train setting: 10
        min_size = 10
        # get pts, compute img range
        # neigh_i_raw = np.matmul(neigh_i, ref_1_rot[:3, :3])

        # initialize img_patch
        img_patch = []

        # pts_to_frame_1 = copy.deepcopy(pts_coord_in_spt_src)
        pts_to_frame_1 = pts_coord_in_spt_src

        # if one cluster has quite few points, ignore the image feature extraction for this cluster
        min_pts = 5
        # if there is only one point, below one will return 3
        if pts_to_frame_1.shape[0] < min_pts:
            # img_patch = []
            src_mask_valid_img_patch[idx_i] = False
        else:
            width_1, height_1 = compute_img_range_brienz_tls(camera_intrinsic, pts_to_frame_1, color_src_1)
            img_size_1 = [width_1[1] - width_1[0], height_1[1] - height_1[0]]
            if all(num >= min_size for num in img_size_1):
                img_patch = extract_img_patch_brienz_tls(color_src_1, width_1, height_1, img_size=img_size,
                                                         enlarge_patch_size=enlarge_patch_size)
            else:
                src_mask_valid_img_patch[idx_i] = False

        if img_patch != []:
            img_patch_src.append(img_patch)

        # retain the indices, while ignore the invalid patch when the mask value is False
        # img_patch_src.append(img_patch)
    return img_patch_src, src_mask_valid_img_patch


def compute_confidence(source_points, target_points, transformation, max_dist_threshold=None):
    """
    Compute confidence based on ICP residual error.

    Args:
        source_points (numpy.ndarray): Nx3 array of source points.
        target_points (numpy.ndarray): Mx3 array of target points.
        transformation (numpy.ndarray): 4x4 transformation matrix.
        max_dist_threshold (float, optional): Maximum distance threshold to filter outliers.

    Returns:
        numpy.ndarray: Confidence scores for each source point.
    """
    # Transform source points using ICP transformation
    source_transformed = (transformation[:3, :3] @ source_points.T).T + transformation[:3, 3]

    # Build KDTree for nearest neighbor search
    target_tree = KDTree(target_points)
    distances, _ = target_tree.query(source_transformed)

    # Set dynamic threshold if not provided (e.g., 95th percentile)
    if max_dist_threshold is None:
        max_dist_threshold = np.percentile(distances, 95)

    # Clip distances to prevent extreme outliers from dominating
    distances_clipped = np.clip(distances, 0, max_dist_threshold)
    eps = 1e-6
    sigma = np.median(distances_clipped) + eps # Use median to reduce sensitivity to outliers

    # Compute confidence as an exponential decay function
    confidence = np.exp(- (distances_clipped ** 2) / (sigma ** 2))

    # Assign zero confidence to outliers beyond threshold
    confidence[distances > max_dist_threshold] = 0

    return confidence


def compute_confidence_cuda(source_points, target_points, transformation, max_dist_threshold=None):
    """
    Compute confidence based on ICP residual error using CUDA.

    Args:
        source_points (torch.Tensor): (N,3) tensor of source points on CUDA.
        target_points (torch.Tensor): (M,3) tensor of target points on CUDA.
        transformation (torch.Tensor): (4,4) transformation matrix on CUDA.
        max_dist_threshold (float, optional): Distance threshold to filter outliers.

    Returns:
        torch.Tensor: (N,) confidence scores.
    """
    # Transform source points
    ones = torch.ones((source_points.shape[0], 1), device=source_points.device)
    source_homo = torch.cat([source_points, ones], dim=1)  # Convert to homogeneous coords
    source_transformed = (transformation @ source_homo.T).T[:, :3]

    # Use KDTree on CPU for nearest neighbor search (PyTorch does not support KDTree)
    target_tree = KDTree(target_points.cpu().numpy())
    distances, _ = target_tree.query(source_transformed.cpu().numpy())

    # Convert back to CUDA tensor
    distances = torch.tensor(distances, device=source_points.device)

    # Dynamic thresholding
    if max_dist_threshold is None:
        max_dist_threshold = torch.quantile(distances, 0.95).item()

    # Clip extreme distances
    distances_clipped = torch.clamp(distances, 0, max_dist_threshold)
    sigma = torch.median(distances_clipped)  # Robust estimate of noise

    # Compute confidence
    confidence = torch.exp(- (distances_clipped ** 2) / (sigma ** 2))
    confidence[distances > max_dist_threshold] = 0  # Outlier rejection

    return confidence


class Coarse2Fine_Base(abc.ABC):
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
        self.para = self.config.parameter_setting
        self.visualize = self.config.visualization

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
                # image_names.append(image_name)

                # Parse translation and rotation
                translation = np.fromstring(file.readline().strip(), sep=' ')
                rotation = np.array([np.fromstring(file.readline().strip(), sep=' ') for _ in range(3)])

                # Construct the 4x4 transformation matrix
                transformation_matrix = torch.eye(4, dtype=torch.float32).to(self.device)
                transformation_matrix[:3, :3] = torch.from_numpy(rotation).to(self.device)
                transformation_matrix[:3, 3] = torch.from_numpy(translation).to(self.device)

                transformation_matrix = torch.linalg.inv(transformation_matrix)

                # Append to source or target lists based on the year in the image name
                # src: Brienz or Mattertal
                # if "2002" in image_name or "210720" in image_name:
                #     img_names_src.append(image_name)
                #     transformation_src.append(transformation_matrix)
                # # tgt: Brienz or Mattertal
                # elif "2011" in image_name or "220706" in image_name:
                #     img_names_tgt.append(image_name)
                #     transformation_tgt.append(transformation_matrix)
                ## TODO: should work for other epochs or other datasets
                if src_id in image_name:
                    img_names_src.append(image_name)
                    transformation_src.append(transformation_matrix)
                # tgt: Brienz or Mattertal
                elif tgt_id in image_name:
                    img_names_tgt.append(image_name)
                    transformation_tgt.append(transformation_matrix)
                # transformation_matrices.append(transformation_matrix)

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

            # in case img data used for constructing patch
            self.data_input_2d.src_image = src_images[0]
            self.data_input_2d.tgt_image = tgt_images[0]
            self.data_input_2d.src_extrinsic = src_extrinsics[0]
            self.data_input_2d.tgt_extrinsic = tgt_extrinsics[0]

        a = 0
        a = 0

    def _read_data(self):
        # full pcd for partition, subsampled pcd for feature computation
        # 3d
        # if 'full' in self.input_root:
        #     self.src_pcd_path = self.config.src_tile_overlap_path
        #     self.tgt_pcd_path = self.config.tgt_tile_overlap_path
        # else:
        #     self.src_pcd_path = osp.join(self.input_root, 'raw_pcd', self.data.src_pcd)
        #     self.tgt_pcd_path = osp.join(self.input_root, 'raw_pcd', self.data.tgt_pcd)

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

        if not self.method.coarse_matching_only_3d & self.method.fine_matching_only_3d:
            # 2d
            camera_intrinsic = np.loadtxt(osp.join(self.input_root, 'image', 'camera_intrinsic.txt'), delimiter=' ')
            self.data_input_2d.camera_intrinsic = array2tensor(camera_intrinsic).to(torch.float32).to(self.device)

            # if 'full' in self.input_root:
            if self.data.multiple_case:
                self._find_the_most_matched_image()

                a = 0

                # output
                # self.data_input_2d.src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
                # self.data_input_2d.tgt_image = cv2.imread(tgt_image_path, cv2.IMREAD_COLOR)
                # self.data_input_2d.src_extrinsic = array2tensor(src_extrinsic).to(torch.float32).to(
                #     self.device)
                # self.data_input_2d.tgt_extrinsic = array2tensor(tgt_extrinsic).to(torch.float32).to(
                #     self.device)
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

                elif self.data.dataset == 'brienz_tls':
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
        # if 'full' in self.output_root:
            return osp.join(self.output_root, 'features', f'features_tile_{self.config.tile_id}.npz')
        else:
            return osp.join(self.output_root, 'features', f'features.npz')

    def _feat_desc_nn(self, input):
        if self.method.point_feat_compute:
            return self.config.feat_desc_nn(input)

    def _voxel_subsampling(self):
        # self.data_interim.src_pcd_sub = self.data_input_3d.src_pcd.voxel_down_sample(voxel_size=self.method.voxel_size_init)
        # self.data_interim.tgt_pcd_sub = self.data_input_3d.tgt_pcd.voxel_down_sample(voxel_size=self.method.voxel_size_init)

        # use an adaptive resolution according to the current point cloud tile
        # intialize the point cloud tile
        self.data_interim.src_pts_sub = (
            pcd2tensor(self.data_input_3d.src_pcd, return_colors=False))
        self.data_interim.tgt_pts_sub = (
            pcd2tensor(self.data_input_3d.tgt_pcd, return_colors=False))

        self.method.voxel_size = self._compute_median_resolution()
        self.data_interim.src_pcd_sub = self.data_input_3d.src_pcd.voxel_down_sample(voxel_size=self.method.voxel_size)
        self.data_interim.tgt_pcd_sub = self.data_input_3d.tgt_pcd.voxel_down_sample(voxel_size=self.method.voxel_size)

        self.data_interim.src_pts_sub = (
            pcd2tensor(self.data_interim.src_pcd_sub, return_colors=False))
        self.data_interim.tgt_pts_sub = (
            pcd2tensor(self.data_interim.tgt_pcd_sub, return_colors=False))

        # self.data_interim.src_pts_sub, self.data_interim.src_clr_sub = (
        #     pcd2tensor(self.data_interim.src_pcd_sub, return_colors=True))
        # self.data_interim.tgt_pts_sub, self.data_interim.tgt_clr_sub = (
        #     pcd2tensor(self.data_interim.tgt_pcd_sub, return_colors=True))

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
        # check if self.model_img_matching exists; ensure the pretrained image model is loaded only once for efficiency
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
                    make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, path=save_img_path, color=None, dpi=30,
                                         text=text, save_individual=True)

                    # fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color=None, dpi=50, text=text)

                    # num_matches.append(len(mkpts0))
                    # save_img = osp.join(visualization_dir, f'src_{src_name}_tgt_{tgt_name}.jpg')
                    # plt.savefig(str(save_img), bbox_inches='tight', pad_inches=0)
                    # plt.close()

        a = 0
    # major functions
    ##################################

    def load_partition(self):
        """ Load raw point clouds with partition types """

        if self.method.partition_type in ['supervoxel', 'superpoint']:
            partition_folder = f'{self.method.partition_type}_partition'
            partition_dir = osp.join(self.output_root, partition_folder)
            if not os.listdir(partition_dir):
                raise FileNotFoundError(f"No partition result in '{partition_dir}'")

        if self.data.dataset in ['rockfall_simulator', 'brienz_tls', 'mattertal', 'common']:
            if self.data.multiple_case:
            # if 'full' in self.output_root:
                partition_path_src = osp.join(self.output_root, partition_folder, f'partition_of_input_src_tile_{self.config.tile_id}.txt')
                partition_path_tgt = osp.join(self.output_root, partition_folder, f'partition_of_input_tgt_tile_{self.config.tile_id}.txt')
            else:
                partition_path_src = osp.join(self.output_root, partition_folder, f'partition_of_input_src.txt')
                partition_path_tgt = osp.join(self.output_root, partition_folder, f'partition_of_input_tgt.txt')
        else:
            raise NotImplementedError

        partition_data_src = np.loadtxt(partition_path_src)
        partition_data_tgt = np.loadtxt(partition_path_tgt)
        # TODO: consider the case using multiple levels of superpoint partition
        # use only one level of the superpoint partition
        if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, int):
            self.data_interim.idx_pts2spt_src = array2tensor(
                partition_data_src[:, 2 + 4 * self.method.level_of_superpoint]).to(torch.int64).to(self.device)
            self.data_interim.idx_pts2spt_tgt = array2tensor(
                partition_data_tgt[:, 2 + 4 * self.method.level_of_superpoint]).to(torch.int64).to(self.device)
        # use multiple levels of the superpoint partition
        elif self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
            idx_pts2spt_src, idx_pts2spt_tgt = [], []
            for level_current in self.method.level_of_superpoint:
                idx_pts2spt_src.append(array2tensor(partition_data_src[:, 2 + 4 * level_current]).to(torch.int64).to(self.device))
                idx_pts2spt_tgt.append(array2tensor(partition_data_tgt[:, 2 + 4 * level_current]).to(torch.int64).to(self.device))
            self.data_interim.idx_pts2spt_src_multiple = idx_pts2spt_src
            self.data_interim.idx_pts2spt_tgt_multiple = idx_pts2spt_tgt
        else:
            self.data_interim.idx_pts2spt_src = array2tensor(partition_data_src[:, 6]).to(torch.int64).to(self.device)
            self.data_interim.idx_pts2spt_tgt = array2tensor(partition_data_tgt[:, 6]).to(torch.int64).to(self.device)

        # TODO: necessary or not
        # self.data_interim.idx_pts2spt_src_original = copy.deepcopy(self.data_interim.idx_pts2spt_src)
        # self.data_interim.idx_pts2spt_tgt_original = copy.deepcopy(self.data_interim.idx_pts2spt_tgt)

        ##################
        # check a single cluster result
        # mask_0 = torch.where(self.data_interim.idx_pts2spt_src == 0)
        # pts_src_0 = self.data_input_3d.src_pts[mask_0]
        # o3d.visualization.draw_geometries([tensor2pcd(pts_src_0)])

        # self.data_interim.idx_pts2spt_tgt = array2tensor(partition_data_tgt[:, 6]).to(torch.int64).to(self.device)
        # self.data_input.pts_coord_src = array2tensor(cluster_data_src[:, :3]).to(self.device)
        # self.data_input.pts_coord_tgt = array2tensor(cluster_data_tgt[:, :3]).to(self.device)

        if self.verbose:
            if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
                self.logging.info(
                    f'Load patches from multiple levels of superpoint partition with src {len(self.data_interim.idx_pts2spt_src_multiple[0])} points and '
                    f'raw tgt {len(self.data_interim.idx_pts2spt_tgt_multiple[0])} points')
            else:
                self.logging.info(f'Load patches from raw pcd with src {len(self.data_interim.idx_pts2spt_src)} points and '
                                  f'raw tgt {len(self.data_interim.idx_pts2spt_tgt)} points')

    def prepare_pts2spt_dict(self):
        """ Combine superpoint patch indices, all point-spt indices and num. of superpoint patches """

        if self.verbose:
            self.logging.info(f'Prepare point-superpoint indices...')
        # self.data_interim.spt_idx_src = (spt_src_0.to(torch.int64), self.data_interim.idx_pts2spt_src, len(spt_src_0))
        # self.data_interim.spt_idx_tgt = (spt_tgt_0.to(torch.int64), self.data_interim.idx_pts2spt_src, len(spt_tgt_0))

        if self.method.small_patch_removal:
            # refined spt clusters, removing small patches
            counts_src = Counter(np.asarray(self.data_interim.idx_pts2spt_src.cpu()))
            counts_tgt = Counter(np.asarray(self.data_interim.idx_pts2spt_tgt.cpu()))
            # valid_spt_src = {idx for idx, count in counts.items() if count > 10 and idx != -1}
            valid_spt_src = {idx for idx, count in counts_src.items() if count > self.method.num_min_matches_for_small_patch}
            valid_spt_tgt = {idx for idx, count in counts_tgt.items() if count > self.method.num_min_matches_for_small_patch}
            spt_src_id = valid_spt_src
            spt_tgt_id = valid_spt_tgt
        else:
            # the full spt clusters, no refinement
            spt_src_id = torch.unique(self.data_interim.idx_pts2spt_src, sorted=True, return_inverse=False)
            spt_tgt_id = torch.unique(self.data_interim.idx_pts2spt_tgt, sorted=True, return_inverse=False)

        # save the indices of pts to specific spt
        idx_spt2pts_src, idx_spt2pts_tgt = [], []
        # save the indices of spt
        idx_spt_src, idx_spt_tgt = [], []
        for idx in tqdm(spt_src_id, position=0, leave=True):
            idx_spt_src.append(idx)
            idx_spt2pts_src.append(self.data_input_3d.idx_initial_src[self.data_interim.idx_pts2spt_src == idx])
        for idx in tqdm(spt_tgt_id, position=0, leave=True):
            idx_spt_tgt.append(idx)
            idx_spt2pts_tgt.append(self.data_input_3d.idx_initial_tgt[self.data_interim.idx_pts2spt_tgt == idx])

        # self.data_interim.spt_idx_src = (spt_src_0.to(torch.int64), self.data_input.pts2spt_idx_src, len(spt_src_0))
        # self.data_interim.spt_idx_tgt = (spt_tgt_0.to(torch.int64), self.data_input.pts2spt_idx_tgt, len(spt_tgt_0))

        if self.method.small_patch_removal:
            idx_spt_src = np.hstack(idx_spt_src)
            idx_spt_tgt = np.hstack(idx_spt_tgt)
            self.data_interim.idx_spt_src = torch.from_numpy(idx_spt_src).to(self.device)
            self.data_interim.idx_spt_tgt = torch.from_numpy(idx_spt_tgt).to(self.device)
        else:
            self.data_interim.idx_spt_src = torch.hstack(idx_spt_src)
            self.data_interim.idx_spt_tgt = torch.hstack(idx_spt_tgt)

        # output of the current function
        self.data_interim.idx_spt2pts_src = idx_spt2pts_src
        self.data_interim.idx_spt2pts_tgt = idx_spt2pts_tgt

        if self.verbose:
            self.logging.info(f'Preparing superpoint indices is done!')

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

    def global_matches_from_2d_with_different_types(self):
        'what is needed from 2d matching? 3d ids of 2d matches?'
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

            # for visualization
            if self.data.dataset == 'rockfall_simulator':
                max_mag_visualize = 0.06
            else:
                max_mag_visualize = 5

            if corres_2d_initial.shape[0] > 0:
                if self.method.lifting_type == 'nn_search':
                    # find the corresponding indices of src_pixel from tgt_pixel
                    # idx all, valid indices,
                    idx_proj_src2tgt, mask_in_pixel_thres_src, _ = map_corr_2d_to_3d(
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
                    # if 'full' in self.input_root:
                        np.savetxt(osp.join(self.output_root, 'results',
                                            f'c2f_dvfms_2d_raw_wo_pruning_visualize_tile_{self.config.tile_id}.txt'),
                                   dvfs_3d_from_2d_vector_mag_visualize.cpu())
                    # else:
                    #     np.savetxt(osp.join(self.output_root, 'results',
                    #                         'c2f_dvfms_from_2d_src2tgt_without_pruning_visualize_0_5.txt'),
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
                        # if 'full' in self.input_root:
                            np.savetxt(
                                osp.join(self.output_root, 'results',
                                         f'c2f_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5_tile_{self.config.tile_id}.txt'),
                                dvfs_3d_from_2d_vector_mag_visualize_2.cpu())
                        else:
                            np.savetxt(
                                osp.join(self.output_root, 'results',
                                         'c2f_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5.txt'),
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
                mask_within_max_mag = magnitude <= self.para.max_magnitude

                # not really needed if the idx (with unmatched -1) is saved
                # self.data_interim.corres_3d_from_2d = corres_3d[mask_within_max_mag, :]
                corres_3d_from_2d = corres_3d[mask_within_max_mag, :]

                magnitude_visualize = magnitude[mask_within_max_mag]
                if magnitude_visualize.numel() < 2:
                    # min: 2
                    if self.verbose:
                        self.logging.info(f'Current tile_{self.config.tile_id} has no valid corres_3d_from_2d')
                else:
                    magnitude_visualize[0] = 0
                    magnitude_visualize[1] = max_mag_visualize
                    # TODO: comment for faster computation
                    # dvfs_3d_from_2d_vector_mag = torch.hstack((corres_3d_from_2d[:, :3], magnitude_visualize[:, None]))
                    # if self.data.multiple_case:
                    # # if 'full' in self.input_root:
                    #     np.savetxt(osp.join(self.output_root, 'results',
                    #                         f'c2f_dvfms_from_2d_with_max_mag_removal_visualize_0_5_tile_{self.config.tile_id}.txt'),
                    #                dvfs_3d_from_2d_vector_mag.cpu())
                    # # else:
                    # #     np.savetxt(osp.join(self.output_root, 'results',
                    # #                         'c2f_dvfms_from_2d_with_max_mag_removal_visualize_0_5.txt'),
                    # #                dvfs_3d_from_2d_vector_mag.cpu())

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
                    # if 'full' in self.input_root:
                        self._get_2d_matches_online()

                        if self.verbose:
                            self.logging.info('Implementing 2d matching online is done!')

                    # load image matches from Efficient LoftR
                    if self.data.multiple_case:
                    # if 'full' in self.input_root:
                        corres_2d_initial = read_2d_matches(
                            osp.join(self.output_root, self.config.path_name.img_matching_result_dir,
                                     f'tile_{self.config.tile_id}', 'corres_2d'))
                    else:
                        corres_2d_initial = read_2d_matches(
                            osp.join(self.output_root, self.config.path_name.img_matching_result_dir, 'corres_2d'))

                    # for visualization
                    if self.data.dataset == 'rockfall_simulator':
                        max_mag_visualize = 0.06
                    else:
                        max_mag_visualize = 5

                    if corres_2d_initial.shape[0] > 0:
                        if self.method.lifting_type == 'nn_search':
                            # find the corresponding indices of src_pixel from tgt_pixel
                            # idx all, valid indices,
                            idx_proj_src2tgt, mask_in_pixel_thres_src, _ = map_corr_2d_to_3d(
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
                            # if 'full' in self.input_root:
                                np.savetxt(osp.join(self.output_root, 'results',
                                                    f'c2f_dvfms_from_2d_src2tgt_without_pruning_visualize_0_5_tile_{self.config.tile_id}_{num_src}_{num_tgt}.txt'),
                                           dvfs_3d_from_2d_vector_mag_visualize.cpu())
                            # else:
                            #     np.savetxt(osp.join(self.output_root, 'results',
                            #                         'c2f_dvfms_from_2d_src2tgt_without_pruning_visualize_0_5.txt'),
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
                                # if 'full' in self.input_root:
                                    np.savetxt(
                                        osp.join(self.output_root, 'results',
                                                 f'c2f_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5_tile_{self.config.tile_id}.txt'),
                                        dvfs_3d_from_2d_vector_mag_visualize_2.cpu())
                                else:
                                    np.savetxt(
                                        osp.join(self.output_root, 'results',
                                                 'c2f_dvfms_from_2d_tgt2src_without_pruning_visualize_0_5.txt'),
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
                        mask_within_max_mag = magnitude <= self.para.max_magnitude

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
                            if self.data.multiple_case:
                            # if 'full' in self.input_root:
                                np.savetxt(osp.join(self.output_root, 'results',
                                                    f'c2f_dvfms_from_2d_with_max_mag_removal_visualize_0_5_tile_{self.config.tile_id}_{num_src}_{num_tgt}.txt'),
                                           dvfs_3d_from_2d_vector_mag.cpu())
                            # else:
                            #     np.savetxt(osp.join(self.output_root, 'results',
                            #                         'c2f_dvfms_from_2d_with_max_mag_removal_visualize_0_5.txt'),
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

        torch.cuda.empty_cache()
        gc.collect()

    def compute_point_feat(self):
        """
        Compute features using either local patch-based or fully-convolutional methods
        options: fpfh, gfm, local-patch-based, coarse-to-fine methods
        :return:
        """

        if self.verbose:
            self.logging.info('Start computing voxel features...')

        feat_compute = self.method.point_feat_compute
        if not feat_compute:
            if not osp.exists(self._feat_path):
                raise FileNotFoundError(f"The feature path '{self._feat_path}' is not found")
            self._load_extracted_features()

        elif self.method.feat_type == 'DIPs':
            src_tile_feat, tgt_tile_feat = [], []
            neigh_radius = np.sqrt(3) * (10 * self._compute_median_resolution())
            # neigh_radius = np.sqrt(3) * (3 * self._compute_median_resolution())

            # If running into the GPU memory problems reduce the number of points in a batch (default is 2000).
            # which point cloud is used for feature computation, non-overlap or overlap one?
            src_batch = Preprocess_Dataset(self.data_interim.src_pcd_sub, self.data_interim.src_pcd_sub,
                                           self.para.points_per_batch, neigh_radius)
            tgt_batch = Preprocess_Dataset(self.data_interim.tgt_pcd_sub, self.data_interim.tgt_pcd_sub,
                                           self.para.points_per_batch, neigh_radius)

            src_batch = torch.utils.data.DataLoader(src_batch, batch_size=self.para.batch_size,
                                                    num_workers=self.para.num_workers, shuffle=False, drop_last=False)
            tgt_batch = torch.utils.data.DataLoader(tgt_batch, batch_size=self.para.batch_size,
                                                    num_workers=self.para.num_workers, shuffle=False, drop_last=False)

            # if self.config.debug:
            #     for src_i in src_batch:
            #         src_i = src_i.squeeze(0).cuda()
            #         src_i, _, _ = self.feat_desc_nn(src_i)
            #
            #         src_tile_feat.append(src_i)
            #
            #     for tgt_i in tgt_batch:
            #         tgt_i = tgt_i.squeeze(0).cuda()
            #         tgt_i, _, _ = self.feat_desc_nn(tgt_i)
            #
            #         tgt_tile_feat.append(tgt_i)
            # else:
            for src_i in tqdm(src_batch, position=0, leave=True, desc=f'Each batch has {self.para.points_per_batch} voxels'):
                src_i = src_i.squeeze(0).cuda()
                # input: (1, 1000, 3, 256) --> output: (1, 1000, 64)
                # num_worker, num_point_per_batch, 3D, num_neighbors
                src_i, _, _ = self._feat_desc_nn(src_i)

                src_tile_feat.append(src_i)
            for tgt_i in tqdm(tgt_batch, position=0, desc=f'Each batch has {self.para.points_per_batch} voxels'):
                tgt_i = tgt_i.squeeze(0).cuda()
                tgt_i, _, _ = self._feat_desc_nn(tgt_i)

                tgt_tile_feat.append(tgt_i)

            # for src_i, tgt_i in zip(src_batch, tgt_batch):
            #     src_i = src_i.squeeze(0).cuda()
            #     src_i, _, _ = self._feat_desc_nn(src_i)
            #
            #     tgt_i = tgt_i.squeeze(0).cuda()
            #     tgt_i, _, _ = self._feat_desc_nn(tgt_i)
            #
            #     src_tile_feat.append(src_i)
            #     tgt_tile_feat.append(tgt_i)
            self.data_interim.tile_pts_sub_feat_src = torch.cat(src_tile_feat, dim=0)
            self.data_interim.tile_pts_sub_feat_tgt = torch.cat(tgt_tile_feat, dim=0)

            if self.verbose:
                self.logging.info('Compute point features using pretrained local patch-based model is done!')

            if self.save_interim:
                dir_exist(osp.dirname(self._feat_path))
                save_type = '.npz'
                if save_type == '.npz':
                    np.savez_compressed(
                        self._feat_path,
                        src_feat=self.data_interim.tile_pts_sub_feat_src.cpu(),
                        tgt_feat=self.data_interim.tile_pts_sub_feat_tgt.cpu()
                    )
                if self.verbose:
                    self.logging.info('Save features npz file to %s.', self._feat_path)
        else:
            raise NotImplementedError
            # self.logging.error('Feature descriptor type %s is not supported', self.config.feat_type)

        return None

    def _load_extracted_features(self):
        """ Use pretrained local patch-based model to extract 3D geometric features """
        if self.verbose:
            self.logging.info(f'Start loading extracted features...')
        # feat_path = osp.join(self.output_dir, 'features', f'features_tile_{self.config.data.tile_id_src}.npz')

        if self.data.multiple_case:
        # if 'full' in self.output_root:
            feat_path = osp.join(self.output_root, 'features', f'features_tile_{self.config.tile_id}.npz')
        else:
            feat_path = osp.join(self.output_root, 'features', f'features.npz')

        features = np.load(feat_path)
        self.data_interim.tile_pts_sub_feat_src = array2tensor(features['src_feat']).to(self.device)
        self.data_interim.tile_pts_sub_feat_tgt = array2tensor(features['tgt_feat']).to(self.device)
        if self.verbose:
            self.logging.info(f'Loading features from {feat_path}. The loading is done!')

    def _compute_spt_feat_and_coord(self):
        """ Compute the coordinates and features for each source superpoint and target superpoint """
        if self.verbose:
            self.logging.info(f'Start computing spt coordinates and features...')

        # mask_voxel_src = torch.zeros(len(self.data_interim.idx_pts2spt_src), dtype=bool).to(self.device)
        # mask_voxel_tgt = torch.zeros(len(self.data_interim.idx_pts2spt_tgt), dtype=bool).to(self.device)

        # mask_voxel_src[self.data_interim.idx_voxel2pts_src] = True
        # mask_voxel_tgt[self.data_interim.idx_voxel2pts_tgt] = True

        spt_feat_src, spt_feat_tgt, spt_coord_src, spt_coord_tgt = [], [], [], []
        for pts_idx_src in tqdm(self.data_interim.idx_spt2pts_src):
            idx_voxel_in_spt_src = self.data_interim.idx_pts2voxel_src[pts_idx_src]
            idx_voxel_in_spt_src = idx_voxel_in_spt_src[idx_voxel_in_spt_src >= 0]
            voxel_feats_in_spt_src = self.data_interim.tile_pts_sub_feat_src[idx_voxel_in_spt_src, :]
            voxel_coord_in_spt_src = self.data_interim.src_pts_sub[idx_voxel_in_spt_src, :]

            if self.method.feat_aggregate_type == 'avg':
                # average
                spt_feat_src_temp_mean = torch.mean(voxel_feats_in_spt_src, dim=0)
                spt_feat_src_temp = spt_feat_src_temp_mean
                # std
                # spt_feat_src_temp_std = torch.std(voxel_feats_in_spt_src, dim=0) * 5
                # spt_feat_src_temp = torch.cat((spt_feat_src_temp_mean, spt_feat_src_temp_std), dim=0)
            elif self.method.feat_aggregate_type == 'vlad':
                from sklearn.cluster import KMeans
                from src.old.c2f_matching_functions import compute_vlad

                # pts_feat_in_curr_spt_src_np = pts_feat_in_curr_spt.cpu().numpy()
                features = voxel_feats_in_spt_src.cpu()
                # features = torch.rand(pts_feat_in_curr_spt_src_np.shape[0], pts_feat_in_curr_spt_src_np.shape[1])
                num_clusters = 1
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
                spt_feat_src_temp = compute_vlad(features, kmeans).to(self.device)

            spt_coord_src_temp = torch.mean(voxel_coord_in_spt_src, dim=0)

            spt_feat_src.append(spt_feat_src_temp)
            spt_coord_src.append(spt_coord_src_temp)

        for pts_idx_tgt in tqdm(self.data_interim.idx_spt2pts_tgt):
            idx_voxel_in_spt_tgt = self.data_interim.idx_pts2voxel_tgt[pts_idx_tgt]
            idx_voxel_in_spt_tgt = idx_voxel_in_spt_tgt[idx_voxel_in_spt_tgt >= 0]
            voxel_feats_in_spt_tgt = self.data_interim.tile_pts_sub_feat_tgt[idx_voxel_in_spt_tgt, :]
            voxel_coord_in_spt_tgt = self.data_interim.tgt_pts_sub[idx_voxel_in_spt_tgt, :]
            if self.method.feat_aggregate_type == 'avg':
                # average
                spt_feat_tgt_temp_mean = torch.mean(voxel_feats_in_spt_tgt, dim=0)
                spt_feat_tgt_temp = spt_feat_tgt_temp_mean
                # std
                # spt_feat_tgt_temp_std = torch.std(voxel_feats_in_spt_src, dim=0) * 5
                # spt_feat_tgt_temp = torch.cat((spt_feat_tgt_temp_mean, spt_feat_tgt_temp_std), dim=0)
            elif self.method.feat_aggregate_type == 'vlad':
                from sklearn.cluster import KMeans
                from src.old.c2f_matching_functions import compute_vlad

                # pts_feat_in_curr_spt_src_np = pts_feat_in_curr_spt.cpu().numpy()
                features = voxel_feats_in_spt_tgt.cpu()
                # features = torch.rand(pts_feat_in_curr_spt_src_np.shape[0], pts_feat_in_curr_spt_src_np.shape[1])
                num_clusters = 1
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
                spt_feat_tgt_temp = compute_vlad(features, kmeans).to(self.device)

            spt_coord_tgt_temp = torch.mean(voxel_coord_in_spt_tgt, dim=0)

            spt_feat_tgt.append(spt_feat_tgt_temp)
            spt_coord_tgt.append(spt_coord_tgt_temp)

        spt_feat_src = torch.stack(spt_feat_src, dim=0)
        spt_coord_src = torch.stack(spt_coord_src, dim=0)
        spt_feat_tgt = torch.stack(spt_feat_tgt, dim=0)
        spt_coord_tgt = torch.stack(spt_coord_tgt, dim=0)

        self.data_interim.spt_feat_src = spt_feat_src
        self.data_interim.spt_feat_tgt = spt_feat_tgt
        self.data_interim.spt_coord_src = spt_coord_src
        self.data_interim.spt_coord_tgt = spt_coord_tgt

        if self.verbose:
            self.logging.info(f'Computing superpoint coordinates and features is done! '
                              f'Num. of spt: {spt_feat_src.shape[0]} and {spt_feat_tgt.shape[0]}')

    def _extract_img_pretrain_feats(self):
        # for pixel_wise
        # follow CoFF
        from torchvision.transforms import transforms, InterpolationMode
        image_resize = self.data_input_2d.src_image.shape[:2]
        image_resize = list(image_resize)

        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_resize, InterpolationMode.NEAREST),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
        ])

        img_transformed_src = img_transform(self.data_input_2d.src_image)
        img_transformed_tgt = img_transform(self.data_input_2d.tgt_image)

        pretrain_root = osp.join(self.config.path_name.project_dir, self.config.path_name.weight_dir,
                                 self.config.path_name.pretrained_pixel_2d_weight)
        self.logging.info('Loading 2D pretrained checkpoint for pixel-wise image feature extraction...')
        backbone2d = load_pretrain_model('Res50UNet', 128, pretrain_root)
        backbone2d = backbone2d.to(self.device)

        # extract 128-dim img feats from pretrained weight
        for param in backbone2d.parameters():
            param.requires_grad = False
        img_feat_src = backbone2d(img_transformed_src.unsqueeze(0).to(self.device)).squeeze(0)
        img_feat_tgt = backbone2d(img_transformed_tgt.unsqueeze(0).to(self.device)).squeeze(0)

        voxel_pts_coord_src = self.data_interim.src_pts_sub
        voxel_pts_coord_tgt = self.data_interim.tgt_pts_sub

        src_pts_camera = (self.data_input_2d.src_extrinsic[:3, :3] @ voxel_pts_coord_src.T +
                          self.data_input_2d.src_extrinsic[:3, 3][:, None])

        tgt_pts_camera = (self.data_input_2d.tgt_extrinsic[:3, :3] @ voxel_pts_coord_tgt.T +
                          self.data_input_2d.tgt_extrinsic[:3, 3][:, None])

        # necessary because the pretrained model extraction at 1/2 resolution
        image_resize_after = copy.deepcopy(image_resize)
        image_resize_after[0] = int(image_resize_after[0] / 2)
        image_resize_after[1] = int(image_resize_after[1] / 2)
        camera_intrinsic_adjust = adjust_intrinsic(self.data_input_2d.camera_intrinsic.cpu(), image_resize, image_resize_after)
        camera_intrinsic_adjust = torch.from_numpy(camera_intrinsic_adjust).to(self.device)

        #####################
        # src part
        src_pixel = camera_intrinsic_adjust @ src_pts_camera[:3, :]
        src_pixel_normalized = src_pixel / src_pixel[2, :]
        src_pts_u = src_pixel_normalized[0, :]
        # 1920 / 2
        src_pts_v = image_resize_after[1] - src_pixel_normalized[1, :]

        # may should be image_resize_after instead of image_resize
        mask_valid_pts_uv_src = ((src_pts_u > 0) & (src_pts_u < image_resize_after[0]) &
                                 (src_pts_v > 0) & (src_pts_v < image_resize_after[1]))
        # remove invalid pixels that exceed the image range
        src_pts_u = src_pts_u[mask_valid_pts_uv_src]
        src_pts_v = src_pts_v[mask_valid_pts_uv_src]

        proj_uv_src = torch.cat((src_pts_u[:, None], src_pts_v[:, None]), dim=1)
        proj_uv_src_2 = proj_uv_src.to(torch.int32)
        # set zeros or ones when 2d data is missing
        voxel_pts_img_feat_src = torch.zeros((voxel_pts_coord_src.shape[0], 128), dtype=torch.float32).to(self.device)

        # get the indices of true values
        indices_true_src = torch.nonzero(mask_valid_pts_uv_src, as_tuple=False).squeeze().cpu()

        max_index = indices_true_src.size(0) - 1
        for n, i in enumerate(indices_true_src):
            # i.item() the indice of i
            idx = i.item()
            if idx <= max_index:
                voxel_pts_img_feat_src[i, :] = img_feat_src[:, proj_uv_src_2[i.item(), 0], proj_uv_src_2[i.item(), 1]]
        #####################

        #####################
        # tgt part
        tgt_pixel = camera_intrinsic_adjust @ tgt_pts_camera[:3, :]
        tgt_pixel_normalized = tgt_pixel / tgt_pixel[2, :]
        tgt_pts_u = tgt_pixel_normalized[0, :]
        tgt_pts_v = image_resize_after[1] - tgt_pixel_normalized[1, :]

        # may should be image_resize_after instead of image_resize
        mask_valid_pts_uv_tgt = ((tgt_pts_u > 0) & (tgt_pts_u < image_resize_after[0]) &
                                 (tgt_pts_v > 0) & (tgt_pts_v < image_resize_after[1]))
        # remove invalid pixels that exceed the image range
        tgt_pts_u = tgt_pts_u[mask_valid_pts_uv_tgt]
        tgt_pts_v = tgt_pts_v[mask_valid_pts_uv_tgt]

        proj_uv_tgt = torch.cat((tgt_pts_u[:, None], tgt_pts_v[:, None]), dim=1)
        proj_uv_tgt_2 = proj_uv_tgt.to(torch.int32)
        # set zeros or ones when 2d data is missing
        voxel_pts_img_feat_tgt = torch.zeros((voxel_pts_coord_tgt.shape[0], 128), dtype=torch.float32).to(self.device)

        # get the indices of true values
        indices_true_tgt = torch.nonzero(mask_valid_pts_uv_tgt, as_tuple=False).squeeze().cpu()

        max_index_tgt = indices_true_tgt.size(0) - 1
        for m, j in enumerate(indices_true_tgt):
            # i.item() the indice of i
            idx = j.item()
            if idx <= max_index_tgt:
                voxel_pts_img_feat_tgt[j, :] = img_feat_tgt[:, proj_uv_tgt_2[j.item(), 0], proj_uv_tgt_2[j.item(), 1]]
        #####################

        return voxel_pts_img_feat_src, voxel_pts_img_feat_tgt

    def _construct_img_patch(self):
        ####################
        # construct image patches before data augmentation
        # # 0.1 construct clusters
        # src_partition = torch.from_numpy(src_partition)
        # tgt_partition = torch.from_numpy(tgt_partition)
        #
        # # select the pts in the specific spt, the same used in the net (batch dim), no small patch removal
        # idx_pts2spt_src = src_partition[:, 6].to(torch.int64)
        # idx_pts2spt_tgt = tgt_partition[:, 6].to(torch.int64)
        # spt_src_id = torch.unique(idx_pts2spt_src[:], sorted=True, return_inverse=False)
        # spt_tgt_id = torch.unique(idx_pts2spt_tgt[:], sorted=True, return_inverse=False)
        # # Use vectorized operations to find where matches occur
        # mask_src = (idx_pts2spt_src[:].unsqueeze(0) == spt_src_id.unsqueeze(1))
        # idx_spt2pts_src = mask_src.nonzero(as_tuple=True)[1].split(mask_src.sum(dim=1).tolist())
        # mask_tgt = (idx_pts2spt_tgt[:].unsqueeze(0) == spt_tgt_id.unsqueeze(1))
        # idx_spt2pts_tgt = mask_tgt.nonzero(as_tuple=True)[1].split(mask_tgt.sum(dim=1).tolist())

        idx_spt2pts_src = self.data_interim.idx_spt2pts_src
        idx_spt2pts_tgt = self.data_interim.idx_spt2pts_tgt

        # 0.2 load img pose, intrinsic
        camera_intrinsic = self.data_input_2d.camera_intrinsic
        # src_extrinsic = self.data_input_2d.src_extrinsic.to('cpu')
        # tgt_extrinsic = self.data_input_2d.tgt_extrinsic.to('cpu')
        #
        # pos_src_1_rot, pose_src_1_tra = src_extrinsic[:3, :3], src_extrinsic[:3, 3]
        # pos_tgt_1_rot, pose_tgt_1_tra = tgt_extrinsic[:3, :3], tgt_extrinsic[:3, 3]

        color_src_1 = self.data_input_2d.src_image
        color_tgt_1 = self.data_input_2d.tgt_image

        src_pts = self.data_input_3d.src_pts
        tgt_pts = self.data_input_3d.tgt_pts

        # #################
        # # get the pts of first cluster for checking
        # for n, i in enumerate(idx_spt2pts_src[:10]):
        #     pts_cluster_0_src = src_pts[i, :]
        #     np.savetxt(osp.join('/scratch2/zhawang/projects/deformation/DeformHD_local/output/Brienz_TLS/2002_2011/ROI_7',
        #                         f'pts_cluster_{n}_src.txt'), pts_cluster_0_src.cpu())
        # #################

        src_pts_camera = (self.data_input_2d.src_extrinsic[:3, :3] @ src_pts.T +
                          self.data_input_2d.src_extrinsic[:3, 3][:, None]).T
        tgt_pts_camera = (self.data_input_2d.tgt_extrinsic[:3, :3] @ tgt_pts.T +
                          self.data_input_2d.tgt_extrinsic[:3, 3][:, None]).T

        img_patch_src, mask_valid_img_patch_src = construct_cluster_img_patch(idx_spt2pts_src, src_pts_camera,
                                                                              color_src_1, camera_intrinsic,
                                                                              enlarge_patch_size=self.method.enlarge_img_patch)
        img_patch_tgt, mask_valid_img_patch_tgt = construct_cluster_img_patch(idx_spt2pts_tgt, tgt_pts_camera,
                                                                              color_tgt_1, camera_intrinsic,
                                                                              enlarge_patch_size=self.method.enlarge_img_patch)

        return img_patch_src, img_patch_tgt, mask_valid_img_patch_src, mask_valid_img_patch_tgt

        ####################

    def _compute_spt_feat_and_coord_with_fused_patch_wise_img_feats(self):
        """ Compute the coordinates and features for each source superpoint and target superpoint """
        if self.verbose:
            self.logging.info(f'Start computing spt coordinates and features with fused features...')

        # mask_voxel_src = torch.zeros(len(self.data_interim.idx_pts2spt_src), dtype=bool).to(self.device)
        # mask_voxel_tgt = torch.zeros(len(self.data_interim.idx_pts2spt_tgt), dtype=bool).to(self.device)

        # mask_voxel_src[self.data_interim.idx_voxel2pts_src] = True
        # mask_voxel_tgt[self.data_interim.idx_voxel2pts_tgt] = True

        # get the entire img features, related to src, tgt pts indices
        # voxel_pts_img_feat_src, voxel_pts_img_feat_tgt = self._extract_img_pretrain_feats()

        dict_input = dict()
        dict_input['idx_pts2spt_src'] = self.data_interim.idx_pts2spt_src
        dict_input['idx_pts2spt_tgt'] = self.data_interim.idx_pts2spt_tgt
        dict_input['idx_pts2voxel_src'] = self.data_interim.idx_pts2voxel_src
        dict_input['idx_pts2voxel_tgt'] = self.data_interim.idx_pts2voxel_tgt
        dict_input['idx_spt2pts_src'] = self.data_interim.idx_spt2pts_src
        dict_input['idx_spt2pts_tgt'] = self.data_interim.idx_spt2pts_tgt

        img_patch_src, img_patch_tgt, mask_valid_img_patch_src, mask_valid_img_patch_tgt = (
            self._construct_img_patch())
        # add img patches
        dict_input['img_patch_src'] = img_patch_src
        dict_input['img_patch_tgt'] = img_patch_tgt
        dict_input['mask_valid_img_patch_src'] = mask_valid_img_patch_src
        dict_input['mask_valid_img_patch_tgt'] = mask_valid_img_patch_tgt

        # dict_input['src_partition'] = torch.ones((len(self.data_interim.idx_voxel2pts_src), 7), dtype=torch.int64).to(
        #     self.device)
        # dict_input['tgt_partition'] = torch.ones((len(self.data_interim.idx_voxel2pts_tgt), 7), dtype=torch.int64).to(
        #     self.device)
        # dict_input['src_partition'][:, 6] = self.data_interim.idx_pts2spt_src[self.data_interim.idx_voxel2pts_src]
        # dict_input['tgt_partition'][:, 6] = self.data_interim.idx_pts2spt_tgt[self.data_interim.idx_voxel2pts_tgt]
        #
        # dict_input['src_partition'] = dict_input['src_partition'][None, :, :]
        # dict_input['tgt_partition'] = dict_input['tgt_partition'][None, :, :]

        # voxel_feats_src = torch.cat((self.data_interim.tile_pts_sub_feat_src, voxel_pts_img_feat_src), dim=1)
        # voxel_feats_tgt = torch.cat((self.data_interim.tile_pts_sub_feat_tgt, voxel_pts_img_feat_tgt), dim=1)

        voxel_feats_src = self.data_interim.tile_pts_sub_feat_src
        voxel_feats_tgt = self.data_interim.tile_pts_sub_feat_tgt

        dict_input['src_feats'] = voxel_feats_src[None, :, :]
        dict_input['tgt_feats'] = voxel_feats_tgt[None, :, :]

        dict_input['src_pts'] = self.data_interim.src_pts_sub[None, :, :]
        dict_input['tgt_pts'] = self.data_interim.tgt_pts_sub[None, :, :]

        # need config initialize, define a function?
        # from src.feature_aggregation.cluster_feature_net_inference import ClusterFeatureNet
        # from src.feature_aggregation.cluster_feature_net_self_attention_with_img_patch_cross_attention_inference_combined import ClusterFeatureNetWithAttention
        # from src.feature_aggregation.old.cluster_feature_net_self_attention_with_img_patch_inference import ClusterFeatureNetWithAttention
        from src.feature_aggregation.cluster_feature_net_self_attention_with_patch_wise_fusion import ClusterFeatureNetWithAttention

        cfg = dict()
        cfg['mode'] = 'test'

        cfg['input_feat_dim'] = 64
        cfg['hidden_feat_dim'] = 64
        cfg['output_feat_dim'] = 64
        cfg['hidden_feat_dim1_patch_wise'] = 128
        cfg['hidden_feat_dim2_patch_wise'] = 128
        cfg['output_feat_dim_patch_wise'] = 64

        # # cfg['input_feat_dim'] = 192
        # cfg['input_feat_dim'] = 64
        # cfg['hidden_feat_dim'] = 64
        # # cfg['hidden_feat_dim1'] = 256
        # # cfg['hidden_feat_dim2'] = 128
        # cfg['output_feat_dim'] = 64
        # cfg['cross_atte_dim'] = 64

        cfg = edict(cfg)
        # using pooling+mlp
        # self.feat_aggregate_model = ClusterFeatureNet(cfg).to(self.device)
        # using self-attention + mlp

        # load pretrained model
        # weight_path = '/scratch2/zhawang/projects/deformation/DeformHD_local/weights/'
        # # weight_name = 'model_best_recall_0.075_0.1_0.1.pth'
        # state = torch.load(osp.join(weight_path, weight_name))

        weight_root = osp.join(self.config.path_name.project_dir, self.config.path_name.weight_dir,
                               self.config.path_name.pretrained_feature_aggregation_weight)
        state = torch.load(weight_root, weights_only=True)

        # load pretrained img model to cfg
        cfg['device'] = self.device
        cfg.model_img = PatchNetAutoencoder(64, normalize=True)

        if self.method.use_updated_lcd:
            #############
            # update patch wise model
            state_dict_img = state['state_dict_img']
            cfg.model_img.load_state_dict(state_dict_img, strict=True)
            #############
        else:
            ##############
            # no update for patch wise model
            state_dict_img = torch.load(self.config.path_name.pretrained_2d_weight)
            cfg.model_img.load_state_dict(state_dict_img['patchnet'], strict=True)
            ##############

        cfg.model_img.eval()

        self.feat_aggregate_model = ClusterFeatureNetWithAttention(cfg).to(self.device)
        self.feat_aggregate_model.load_state_dict(state['state_dict'])

        self.feat_aggregate_model.eval()
        with torch.no_grad():
            dict_output = self.feat_aggregate_model(dict_input)

            spt_feat_src = dict_output['spt_feat_src']
            spt_coord_src = dict_output['spt_coord_src']
            spt_feat_tgt = dict_output['spt_feat_tgt']
            spt_coord_tgt = dict_output['spt_coord_tgt']

            self.data_interim.spt_feat_src = spt_feat_src
            self.data_interim.spt_feat_tgt = spt_feat_tgt
            self.data_interim.spt_coord_src = spt_coord_src
            self.data_interim.spt_coord_tgt = spt_coord_tgt

        if self.verbose:
            self.logging.info(f'Computing superpoint coordinates and features is done! '
                              f'Num. of spt: {spt_feat_src.shape[0]} and {spt_feat_tgt.shape[0]}')

    def _compute_spt_feat_and_coord_with_fused_pixel_wise_img_feats(self):
        """ Compute the coordinates and features for each source superpoint and target superpoint """
        if self.verbose:
            self.logging.info(f'Start computing spt coordinates and features with fused features...')

        # mask_voxel_src = torch.zeros(len(self.data_interim.idx_pts2spt_src), dtype=bool).to(self.device)
        # mask_voxel_tgt = torch.zeros(len(self.data_interim.idx_pts2spt_tgt), dtype=bool).to(self.device)

        # mask_voxel_src[self.data_interim.idx_voxel2pts_src] = True
        # mask_voxel_tgt[self.data_interim.idx_voxel2pts_tgt] = True

        # get the entire img features, related to src, tgt pts indices
        voxel_pts_img_feat_src, voxel_pts_img_feat_tgt = self._extract_img_pretrain_feats()

        dict_input = dict()
        dict_input['idx_pts2spt_src'] = self.data_interim.idx_pts2spt_src
        dict_input['idx_pts2spt_tgt'] = self.data_interim.idx_pts2spt_tgt
        dict_input['idx_pts2voxel_src'] = self.data_interim.idx_pts2voxel_src
        dict_input['idx_pts2voxel_tgt'] = self.data_interim.idx_pts2voxel_tgt
        dict_input['idx_spt2pts_src'] = self.data_interim.idx_spt2pts_src
        dict_input['idx_spt2pts_tgt'] = self.data_interim.idx_spt2pts_tgt

        # img_patch_src, img_patch_tgt, mask_valid_img_patch_src, mask_valid_img_patch_tgt = (
        #     self._construct_img_patch())
        # # add img patches
        # dict_input['img_patch_src'] = img_patch_src
        # dict_input['img_patch_tgt'] = img_patch_tgt
        # dict_input['mask_valid_img_patch_src'] = mask_valid_img_patch_src
        # dict_input['mask_valid_img_patch_tgt'] = mask_valid_img_patch_tgt

        # dict_input['src_partition'] = torch.ones((len(self.data_interim.idx_voxel2pts_src), 7), dtype=torch.int64).to(
        #     self.device)
        # dict_input['tgt_partition'] = torch.ones((len(self.data_interim.idx_voxel2pts_tgt), 7), dtype=torch.int64).to(
        #     self.device)
        # dict_input['src_partition'][:, 6] = self.data_interim.idx_pts2spt_src[self.data_interim.idx_voxel2pts_src]
        # dict_input['tgt_partition'][:, 6] = self.data_interim.idx_pts2spt_tgt[self.data_interim.idx_voxel2pts_tgt]
        #
        # dict_input['src_partition'] = dict_input['src_partition'][None, :, :]
        # dict_input['tgt_partition'] = dict_input['tgt_partition'][None, :, :]

        # use voxel pts features + img pixel features
        voxel_feats_src = torch.cat((self.data_interim.tile_pts_sub_feat_src, voxel_pts_img_feat_src), dim=1)
        voxel_feats_tgt = torch.cat((self.data_interim.tile_pts_sub_feat_tgt, voxel_pts_img_feat_tgt), dim=1)

        # voxel_feats_src = self.data_interim.tile_pts_sub_feat_src
        # voxel_feats_tgt = self.data_interim.tile_pts_sub_feat_tgt

        dict_input['src_feats'] = voxel_feats_src[None, :, :]
        dict_input['tgt_feats'] = voxel_feats_tgt[None, :, :]

        dict_input['src_pts'] = self.data_interim.src_pts_sub[None, :, :]
        dict_input['tgt_pts'] = self.data_interim.tgt_pts_sub[None, :, :]

        # need config initialize, define a function?
        # from src.feature_aggregation.cluster_feature_net_inference import ClusterFeatureNet
        from src.feature_aggregation._OLD.cluster_feature_net_self_attention_with_pixel_wise_fusion import ClusterFeatureNetWithAttention
        # from src.feature_aggregation.cluster_feature_net_self_attention_with_img_patch_cross_attention_inference_combined import ClusterFeatureNetWithAttention

        cfg = dict()
        cfg['mode'] = 'test'

        cfg['input_feat_dim_pixel_wise'] = 192
        cfg['hidden_feat_dim1_pixel_wise'] = 128
        cfg['output_dim_self_atten'] = 128
        cfg['hidden_feat_dim2_pixel_wise'] = 128
        cfg['output_feat_dim_pixel_wise'] = 64

        cfg = edict(cfg)
        # using pooling+mlp
        # self.feat_aggregate_model = ClusterFeatureNet(cfg).to(self.device)
        # using self-attention + mlp

        # load pretrained feat aggregation model
        weight_root = osp.join(self.config.path_name.project_dir, self.config.path_name.weight_dir,
                               self.config.path_name.pretrained_feature_aggregation_weight)
        state = torch.load(weight_root)

        # weight_path = '/scratch2/zhawang/projects/deformation/DeformHD_local/weights/'
        # weight_name = 'model_best_recall_0.075_0.1_0.1.pth'

        # load pretrained img model to cfg
        cfg['device'] = self.device

        self.feat_aggregate_model = ClusterFeatureNetWithAttention(cfg).to(self.device)
        self.feat_aggregate_model.load_state_dict(state['state_dict'])

        self.feat_aggregate_model.eval()
        with torch.no_grad():
            dict_output = self.feat_aggregate_model(dict_input)

            spt_feat_src = dict_output['spt_feat_src']
            spt_coord_src = dict_output['spt_coord_src']
            spt_feat_tgt = dict_output['spt_feat_tgt']
            spt_coord_tgt = dict_output['spt_coord_tgt']

            self.data_interim.spt_feat_src = spt_feat_src
            self.data_interim.spt_feat_tgt = spt_feat_tgt
            self.data_interim.spt_coord_src = spt_coord_src
            self.data_interim.spt_coord_tgt = spt_coord_tgt

        if self.verbose:
            self.logging.info(f'Computing superpoint coordinates and features is done! '
                              f'Num. of spt: {spt_feat_src.shape[0]} and {spt_feat_tgt.shape[0]}')

    def _compute_spt_feat_and_coord_with_fused_feats(self):
        """ Compute the coordinates and features for each source superpoint and target superpoint """
        if self.verbose:
            self.logging.info(f'Start computing spt coordinates and features with fused features...')

        # mask_voxel_src = torch.zeros(len(self.data_interim.idx_pts2spt_src), dtype=bool).to(self.device)
        # mask_voxel_tgt = torch.zeros(len(self.data_interim.idx_pts2spt_tgt), dtype=bool).to(self.device)

        # mask_voxel_src[self.data_interim.idx_voxel2pts_src] = True
        # mask_voxel_tgt[self.data_interim.idx_voxel2pts_tgt] = True

        # get the entire img features, related to src, tgt pts indices
        # voxel_pts_img_feat_src, voxel_pts_img_feat_tgt = self._extract_img_pretrain_feats()

        dict_input = dict()
        dict_input['idx_pts2spt_src'] = self.data_interim.idx_pts2spt_src
        dict_input['idx_pts2spt_tgt'] = self.data_interim.idx_pts2spt_tgt
        dict_input['idx_pts2voxel_src'] = self.data_interim.idx_pts2voxel_src
        dict_input['idx_pts2voxel_tgt'] = self.data_interim.idx_pts2voxel_tgt
        dict_input['idx_spt2pts_src'] = self.data_interim.idx_spt2pts_src
        dict_input['idx_spt2pts_tgt'] = self.data_interim.idx_spt2pts_tgt

        # dict_input['src_partition'] = torch.ones((len(self.data_interim.idx_voxel2pts_src), 7), dtype=torch.int64).to(
        #     self.device)
        # dict_input['tgt_partition'] = torch.ones((len(self.data_interim.idx_voxel2pts_tgt), 7), dtype=torch.int64).to(
        #     self.device)
        # dict_input['src_partition'][:, 6] = self.data_interim.idx_pts2spt_src[self.data_interim.idx_voxel2pts_src]
        # dict_input['tgt_partition'][:, 6] = self.data_interim.idx_pts2spt_tgt[self.data_interim.idx_voxel2pts_tgt]
        #
        # dict_input['src_partition'] = dict_input['src_partition'][None, :, :]
        # dict_input['tgt_partition'] = dict_input['tgt_partition'][None, :, :]

        # voxel_feats_src = torch.cat((self.data_interim.tile_pts_sub_feat_src, voxel_pts_img_feat_src), dim=1)
        # voxel_feats_tgt = torch.cat((self.data_interim.tile_pts_sub_feat_tgt, voxel_pts_img_feat_tgt), dim=1)

        voxel_feats_src = self.data_interim.tile_pts_sub_feat_src
        voxel_feats_tgt = self.data_interim.tile_pts_sub_feat_tgt

        dict_input['src_feats'] = voxel_feats_src[None, :, :]
        dict_input['tgt_feats'] = voxel_feats_tgt[None, :, :]

        dict_input['src_pts'] = self.data_interim.src_pts_sub[None, :, :]
        dict_input['tgt_pts'] = self.data_interim.tgt_pts_sub[None, :, :]

        # need config initialize, define a function?
        # from src.feature_aggregation.old.cluster_feature_net_self_attention_inference import ClusterFeatureNetWithAttention
        from src.feature_aggregation.cluster_feature_net_self_attention import ClusterFeatureNetWithAttention

        cfg = dict()
        cfg['mode'] = 'test'
        # cfg['input_feat_dim'] = 192
        cfg['input_feat_dim'] = 64
        cfg['hidden_feat_dim'] = 64
        # cfg['hidden_feat_dim1'] = 256
        # cfg['hidden_feat_dim2'] = 128
        cfg['output_feat_dim'] = 64
        cfg = edict(cfg)
        # using pooling+mlp
        # self.feat_aggregate_model = ClusterFeatureNet(cfg).to(self.device)
        # using self-attention + mlp
        # self.feat_aggregate_model = ClusterFeatureNetWithAttention(cfg).to(self.device)

        # load pretrained feat aggregation model
        weight_root = osp.join(self.config.path_name.project_dir, self.config.path_name.weight_dir,
                               self.config.path_name.pretrained_feature_aggregation_weight)
        state = torch.load(weight_root)

        # load pretrained model
        # weight_path = '/scratch2/zhawang/projects/deformation/DeformHD_local/weights/'
        # weight_name = 'model_best_recall_margin_no_change_0.075_0.1_0.1_longer_epochs.pth'

        cfg['device'] = self.device

        self.feat_aggregate_model = ClusterFeatureNetWithAttention(cfg).to(self.device)
        self.feat_aggregate_model.load_state_dict(state['state_dict'])

        # state = torch.load(osp.join(weight_path, weight_name))
        # self.feat_aggregate_model.load_state_dict(state['state_dict'])

        self.feat_aggregate_model.eval()
        with torch.no_grad():
            dict_output = self.feat_aggregate_model(dict_input)

            spt_feat_src = dict_output['spt_feat_src']
            spt_coord_src = dict_output['spt_coord_src']
            spt_feat_tgt = dict_output['spt_feat_tgt']
            spt_coord_tgt = dict_output['spt_coord_tgt']

            self.data_interim.spt_feat_src = spt_feat_src
            self.data_interim.spt_feat_tgt = spt_feat_tgt
            self.data_interim.spt_coord_src = spt_coord_src
            self.data_interim.spt_coord_tgt = spt_coord_tgt

        if self.verbose:
            self.logging.info(f'Computing superpoint coordinates and features is done! '
                              f'Num. of spt: {spt_feat_src.shape[0]} and {spt_feat_tgt.shape[0]}')

    def implement_partition(self):

        partition = self.method.partition
        if not partition:
            self.logging.info('Skip the partition process. The partition result will be loaded from path.')
        elif self.method.partition_type in ['supervoxel']:
            # can be incorporated with other types, e.g., hdbscan
            if self.verbose:
                self.logging.info(f'Start partition with {self.method.partition_type}...')

            svl_radius = np.max((np.sqrt(3) * (10 * self._compute_median_resolution()), self.method.voxel_size))
            # svl_radius = svl_radius * 3
            if self.data.dataset == 'rockfall_simulator':
                svl_radius = 0.1

            partition_path = osp.join(self.output_root, f'{self.method.partition_type}_partition')
            dir_exist(partition_path)

            if self.method.partition_type == 'supervoxel':
                # TODO: segment_id_src_pts_input may not needed
                # if 'full' in self.input_root:
                if self.data.multiple_case:
                    segment_id_src_pts_input = supervoxel_partition.computeSupervoxel(
                        self.src_pcd_path, self.para.n_normals, svl_radius,
                        osp.join(partition_path, f'partition_of_input_src_tile_{self.config.tile_id}.txt'))

                    segment_id_tgt_pts_input = supervoxel_partition.computeSupervoxel(
                        self.tgt_pcd_path, self.para.n_normals, svl_radius,
                        osp.join(partition_path, f'partition_of_input_tgt_tile_{self.config.tile_id}.txt'))
                else:
                    segment_id_src_pts_input = supervoxel_partition.computeSupervoxel(
                        self.src_pcd_path, self.para.n_normals, svl_radius,
                        osp.join(partition_path, 'partition_of_input_src.txt'))

                    segment_id_tgt_pts_input = supervoxel_partition.computeSupervoxel(
                        self.tgt_pcd_path, self.para.n_normals, svl_radius,
                        osp.join(partition_path, 'partition_of_input_tgt.txt'))

                # type for each pts in input src pts
                # segment_id_src_pts_input = np.asarray(segment_id_src_pts_input).reshape(-1, 1)
                # segment_id_tgt_pts_input = np.asarray(segment_id_tgt_pts_input).reshape(-1, 1)

        elif self.method.partition_type == 'superpoint':
            if self.verbose:
                self.logging.info(f'Start partition with {self.method.partition_type}...')

            partition_path = osp.join(self.output_root, f'{self.method.partition_type}_partition')
            dir_exist(partition_path)

            # from superpoint_transformer.src.utils import init_config
            # from superpoint_transformer.src.transforms.data import NAGAddKeysTo

            if self.data.multiple_case:
            # if 'full' in self.input_root:
                superpoint_partition(self.config.tile_id, partition_path, self.logging, device=self.device)
            else:
                raise NotImplementedError

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
        neigh.fit(np.asarray(self.data_interim.src_pts_sub.cpu()))
        dist01, _ = neigh.kneighbors(np.asarray(self.data_interim.src_pts_sub.cpu()), return_distance=True)

        # Resolution of the source point cloud
        resolution_s = np.median(dist01[:, -1])

        # Compute the point cloud resolution of the target point cloud (k=2 as the closest point is the point itself)
        neigh.fit(np.asarray(self.data_interim.tgt_pts_sub.cpu()))
        dist01, _ = neigh.kneighbors(np.asarray(self.data_interim.tgt_pts_sub.cpu()), return_distance=True)

        # Resolution of the target point cloud
        resolution_t = np.median(dist01[:, -1])

        # end_time = time.time()

        # if args.verbose:
        #     args.logging.info(
        #         '2.3 Median point cloud resolution of the tiles computed in: {:.2f} s'.format(end_time - start_time))

        # Lower of the both point cloud resolutions (i.e. larger median distance to the closest points)

        tile_st_resolution = max(resolution_s, resolution_t)

        # only count for current tile
        self.para.median_max_resolution = tile_st_resolution

        return tile_st_resolution

    def global_matches_from_3d(self):
        if self.verbose:
            self.logging.info('Start global matches from 3d...')

        search_type = self.method.global_matching_from_3d_type
        # TODO: other version of knn based on nanoflann, or FRNN
        if search_type == 'hnsw':
            ###############
            # implement knn for all
            # too expensive for the memory
            # corres_3d = torch.cdist(self.data_interim.tile_pts_sub_feat_src, self.data_interim.tile_pts_sub_feat_tgt)

            # Intitialize the library, specify the space, the type of the vector and add data points
            p = hnswlib.Index(space='l2', dim=64)  # possible options are l2, cosine or ip
            p.init_index(max_elements=2*self.data_interim.tile_pts_sub_feat_src.shape[0],
                         ef_construction=self.para.efC, M=self.para.M)
            p.set_ef(self.para.efS)
            p.set_num_threads(self.para.num_threads)
            p.add_items(self.data_interim.tile_pts_sub_feat_tgt.cpu().numpy())

            # Query the elements for themselves and measure recall:
            # knn based correspondence searching
            labels, distances = p.knn_query(self.data_interim.tile_pts_sub_feat_src.cpu().numpy(), k=1)

            # voxel_labels = labels.reshape(-1)
            # voxel_labels = voxel_labels.astype(np.int64)
            # voxel_labels = torch.from_numpy(voxel_labels).to(self.device)
        elif search_type == 'cdist_cpu':
            # Assume src and tgt features are tensors on the same device (CPU/GPU)
            src_features = self.data_interim.tile_pts_sub_feat_src.cpu()
            tgt_features = self.data_interim.tile_pts_sub_feat_tgt.cpu()

            # Calculate pairwise distances and get the indices of nearest neighbors
            distances = torch.cdist(src_features, tgt_features)  # L2 distance by default
            _, labels = distances.min(dim=1)  # Find the closest match in tgt for each src point
        elif search_type == 'cdist':
            # knn matching results are not the same
            src_features = self.data_interim.tile_pts_sub_feat_src
            tgt_features = self.data_interim.tile_pts_sub_feat_tgt

            batch_size = 1024  # Adjust based on available GPU memory
            num_src = src_features.shape[0]
            # num_tgt = tgt_features.shape[0]

            # Preallocate storage for nearest neighbor indices and distances
            labels = torch.empty(num_src, dtype=torch.long, device=src_features.device)
            distances = torch.empty(num_src, device=src_features.device)

            # Iterate over batches of src_features only
            for i in range(0, num_src, batch_size):
                src_batch = src_features[i:i + batch_size]
                # Compute pairwise distances between the src_batch and all tgt_features
                dist_batch = torch.cdist(src_batch, tgt_features)  # (batch_size, num_tgt)

                # Get the minimum distances and indices for the current batch
                min_dist, min_idx = dist_batch.min(dim=1)

                # Store results in the preallocated arrays
                labels[i:i + batch_size] = min_idx
                distances[i:i + batch_size] = min_dist

            # voxel_labels = labels
            a = 0
        elif search_type == 'faiss':
            import faiss

            # **FAISS KNN Search**
            src_features = self.data_interim.tile_pts_sub_feat_src.cpu().numpy().astype(np.float32)
            tgt_features = self.data_interim.tile_pts_sub_feat_tgt.cpu().numpy().astype(np.float32)

            dim = src_features.shape[1]  # 64D feature space

            # TODO: unnecessary
            # Use GPU if available
            use_gpu = torch.cuda.is_available()

            if use_gpu:
                # res = faiss.StandardGpuResources()
                # index = faiss.IndexFlatL2(dim)  # L2 distance metric
                # index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU

                index = faiss.IndexHNSWFlat(dim, 32)  # HNSW graph with 32 neighbors
                index.hnsw.efSearch = 64  # Increase search quality

            else:
                index = faiss.IndexFlatL2(dim)  # CPU version

            index.add(tgt_features)  # Add target features to FAISS index

            batch_size = 1024  # Set batch size to prevent memory overflow
            num_samples = src_features.shape[0]  # Total number of queries

            all_labels = []
            all_distances = []

            # Perform batch-wise search to reduce memory usage
            for i in range(0, num_samples, batch_size):
                batch_src = src_features[i: i + batch_size]  # Extract batch
                distances, labels = index.search(batch_src, 1)  # K=1 nearest neighbor search
                all_labels.append(labels)
                all_distances.append(distances)

            # Concatenate results
            labels = np.concatenate(all_labels, axis=0).flatten()
            distances = np.concatenate(all_distances, axis=0).flatten()

            # Convert results to PyTorch tensors
            labels = torch.from_numpy(labels).to(self.device)
            distances = torch.from_numpy(distances).to(self.device)
        else:
            raise NotImplementedError(f"Method {search_type} is not implemented")

        # remove the voxel matches that exceed the max. mag.
        # is better than doing it latter, because voxel2pts will assign multiple voxels to a same point, but only one
        # will be used. So remove outlier voxel matches in advance is better

        voxel_corres_src = self.data_interim.src_pts_sub
        # voxel_corres_tgt = self.data_interim.tgt_pts_sub[voxel_labels]
        voxel_corres_tgt = self.data_interim.tgt_pts_sub[labels]
        magnitude = torch.linalg.norm(voxel_corres_src - voxel_corres_tgt, axis=1).to(self.device)
        mask_within_max_mag = magnitude <= self.para.max_magnitude

        # keep the indices of unmatched pts as -1
        corres_3d_voxel_from_3d_idx = - torch.ones(self.data_input_3d.idx_initial_src.shape[0], 2).to(torch.int64).to(
            self.device)
        corres_3d_voxel_from_3d_idx[:, 0] = torch.arange(self.data_input_3d.idx_initial_src.shape[0])
        # some voxels relate to the same src
        corres_3d_voxel_from_3d_idx[self.data_interim.idx_voxel2pts_src[mask_within_max_mag.cpu()], 1] = (
            torch.tensor(self.data_interim.idx_voxel2pts_tgt[labels.cpu()[mask_within_max_mag.cpu()]],
                         dtype=torch.int64).to(self.device))

        # mask_voxel = torch.where(corres_3d_voxel_from_3d_idx[:, 1] != -1)[0]

        self.data_interim.corres_3d_voxel_from_3d_idx = corres_3d_voxel_from_3d_idx
        ###############

        ##############
        # TODO: can be merged to the final output function
        # output the original 3d matches from 2d or 3d
        # mask_2d_temp = self.data_interim.corres_3d_from_2d_idx[:, 1] > -1

        # for better visualization in CloudCompare
        # only for visualization, not for real computation; can be commented for faster computation
        if self.data.dataset == 'rockfall_simulator':
            max_mag_visualize = 0.06
        elif self.data.dataset == 'brienz_tls':
            max_mag_visualize = 5
        elif self.data.dataset == 'mattertal':
            max_mag_visualize = 10
        else:
            max_mag_visualize = 10

        dvfs_3d_voxel_vector_mag = torch.hstack((voxel_corres_src[mask_within_max_mag], magnitude[mask_within_max_mag, None]))
        dvfs_3d_voxel_vector_mag[0, 3] = 0
        dvfs_3d_voxel_vector_mag[1, 3] = max_mag_visualize

        dir_exist(osp.join(self.output_root, 'results'))
        if self.data.multiple_case:
        # if 'full' in self.output_root:
            np.savetxt(osp.join(self.output_root, 'results', f'c2f_dvfms_from_global_3d_src2tgt_wo_pruning_visualize_tile_{self.config.tile_id}.txt'),
                       dvfs_3d_voxel_vector_mag.cpu())
        else:
            np.savetxt(osp.join(self.output_root, 'results',
                                'c2f_dvfms_from_global_3d_src2tgt_wo_pruning_visualize.txt'),
                       dvfs_3d_voxel_vector_mag.cpu())

        self.logging.info('Global point matches from 3d is done!')
        ##############

    def coarse_matching_with_different_types(self):
        ############
        # coarse / patch matching, parameters are given in config file
        # only 3d, only 2d, or fusion
        # 3d types, averaging, learning-based pure 3D, learning-based 3D + patch_wise, learning-based 3D + pixel_wise
        # mutual nn, or single nn, or intersection nn?
        # 2D types, statistic conuting?
        ############

        # follow coarse_matching_2d
        # aggregate features and coordinates, then match clusters
        # can be replaced by network learning? also replace the matching by learning-based optimal transport?
        if self.verbose:
            self.logging.info('Start coarse matching...')
        # first construct cluster features and coordinates
        # then do cluster matching

        if self.method.coarse_matching_fusion or self.method.coarse_matching_only_3d:
            ################
            # 3d method
            # first construct cluster features and coordinates, then do cluster matching

            if self.method.feat_aggregate_type == 'learning_based':
                # self._compute_spt_feat_and_coord()

                if self.method.use_img_patch_enhanced_3d_aggregation:
                    self._compute_spt_feat_and_coord_with_fused_patch_wise_img_feats()
                elif self.method.use_img_pixel_enhanced_3d_aggregation:
                    self._compute_spt_feat_and_coord_with_fused_pixel_wise_img_feats()
                elif self.method.use_normal_3d_aggregation:
                    # need to check
                    # pure 3D learning based
                    self._compute_spt_feat_and_coord_with_fused_feats()
                else:
                    raise NotImplementedError
            else:
                # non-learning based, averaging method
                raise NotImplementedError

            # compute spt feat and coord, then establish cluster matches
            # and remove these above max. mag
            dist_coord = torch.cdist(self.data_interim.spt_coord_src, self.data_interim.spt_coord_tgt)
            mask_dist_above_max_mag = dist_coord > self.para.max_magnitude
            dist_feat = torch.cdist(self.data_interim.spt_feat_src, self.data_interim.spt_feat_tgt)
            dist_feat[mask_dist_above_max_mag] = torch.inf

            # get the nearest indices
            spt_corres_dist_and_tgt = torch.min(dist_feat, dim=1)

            # not implemented
            # or self.method.coarse_refinement_3d_type == 'nn_union'

            if self.method.coarse_refinement_3d_type == 'nn_mutual':
                # mutual selection
                spt_corres_dist_and_src = torch.min(dist_feat, dim=0)

                # only indices are equal from mutual selection are valid
                mask_mutual = torch.zeros(dist_feat.shape[0], dtype=torch.bool).to(self.device)
                for m, i in enumerate(spt_corres_dist_and_tgt[1]):
                    mask_temp = spt_corres_dist_and_src[1][i] == m
                    if mask_temp:
                        mask_mutual[m] = True

            # [0]: feat dist, [1]: indices
            mask_in_mag = spt_corres_dist_and_tgt[0] < torch.inf

            if self.method.coarse_refinement_3d_type == 'only_max_mag':
                mask = mask_in_mag
            elif self.method.coarse_refinement_3d_type == 'nn_mutual':
                # consider mutual selection or not
                mask = mask_mutual & mask_in_mag
            elif self.method.coarse_refinement_3d_type == 'nn_union':
                raise NotImplementedError
            else:
                raise NotImplementedError

            # already local indices, which mean the indices correspond to cluster pts
            idx_corres_spt_tgt_local = spt_corres_dist_and_tgt[1][mask]
            # spt_corres_tgt = [self.data_interim.idx_spt2pts_tgt[i] for i in idx_corres_spt_tgt_local]

            # spt_corres_src = self.data_interim.idx_spt2pts_src
            # only mutual selected spt corres are used for fine matching
            spt_corres_src_3d = [self.data_interim.idx_spt2pts_src[i] for i in torch.where(mask)[0]]

            # no invalid mat    ches, because all src cluster can find one, maybe can filter some of them
            # spt_corres_src = [tensor for tensor, is_valid in zip(spt_corres_src, mask) if is_valid]
            spt_corres_tgt_3d = [self.data_interim.idx_spt2pts_tgt[i] for i in idx_corres_spt_tgt_local]
            ################

            self.logging.info('Coarse matching from 3D source is done!')

        if self.method.coarse_matching_fusion or self.method.coarse_matching_only_2d:
            ################
            # 2d method
            idx_corres_spt_tgt = []
            for pts_idx_temp in tqdm(self.data_interim.idx_spt2pts_src, position=0, leave=True):
                idx_tgt_pts = self.data_interim.corres_3d_from_2d_idx[pts_idx_temp, 1]
                # make sure only tgt pts of valid spt is selected. idx == -1 means invalid
                mask_non_neg_ones = torch.where(idx_tgt_pts >= 0)
                idx_tgt_spt = self.data_interim.idx_pts2spt_tgt[idx_tgt_pts[mask_non_neg_ones]]

                # find the most counted tgt spt indice for each src spt
                unique_elements, counts = torch.unique(idx_tgt_spt, return_counts=True)
                sorted_indices = torch.argsort(counts, descending=True)
                # only consider the first
                idx_corres_spt_tgt_temp = unique_elements[sorted_indices[:1]]
                idx_corres_spt_tgt.append(idx_corres_spt_tgt_temp)

                # counts_tgt = Counter(np.asarray(idx_tgt_spt.cpu()))
                # valid_spt_src = {idx for idx, count in counts.items() if count > 10 and idx != -1}
                # valid_spt_src = {idx for idx, count in counts_src.items() if count > 10}

            # get the 1-1 spt correspondences
            spt_corres_src = self.data_interim.idx_spt2pts_src

            # for some src spt, no valid 2d matches exist; therefore, no corresponding tgt spt will be found
            idx_empty_spt_tgt = np.asarray([i for i, tensor in enumerate(idx_corres_spt_tgt) if tensor.numel() == 0])
            mask_non_empty_spt_tgt = torch.ones(len(spt_corres_src), dtype=bool)
            mask_non_empty_spt_tgt[idx_empty_spt_tgt] = False

            # only select the valid src spt, which means only valid matches exist in both src and tgt spt
            # for constructing the spt matches
            spt_corres_src_2d = [tensor for tensor, is_valid in zip(spt_corres_src, mask_non_empty_spt_tgt) if is_valid]

            # cluster id in idx_spt_src or idx_spt_tgt doesn't totally equal to the indice
            # because some small clusters are removed,
            # the cluster id is equal to idx_pts2spt
            idx_corres_spt_tgt_global = torch.cat(idx_corres_spt_tgt)

            # use the local to get cluster pts from idx_spt2pts
            # because the idx_spt_src has the same indices as idx_spt2pts_src
            # quite expensive, why? what is the goal
            # idx_corres_spt_tgt_local = torch.tensor([self.data_interim.idx_spt_tgt.index(x.item())
            #                                          for x in idx_corres_spt_tgt_global])

            # idx_spt_tgt_tensor = torch.stack(self.data_interim.idx_spt_tgt)
            # meaning
            matches = (self.data_interim.idx_spt_tgt == idx_corres_spt_tgt_global[:, None])  # Broadcasting to compare
            row_mask = matches.any(dim=1) # deal with the issue when all values of one row are False
            valid_indices = torch.nonzero(row_mask, as_tuple=True)[0].tolist()
            spt_corres_src_2d = [spt_corres_src_2d[i] for i in valid_indices]

            idx_corres_spt_tgt_local = matches.nonzero(as_tuple=True)[1]  # Extract matching indices

            # use the local indices (the ones correspond to cluster pts)
            spt_corres_tgt_2d = [self.data_interim.idx_spt2pts_tgt[i] for i in idx_corres_spt_tgt_local]

            # save corres_src from 3d and 2d, respectively, for debugging
            # a = torch.cat(spt_corres_src_2d)

            self.logging.info('Coarse matching from 2D source is done!')
            ################

        if self.method.coarse_matching_only_2d and self.method.fine_matching_only_2d:
            ################
            # extra 3d method
            idx_corres_spt_tgt = []
            for pts_idx_temp in tqdm(self.data_interim.idx_spt2pts_src, position=0, leave=True):
                idx_tgt_pts = self.data_interim.corres_3d_voxel_from_3d_idx[pts_idx_temp, 1]
                # make sure only tgt pts of valid spt is selected. idx == -1 means invalid
                mask_non_neg_ones = torch.where(idx_tgt_pts >= 0)
                idx_tgt_spt = self.data_interim.idx_pts2spt_tgt[idx_tgt_pts[mask_non_neg_ones]]

                # find the most counted tgt spt indice for each src spt
                unique_elements, counts = torch.unique(idx_tgt_spt, return_counts=True)
                sorted_indices = torch.argsort(counts, descending=True)
                # only consider the first
                idx_corres_spt_tgt_temp = unique_elements[sorted_indices[:1]]
                idx_corres_spt_tgt.append(idx_corres_spt_tgt_temp)

            # get the 1-1 spt correspondences
            spt_corres_src = self.data_interim.idx_spt2pts_src

            # for some src spt, no valid 2d matches exist; therefore, no corresponding tgt spt will be found
            idx_empty_spt_tgt = np.asarray([i for i, tensor in enumerate(idx_corres_spt_tgt) if tensor.numel() == 0])
            mask_non_empty_spt_tgt = torch.ones(len(spt_corres_src), dtype=bool)
            mask_non_empty_spt_tgt[idx_empty_spt_tgt] = False

            # only select the valid src spt, which means only valid matches exist in both src and tgt spt
            # for constructing the spt matches
            spt_corres_src_3d_extra = [tensor for tensor, is_valid in zip(spt_corres_src, mask_non_empty_spt_tgt) if is_valid]

            # cluster id in idx_spt_src or idx_spt_tgt doesn't totally equal to the indice
            # because some small clusters are removed,
            # the cluster id is equal to idx_pts2spt
            idx_corres_spt_tgt_global = torch.cat(idx_corres_spt_tgt)

            # idx_spt_tgt_tensor = torch.stack(self.data_interim.idx_spt_tgt)
            matches = (self.data_interim.idx_spt_tgt == idx_corres_spt_tgt_global[:, None])  # Broadcasting to compare
            row_mask = matches.any(dim=1) # deal with the issue when all values of one row are False
            valid_indices = torch.nonzero(row_mask, as_tuple=True)[0].tolist()
            spt_corres_src_3d_extra = [spt_corres_src_3d_extra[i] for i in valid_indices]

            idx_corres_spt_tgt_local = matches.nonzero(as_tuple=True)[1]  # Extract matching indices

            # use the local indices (the ones correspond to cluster pts)
            spt_corres_tgt_3d_extra = [self.data_interim.idx_spt2pts_tgt[i] for i in idx_corres_spt_tgt_local]

            # save corres_src from 3d and 2d, respectively, for debugging
            # a = torch.cat(spt_corres_src_2d)

            self.logging.info('Coarse matching from extra 3D source is done!')
            ################

        # record the number of each type of spt matches
        self.spt_length = []
        if self.method.coarse_matching_only_3d:
            spt_corres_src = copy.deepcopy(spt_corres_src_3d)
            spt_corres_tgt = copy.deepcopy(spt_corres_tgt_3d)
            self.spt_length.append(len(spt_corres_src_3d))
        elif self.method.coarse_matching_only_2d:
            spt_corres_src = copy.deepcopy(spt_corres_src_2d)
            spt_corres_tgt = copy.deepcopy(spt_corres_tgt_2d)
            self.spt_length.append(len(spt_corres_src_2d))
        elif self.method.coarse_matching_fusion:
            # naively combine two types of cluster matches
            spt_corres_src = copy.deepcopy(spt_corres_src_2d)
            spt_corres_tgt = copy.deepcopy(spt_corres_tgt_2d)
            spt_corres_src.extend(spt_corres_src_3d)
            spt_corres_tgt.extend(spt_corres_tgt_3d)
            self.spt_length.append(len(spt_corres_src_2d))
            self.spt_length.append(len(spt_corres_src_3d))
        else:
            raise NotImplementedError

        if self.method.coarse_matching_only_2d and self.method.fine_matching_only_2d:
            spt_corres_src.extend(spt_corres_src_3d_extra)
            spt_corres_tgt.extend(spt_corres_tgt_3d_extra)
            self.spt_length.append(len(spt_corres_tgt_3d_extra))

        # record the output from coarse matching to 'data_output' dict
        self.data_output.spt_corres_src = spt_corres_src
        self.data_output.spt_corres_tgt = spt_corres_tgt

        if self.visualize.visualize_patch:
            if self.visualize.random_choice:
                random_idx = np.random.choice(np.arange(len(self.data_output.spt_corres_src)),
                                              self.visualize.num_of_visualize_samples, replace=False)
            else:
                random_idx = np.arange(self.visualize.num_of_visualize_samples)
            for num in random_idx:
                pts_idx_in_curr_spt_src = self.data_output.spt_corres_src[num]
                pts_idx_in_curr_spt_tgt = self.data_output.spt_corres_tgt[num]

                pts_coord_in_curr_spt_src = self.data_input_3d.src_pts[pts_idx_in_curr_spt_src]
                pts_coord_in_curr_spt_tgt = self.data_input_3d.tgt_pts[pts_idx_in_curr_spt_tgt]

                coord_spt_src = torch.mean(pts_coord_in_curr_spt_src, dim=0)
                coord_spt_tgt = torch.mean(pts_coord_in_curr_spt_tgt, dim=0)

                small_region = self.visualize.small_region

                # crop small regions
                small_base_pts_src, small_base_pts_tgt = self._crop_small_point_cloud_for_visualization(
                    current_spt_pts_src=coord_spt_src,
                    current_spt_pts_tgt=coord_spt_tgt,
                    small_region=small_region)

                src_color = [0.921, 0.569, 0]
                tgt_color = [0, 0.839, 1.]
                red_color = [1, 0, 0]

                offset = np.asarray(self.visualize.offset)
                tgt_pts_temp = copy.deepcopy(pcd2tensor(self.data_input_3d.tgt_pcd)).cpu()
                tgt_pts_temp += offset

                pts_coord_in_curr_spt_tgt_temp = copy.deepcopy(pts_coord_in_curr_spt_tgt).cpu()
                pts_coord_in_curr_spt_tgt_temp += offset

                # o3d.visualization.draw_geometries([self.data_input_3d.src_pcd, tensor2pcd(pts_coord_in_curr_spt_src)])
                # o3d.visualization.draw_geometries([self.data_input_3d.tgt_pcd, tensor2pcd(pts_coord_in_curr_spt_tgt)])

                if self.data.dataset == 'rockfall_simulator':
                    o3d.visualization.draw_geometries([self.data_input_3d.src_pcd.voxel_down_sample(self.para.voxel_size),
                                                       tensor2pcd(tgt_pts_temp, colors=
                                                       self.data_input_3d.tgt_clr).voxel_down_sample(self.para.voxel_size),
                                                       tensor2pcd(pts_coord_in_curr_spt_src),
                                                       tensor2pcd(pts_coord_in_curr_spt_tgt_temp)],
                    front=[0.803380360985678, -0.58209200469215017, 0.1254945961226844],
                    lookat=[-1.6592649221420288, 0.94512602873146534, -0.044175013899803162],
                    up=[-0.20603508866432135, -0.073998807399239225, 0.9757426498532159],
                    zoom=0.69999999999999996
                    )

                else:
                    patch_src = tensor2pcd(pts_coord_in_curr_spt_src)
                    patch_tgt = tensor2pcd(pts_coord_in_curr_spt_tgt_temp)
                    patch_src.paint_uniform_color([1, 0, 0])
                    patch_tgt.paint_uniform_color([1, 0, 0])

                    o3d.visualization.draw_geometries([self.data_input_3d.src_pcd.voxel_down_sample(self.method.voxel_size),
                                                       tensor2pcd(tgt_pts_temp, colors=
                                                       self.data_input_3d.tgt_clr).voxel_down_sample(self.method.voxel_size),
                                                       patch_src,
                                                       patch_tgt],
                    )

                    # small_pcd_src = tensor2pcd(small_base_pts_src)
                    # small_pcd_tgt = tensor2pcd(small_base_pts_tgt)
                    # patch_src.paint_uniform_color([1, 0, 0])
                    # patch_tgt.paint_uniform_color([1, 0, 0])
                    #
                    # o3d.visualization.draw_geometries([tensor2pcd(small_base_pts_src),
                    #                                    tensor2pcd(small_base_pts_tgt),
                    #                                    patch_src,
                    #                                    patch_tgt],
                    # )

        if self.verbose:
            self.logging.info('Coarse matching is done!')

    def fine_matching_with_different_types(self):
        if self.verbose:
            self.logging.info('Start fine matching...')

        # voxel_corres_matrix = torch.cdist(voxel_feats_in_spt_src, voxel_feats_in_spt_tgt)

        # src2tgt
        corres_3d_refine_apply_icp = []
        # tgt2src
        corres_3d_refine_apply_icp_tgt2src = []
        # src2tgt, sparse
        corres_3d_refine_apply_icp_discrete = []

        mask_spt_match_useful = torch.ones(len(self.data_output.spt_corres_src[:]), dtype=torch.bool)
        mask_spt_match_global = torch.ones(len(self.data_output.spt_corres_src[:]), dtype=torch.bool)
        metric_values = []

        # processing for each src-tgt patch match
        for i, pts_idx_in_curr_spt_src in enumerate(tqdm(self.data_output.spt_corres_src[:], position=0, leave=True)):
            pts_idx_in_curr_spt_tgt = self.data_output.spt_corres_tgt[i]

            if self.method.fine_matching_fusion or self.method.fine_matching_only_3d:
                ##########
                idx_corres_from_3d_in_curr_spt_src = self.data_interim.corres_3d_voxel_from_3d_idx[pts_idx_in_curr_spt_src]
                mask_3d = torch.isin(idx_corres_from_3d_in_curr_spt_src[:, 1], pts_idx_in_curr_spt_tgt).to(torch.bool)
                idx_corres_from_3d_in_both_spt = idx_corres_from_3d_in_curr_spt_src[mask_3d]
            if self.method.fine_matching_fusion or self.method.fine_matching_only_2d:
                ##########
                idx_corres_from_2d_in_curr_spt_src = self.data_interim.corres_3d_from_2d_idx[pts_idx_in_curr_spt_src]
                # find the indices of tgt from src that exists in both tgt spt as well
                mask_2d = torch.isin(idx_corres_from_2d_in_curr_spt_src[:, 1], pts_idx_in_curr_spt_tgt).to(torch.bool)
                idx_corres_from_2d_in_both_spt = idx_corres_from_2d_in_curr_spt_src[mask_2d]

            if self.method.fine_matching_only_3d:
                idx_corres_in_both_spt = idx_corres_from_3d_in_both_spt
            elif self.method.fine_matching_only_2d:
                idx_corres_in_both_spt = idx_corres_from_2d_in_both_spt
            elif self.method.fine_matching_fusion:
                idx_corres_in_both_spt = torch.cat((idx_corres_from_3d_in_both_spt, idx_corres_from_2d_in_both_spt))
            else:
                raise NotImplementedError
            #################
            # remove redundant point-point matches? how? necessary?
            # current version, using only 3d point matches, or 2d+3d point matches?
            # idx_corres_in_both_spt = idx_corres_from_3d_in_both_spt

            if self.method.weighting_svd:
                # calculate the weight values
                if idx_corres_in_both_spt.shape[0] > 0:
                    # weight values of 3D matches
                    weight_value = (idx_corres_from_3d_in_both_spt.shape[0] /
                                    (idx_corres_from_3d_in_both_spt.shape[0] + idx_corres_from_2d_in_both_spt.shape[0]))
                    weight_vector =torch.ones(idx_corres_in_both_spt.shape[0]).to(self.device)

                    weight_vector[:idx_corres_from_3d_in_both_spt.shape[0]] = weight_value
                    weight_vector[idx_corres_from_2d_in_both_spt.shape[0]:] = 1 - weight_value
                    ####
                    # set all 2d match weight values to 0 for testing
                    weight_vector[idx_corres_from_2d_in_both_spt.shape[0]:] = 0.01
            else:
                weight_vector = None

            if self.method.fine_matching_fusion or self.method.fine_matching_only_2d or self.method.fine_matching_only_3d:
                if self.method.remove_low_quality_patch_matches:
                    # minimum 10 matches then do the refinement, otherwise it will be filtered out anyway
                    if idx_corres_in_both_spt.shape[0] >= self.method.num_min_matches_for_quality_check:
                        # appy hausdorff dist., before estimation or after (initial) estimation?
                        # idx_corres_in_both_spt = idx_corres_from_2d_in_both_spt
                        pts_coord_in_curr_spt_src_mutual = self.data_input_3d.src_pts[idx_corres_in_both_spt[:, 0]]
                        pts_coord_in_curr_spt_tgt_mutual = self.data_input_3d.tgt_pts[idx_corres_in_both_spt[:, 1]]

                        #################

                        # compute dist within one patch
                        dist_within_cluster_src = torch.cdist(pts_coord_in_curr_spt_src_mutual,
                                                              pts_coord_in_curr_spt_src_mutual, p=2)
                        dist_within_cluster_tgt = torch.cdist(pts_coord_in_curr_spt_tgt_mutual,
                                                              pts_coord_in_curr_spt_tgt_mutual, p=2)
                        distance_difference = torch.abs(dist_within_cluster_src - dist_within_cluster_tgt)
                        num_ele = len(distance_difference) * (len(distance_difference) - 1) / 2
                        dist_mean = torch.sum(torch.triu(distance_difference, diagonal=1)) / num_ele
                        ratio_inlier = (torch.sum(distance_difference <= self.method.thres_dist_diff) - distance_difference.shape[0]) / (num_ele * 2)
                        #################
                        metric_values.append([ratio_inlier, dist_mean])
                        if ratio_inlier <= self.method.thres_inlier_ratio or dist_mean >= self.method.thres_dist_diff:

                            mask_spt_match_useful[i] = False
                            continue
                        else:
                            # idx_corres_in_both_spt = idx_corres_from_2d_in_both_spt
                            weight_vector = None
                    else:
                        metric_values.append([0., 0.])
            else:
                # skip refinement, if 3d parts have many correspondences
                metric_values.append([1., 1.])

            pts_coord_in_curr_spt_src_mutual = self.data_input_3d.src_pts[idx_corres_in_both_spt[:, 0]]
            pts_coord_in_curr_spt_tgt_mutual = self.data_input_3d.tgt_pts[idx_corres_in_both_spt[:, 1]]

            # in case no point-point matches in current cluster match; if yes, ignore the current search
            # if pts_coord_in_curr_spt_tgt_mutual.numel() >= self.method.num_min_fine_match:
            if pts_coord_in_curr_spt_tgt_mutual.shape[0] >= self.method.num_min_fine_match:
                corres_temp = torch.cat((pts_coord_in_curr_spt_src_mutual, pts_coord_in_curr_spt_tgt_mutual), dim=1)

                corr_pts_coord_in_curr_spt_match_pruning, est_transform_svd = refine_local_rigid_correspondences(
                    corres_temp, refine_type='SVD', weights=weight_vector)

                # TODO: both of two ways assign the transformation to src pts, instead of using real point-to-point correspondences?
                # TODO: how about using the real point-to-point correspondences first, and taking the rest when necessary?
                if self.method.icp_refine:
                    # all pts in current spt
                    pts_coord_in_curr_spt_src = self.data_input_3d.src_pts[pts_idx_in_curr_spt_src]
                    pts_coord_in_curr_spt_tgt = self.data_input_3d.tgt_pts[pts_idx_in_curr_spt_tgt]

                    # only matched pts in current spt
                    # still use the point matches for icp estimation, better than using directly all pts, may have redundant src pts
                    pcd_in_curr_spt_src = tensor2pcd(pts_coord_in_curr_spt_src_mutual)
                    pcd_in_curr_spt_tgt = tensor2pcd(pts_coord_in_curr_spt_tgt_mutual)

                    # o3d.visualization.draw_geometries([pcd_in_curr_spt_src, pcd_in_curr_spt_tgt])

                    icp_result = icp_registration(pcd_in_curr_spt_src, pcd_in_curr_spt_tgt,
                                                  threshold=self.para.icp_threshold,
                                                  initial_transform=est_transform_svd.cpu(), icp_type='point2point')

                    fitness = icp_result['fitness']
                    rmse = icp_result['inlier_rmse']

                    icp_transform_refine = np.eye(4)
                    icp_transform_refine[:3, :3] = icp_result['est_transform'][:3, :3]
                    icp_transform_refine[:3, 3] = icp_result['est_transform'][:3, 3]
                    icp_transform_refine = torch.tensor(icp_transform_refine, dtype=torch.float32).cuda()

                    # temp_dvf_region_refine_src = temp_dvf_region_pruning[:, :3]
                    temp_corr_src = pts_coord_in_curr_spt_src[:, :3]
                    temp_corr_tgt = pts_coord_in_curr_spt_tgt[:, :3]
                    temp_corr_tgt_apply_icp = ((icp_transform_refine[:3, :3] @ temp_corr_src.T).T +
                                                  icp_transform_refine[:3, 3])

                    # only use sparse correspondences, instead of all pts in current src spt
                    temp_corr_discrete_src = pts_coord_in_curr_spt_src_mutual[:, :3]
                    temp_corr_discrete_tgt = pts_coord_in_curr_spt_tgt_mutual[:, :3]
                    temp_corr_discrete_tgt_apply_icp = ((icp_transform_refine[:3, :3] @ temp_corr_discrete_src.T).T +
                                               icp_transform_refine[:3, 3])

                    # use confidence values
                    # confidence_dense = compute_confidence(temp_corr_src.cpu().numpy(), temp_corr_tgt_apply_icp.cpu().numpy(), icp_transform_refine.cpu().numpy())

                    ###############
                    if self.method.output_tgt2src:
                        # get the opposite_one
                        temp_corr_tgt = pts_coord_in_curr_spt_tgt[:, :3]
                        temp_corr_src_apply_icp = (
                                    icp_transform_refine[:3, :3].T @ (temp_corr_tgt - icp_transform_refine[:3, 3]).T).T

                        temp_corr_apply_icp_tgt2src = torch.hstack([temp_corr_src_apply_icp, temp_corr_tgt])
                        corres_3d_refine_apply_icp_tgt2src.append(temp_corr_apply_icp_tgt2src)
                    ##############
                    # check how large of the residuals
                    # temp_corr_tgt = temp_corr[:, 3:6]
                    # residual_icp = torch.linalg.norm(temp_corr_tgt_apply_icp - temp_corr_tgt, axis=1)
                    # inlier = residual_icp <= 2.5 * torch.mean(residual_icp)

                    # if torch.sum(inlier) / len(inlier) >= 0.9:
                    ##############

                    temp_corr_apply_icp = torch.hstack([temp_corr_src, temp_corr_tgt_apply_icp])
                    # temp_corr_apply_icp = torch.hstack([temp_corr_src, temp_corr_tgt_apply_icp])[inlier]
                    corres_3d_refine_apply_icp.append(temp_corr_apply_icp)

                    if self.method.assign_type == 'assign_all_src':
                        # temp_corr_apply_icp = torch.hstack([temp_corr_src, temp_corr_tgt_apply_icp])
                        # # temp_corr_apply_icp = torch.hstack([temp_corr_src, temp_corr_tgt_apply_icp])[inlier]
                        # corres_3d_refine_apply_icp.append(temp_corr_apply_icp)

                        temp_corr_apply_icp_discrete = torch.hstack([temp_corr_discrete_src, temp_corr_discrete_tgt_apply_icp])
                        corres_3d_refine_apply_icp_discrete.append(temp_corr_apply_icp_discrete)
                    elif self.method.assign_type == 'assign_then_nn':
                        # TODO: not implemented
                        # TODO: so far only consider the output of src2tgt, dense and sparse, not tgt2src

                        # depending on data acquisition quality and scan resolution
                        # use rmse * 6 & med_res * 3 for low-quality cases; use rmse * 2 & med_res * 1 for normal cases
                        adaptive_threshold = rmse * 2.0
                        if np.isnan(adaptive_threshold) or np.isinf(adaptive_threshold):
                            adaptive_threshold = self.para.median_max_resolution
                        adaptive_threshold = max(adaptive_threshold, self.para.median_max_resolution * 1.0)
                        # adaptive_threshold = 10

                        # only consider the point-to-point matches
                        refined_dvfs_sparse = refine_dvfs_with_threshold(
                            temp_corr_src, temp_corr_tgt_apply_icp, temp_corr_tgt,
                            distance_threshold=adaptive_threshold, batch_size=1024)
                        corres_3d_refine_apply_icp_discrete.append(refined_dvfs_sparse)

                        # refined_dvfs_sparse = refine_dvfs_with_threshold(
                        #     temp_corr_discrete_src, temp_corr_discrete_tgt_apply_icp, temp_corr_discrete_tgt,
                        #     distance_threshold=adaptive_threshold, batch_size=1024)
                        # temp_corr_apply_icp_discrete = torch.hstack([temp_corr_discrete_src, temp_corr_discrete_tgt_apply_icp])
                        corres_3d_refine_apply_icp_discrete.append(refined_dvfs_sparse)
            else:
                mask_spt_match_global[i] = False

        # in case only 2d results in no inlier dvfs
        if corres_3d_refine_apply_icp != []:
            # TODO: if without icp_refine, no output?
            if self.method.icp_refine:
                self.data_output.corres_3d_refine_apply_icp = torch.vstack(corres_3d_refine_apply_icp)

                if self.method.output_tgt2src:
                    # from tgt --> apply icp transform.T
                    self.data_output.corres_3d_refine_apply_icp_tgt2src = torch.vstack(corres_3d_refine_apply_icp_tgt2src)

                # save only matched pts, sparse, for debugging
                self.data_output.corres_3d_refine_apply_icp_discrete = torch.vstack(corres_3d_refine_apply_icp_discrete)

        if self.verbose:
            if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
                self.logging.info(f'Fine matching is done for the superpoint level from {self.method.level_of_superpoint}!')
            else:
                self.logging.info('Fine matching is done!')

    def save_process_dvf(self):
        # TODO: filter out max. magnitudes instead of just using them for visualization
        if self.method.icp_refine:
            # compute magnitudes
            self.data_output.corres_3d_magnitude_refine_apply_icp = torch.linalg.norm(
                self.data_output.corres_3d_refine_apply_icp[:, 3:6] -
                self.data_output.corres_3d_refine_apply_icp[:, :3], dim=1)[:, None]
            if self.method.output_tgt2src:
                self.data_output.corres_3d_magnitude_refine_apply_icp_tgt2src = torch.linalg.norm(
                    self.data_output.corres_3d_refine_apply_icp_tgt2src[:, 3:6] -
                    self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :3], dim=1)[:, None]
            self.data_output.corres_3d_magnitude_refine_apply_icp_discrete = torch.linalg.norm(
                self.data_output.corres_3d_refine_apply_icp_discrete[:, 3:6] -
                self.data_output.corres_3d_refine_apply_icp_discrete[:, :3], dim=1)[:, None]

            dir_exist(osp.join(self.output_root, 'results'))
            if self.data.multiple_case:
                # save tile_id related results
                np.savetxt(osp.join(self.output_root, 'results',
                                    f'c2f_dense_dvfs_src2tgt_tile_{self.config.tile_id}.txt'),
                           self.data_output.corres_3d_refine_apply_icp.cpu(), delimiter=' ', fmt='%.6f')
                np.savetxt(osp.join(self.output_root, 'results',
                                    f'c2f_dense_dvfms_src2tgt_tile_{self.config.tile_id}.txt'),
                           np.hstack((self.data_output.corres_3d_refine_apply_icp[:, :3].cpu(),
                                      self.data_output.corres_3d_magnitude_refine_apply_icp.cpu())),
                           delimiter=' ', fmt='%.6f')
                # save results for visualization
                corres_3d_magnitude_visualize = self.data_output.corres_3d_magnitude_refine_apply_icp.cpu()

                # for better visualization in CloudCompare
                # only for visualization, not for real computation; can be commented for faster computation
                if self.data.dataset == 'rockfall_simulator':
                    max_mag_visualize = 0.06
                elif self.data.dataset == 'brienz_tls':
                    max_mag_visualize = 5
                elif self.data.dataset == 'mattertal':
                    max_mag_visualize = 10
                else:
                    max_mag_visualize = 10

                corres_3d_magnitude_visualize[0] = 0
                corres_3d_magnitude_visualize[1] = max_mag_visualize
                np.savetxt(osp.join(self.output_root, 'results',
                                    f'c2f_dense_dvfms_src2tgt_visualize_tile_{self.config.tile_id}.txt'),
                           np.hstack((self.data_output.corres_3d_refine_apply_icp[:, :3].cpu(),
                                      corres_3d_magnitude_visualize)),
                           delimiter=' ', fmt='%.6f')

                # output the point-to-point correspondences, sparse output
                if hasattr(self.data_output, 'corres_3d_refine_apply_icp_discrete'):
                    corres_3d_magnitude_discrete_visualize = self.data_output.corres_3d_magnitude_refine_apply_icp_discrete.cpu()
                    corres_3d_magnitude_discrete_visualize[0] = 0
                    corres_3d_magnitude_discrete_visualize[1] = max_mag_visualize
                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_sparse_dvfms_src2tgt_visualize_tile_{self.config.tile_id}.txt'),
                               np.hstack((self.data_output.corres_3d_refine_apply_icp_discrete[:, :3].cpu(),
                                          corres_3d_magnitude_discrete_visualize)),
                               delimiter=' ', fmt='%.6f')

                if self.method.output_tgt2src:
                # check if tgt2src dvfs are computed; if yes, output the corresponding results
                # if hasattr(self.data_output, 'corres_3d_refine_apply_icp_tgt2src'):
                    # save tgt2src assignment dvfs
                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_dense_dvfms_tgt2src_tile_{self.config.tile_id}.txt'),
                               np.hstack((self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :3].cpu(),
                                          self.data_output.corres_3d_magnitude_refine_apply_icp_tgt2src.cpu())),
                               delimiter=' ', fmt='%.6f')
                    # save results for visualization
                    corres_3d_magnitude_visualize_2 = self.data_output.corres_3d_magnitude_refine_apply_icp_tgt2src.cpu()

                    corres_3d_magnitude_visualize_2[0] = 0
                    corres_3d_magnitude_visualize_2[1] = max_mag_visualize
                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_dense_dvfms_tgt2src_visualize_tile_{self.config.tile_id}.txt'),
                               np.hstack((self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :3].cpu(),
                                          corres_3d_magnitude_visualize_2)),
                               delimiter=' ', fmt='%.6f')

                    # save mutual intersection and the rest of tgt2src dvfms
                    dvfs_1 = self.data_output.corres_3d_refine_apply_icp[:, :].cpu()
                    dvfs_2 = self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :].cpu()

                    # REVERSE!!!
                    s_tree = cKDTree(dvfs_2[:, :3])
                    distance, indices = s_tree.query(dvfs_1[:, :3], k=1)
                    # dist less than 1 voxel size, or both dist to raw pcd
                    mask = distance < self.method.voxel_size
                    mask_tgt_remain = torch.ones(dvfs_2.shape[0], dtype=bool)
                    mask_tgt_remain[indices[mask]] = False

                    mag_intersect = corres_3d_magnitude_visualize_2[indices[mask]]
                    mag_intersect[0] = 0
                    mag_intersect[1] = max_mag_visualize
                    mag_remain = corres_3d_magnitude_visualize_2[mask_tgt_remain]
                    mag_remain[0] = 0
                    mag_remain[1] = max_mag_visualize

                    #########################
                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_dvfms_tgt2src_mutual_intersect_with_src2tgt_visualize_tile_{self.config.tile_id}.txt'),
                               np.hstack(
                                   (self.data_output.corres_3d_refine_apply_icp_tgt2src[indices[mask], :3].cpu(),
                                    mag_intersect)),
                               delimiter=' ', fmt='%.6f')

                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_dvfms_tgt2src_mutual_remain_with_src2tgt_visualize_tile_{self.config.tile_id}.txt'),
                               np.hstack(
                                   (self.data_output.corres_3d_refine_apply_icp_tgt2src[mask_tgt_remain, :3].cpu(),
                                    mag_remain)),
                               delimiter=' ', fmt='%.6f')
            else:
                np.savetxt(osp.join(self.output_root, 'results', f'c2f_dvfs_src2tgt.txt'),
                           self.data_output.corres_3d_refine_apply_icp.cpu(), delimiter=' ', fmt='%.6f')
                np.savetxt(osp.join(self.output_root, 'results', f'c2f_dvfms_src2tgt.txt'),
                           np.hstack((self.data_output.corres_3d_refine_apply_icp[:, :3].cpu(),
                                      self.data_output.corres_3d_magnitude_refine_apply_icp.cpu())),
                           delimiter=' ', fmt='%.6f')
                # save results for visualization
                corres_3d_magnitude_visualize = self.data_output.corres_3d_magnitude_refine_apply_icp.cpu()

                if self.data.dataset == 'rockfall_simulator':
                    max_mag_visualize = 0.06
                else:
                    max_mag_visualize = 5

                corres_3d_magnitude_visualize[0] = 0
                corres_3d_magnitude_visualize[1] = max_mag_visualize
                np.savetxt(osp.join(self.output_root, 'results', f'c2f_dvfms_src2tgt_visualize_0_5.txt'),
                           np.hstack((self.data_output.corres_3d_refine_apply_icp[:, :3].cpu(),
                                      corres_3d_magnitude_visualize)),
                           delimiter=' ', fmt='%.6f')

                if hasattr(self.data_output, 'corres_3d_refine_apply_icp_discrete'):
                    corres_3d_magnitude_discrete_visualize = self.data_output.corres_3d_magnitude_refine_apply_icp_discrete.cpu()
                    corres_3d_magnitude_discrete_visualize[0] = 0
                    corres_3d_magnitude_discrete_visualize[1] = max_mag_visualize
                    np.savetxt(
                        osp.join(self.output_root, 'results', f'c2f_dvfms_src2tgt_discrete_visualize_0_5.txt'),
                        np.hstack((self.data_output.corres_3d_refine_apply_icp_discrete[:, :3].cpu(),
                                   corres_3d_magnitude_discrete_visualize)),
                        delimiter=' ', fmt='%.6f')

                if self.method.output_tgt2src:
                # check if tgt2src dvfs are computed; if yes, output the corresponding results
                # if hasattr(self.data_output, 'corres_3d_refine_apply_icp_tgt2src'):
                    # save tgt2src assignment dvfs
                    np.savetxt(osp.join(self.output_root, 'results', f'c2f_dvfms_tgt2src.txt'),
                               np.hstack((self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :3].cpu(),
                                          self.data_output.corres_3d_magnitude_refine_apply_icp_tgt2src.cpu())),
                               delimiter=' ', fmt='%.6f')
                    # save results for visualization
                    corres_3d_magnitude_visualize_2 = self.data_output.corres_3d_magnitude_refine_apply_icp_tgt2src.cpu()

                    corres_3d_magnitude_visualize_2[0] = 0
                    corres_3d_magnitude_visualize_2[1] = max_mag_visualize
                    np.savetxt(osp.join(self.output_root, 'results', f'c2f_dvfms_tgt2src_visualize_0_5.txt'),
                               np.hstack((self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :3].cpu(),
                                          corres_3d_magnitude_visualize_2)),
                               delimiter=' ', fmt='%.6f')

                    # save mutual intersection and the rest of tgt2src dvfms
                    dvfs_1 = self.data_output.corres_3d_refine_apply_icp[:, :].cpu()
                    dvfs_2 = self.data_output.corres_3d_refine_apply_icp_tgt2src[:, :].cpu()

                    # REVERSE!!!
                    s_tree = cKDTree(dvfs_2[:, :3])
                    distance, indices = s_tree.query(dvfs_1[:, :3], k=1)
                    # dist less than 1 voxel size, or both dist to raw pcd
                    mask = distance < self.para.voxel_size
                    mask_tgt_remain = torch.ones(dvfs_2.shape[0], dtype=bool)
                    mask_tgt_remain[indices[mask]] = False

                    mag_intersect = corres_3d_magnitude_visualize_2[indices[mask]]
                    mag_intersect[0] = 0
                    mag_intersect[1] = max_mag_visualize
                    mag_remain = corres_3d_magnitude_visualize_2[mask_tgt_remain]
                    mag_remain[0] = 0
                    mag_remain[1] = max_mag_visualize

                    # corres_3d_magnitude_visualize_2[indices[mask]][1] = 5
                    # corres_3d_magnitude_visualize_2[mask_tgt_remain][0] = 0
                    # corres_3d_magnitude_visualize_2[mask_tgt_remain][1] = 5
                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_dvfms_tgt2src_mutual_intersect_with_src2tgt_visualize_0_5.txt'),
                               np.hstack(
                                   (self.data_output.corres_3d_refine_apply_icp_tgt2src[indices[mask], :3].cpu(),
                                    mag_intersect)),
                               delimiter=' ', fmt='%.6f')

                    np.savetxt(osp.join(self.output_root, 'results',
                                        f'c2f_dvfms_tgt2src_mutual_remain_with_src2tgt_visualize_0_5.txt'),
                               np.hstack(
                                   (self.data_output.corres_3d_refine_apply_icp_tgt2src[mask_tgt_remain, :3].cpu(),
                                    mag_remain)),
                               delimiter=' ', fmt='%.6f')

    ##################################
    # internal functions
    ##################################

    def _get_curr_spt_feat_and_coord(self, curr_patch_idx, spt_type='src'):
        # all points lie in current superpoint patch will have True value
        if spt_type == 'src':
            # create temp variables
            spt_idx_src_temp = self.data_interim.spt_idx_src_temp
            pts_feat_src = self.data_input.pts_feat_src

            pts_idx_in_curr_spt = spt_idx_src_temp[1] == spt_idx_src_temp[0][curr_patch_idx]
            pts_feat_in_curr_spt = pts_feat_src[pts_idx_in_curr_spt, :]
        elif spt_type == 'tgt':
            spt_idx_tgt_temp = self.data_interim.spt_idx_tgt_temp
            pts_feat_tgt = self.data_input.pts_feat_tgt

            pts_idx_in_curr_spt = spt_idx_tgt_temp[1] == spt_idx_tgt_temp[0][curr_patch_idx]
            pts_feat_in_curr_spt = pts_feat_tgt[pts_idx_in_curr_spt, :]

        # use different feature aggregation methods. Currently, averaging performs the best
        if self.method.feat_aggregate_type == 'avg':
            # just take the average features for each superpoint patch
            feat_curr_spt = torch.mean(pts_feat_in_curr_spt, dim=0)
            # feat_curr_spt_tgt = torch.mean(pts_feat_in_curr_spt_tgt, dim=0)

            # add additional std
            # mean_feat = np.mean(pts_feat_in_curr_spt_src.cpu().numpy(), axis=0)
            # std_feat = np.std(pts_feat_in_curr_spt_src.cpu().numpy(), axis=0) * 10
            # src_feat_aggregate_temp = np.concatenate((mean_feat, std_feat), axis=0)
        elif self.method.feat_aggregate_type == 'max':
            feat_curr_spt = torch.max(pts_feat_in_curr_spt, dim=0, keepdim=False)[0]
        elif self.method.feat_aggregate_type == 'pca':
            # bad
            pca = PCA(n_components=64)
            pca.fit(pts_feat_in_curr_spt.cpu().numpy())
            transformed_data = pca.transform(pts_feat_in_curr_spt.cpu().numpy())
            transformed_data = torch.from_numpy(transformed_data).to(self.device)
            feat_curr_spt_src = torch.mean(transformed_data, dim=0)
        elif self.method.feat_aggregate_type == 'vlad':
            # numpy version, results are close
            pts_feat_in_curr_spt_src_np = pts_feat_in_curr_spt.cpu().numpy()
            features = torch.rand(pts_feat_in_curr_spt_src_np.shape[0], pts_feat_in_curr_spt_src_np.shape[1])
            num_clusters = 1
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
            src_feat_aggregate_temp = compute_vlad(features, kmeans).to(self.device)
        elif self.method.feat_aggregate_type == 'vlad_2':
            # use vlad to aggregate features in each superpoint patch
            # numpy version
            vlad = Vlad(1)
            vlad.partial_fit(pts_feat_in_curr_spt.cpu().numpy())
            src_feat_aggregate_temp = vlad.transform(np.expand_dims(pts_feat_in_curr_spt.cpu().numpy(), axis=0))
            src_feat_aggregate_temp = src_feat_aggregate_temp.reshape(-1)
            src_feat_aggregate_temp = array2tensor(src_feat_aggregate_temp).to(self.device)
        elif self.method.feat_aggregate_type == 'spatial_pyramid':
            # wrong, idea is using 'max' and 'avg' simultaneously
            num_levels = 3  # Number of levels in the spatial pyramid
            pool_type = 'avg'  # Pooling type, can be 'max' or 'avg'
            pts_feat_in_curr_spt_src_np = pts_feat_in_curr_spt.cpu().numpy()

            output = spatial_pyramid_pooling(pts_feat_in_curr_spt_src_np, num_levels, pool_type)
            a = 0

        # use the centroid coord. to represent each superpoin patch
        if spt_type == 'src':
            pts_coord_in_curr_spt = self.data_input.pts_coord_src[pts_idx_in_curr_spt]
        elif spt_type == 'tgt':
            pts_coord_in_curr_spt = self.data_input.pts_coord_tgt[pts_idx_in_curr_spt]
        coord_curr_spt = torch.mean(pts_coord_in_curr_spt, dim=0)

        return feat_curr_spt, coord_curr_spt

    def _get_coarse_matching_indices(self):
        """
        Get coarse matching indices with size: (num of spt_src, k closest spt_tgt),
        unmatches are filled with invalid indices
        """
        if self.verbose:
            self.logging.info(f'Get initial coarse matching indices')

        self.data_interim.src2tgt_coarse_match_idx = (
                torch.ones((self.data_interim.spt_idx_src_temp[2], self.para.num_k),
                           device=self.device) * self.data_interim.spt_idx_tgt_temp[2]).to(torch.int64)

        if self.method.coarse_matching_mutual:
            self.data_interim.src2tgt_coarse_match_idx_mutual = (
                    torch.ones((self.data_interim.spt_idx_tgt_temp[2], self.para.num_k),
                               device=self.device) * self.data_interim.spt_idx_src_temp[2]).to(torch.int64)

    def _torch_cdist_with_mutual(self, search_type='src'):
        """ Compute distance considering mutual information """
        if self.verbose:
            self.logging.info(f'Search distance constraint based neighbors, {search_type}')
        # can be accelerated?
        #######################
        # test optimal transport

        # data_interim, dict_method, dict_hypara, dict_visualize, dict_temp,

        if search_type == 'src':
            idx_temp = self.data_interim.spt_idx_src_temp[2]
        elif search_type == 'tgt':
            idx_temp = self.data_interim.spt_idx_tgt_temp[2]
        # idx_temp = self.data_interim.spt_idx_src_temp[2]
        for i in tqdm(torch.arange(idx_temp), position=0, leave=True):
            # only select < 10 m corr. for superpoint matching
            if self.method.dist_constraint:
                # select the tgt superpoints that have less than 10 m dist. to the current src superpoint
                if search_type == 'src':
                    mask = self._neighborhood_searching(self.data_interim.coord_aggregate_spt_src[i],
                                                        self.data_interim.coord_aggregate_spt_tgt,
                                                        max_magnitude=self.para.max_magnitude)
                elif search_type == 'tgt':
                    mask = self._neighborhood_searching(self.data_interim.coord_aggregate_spt_tgt[i],
                                                        self.data_interim.coord_aggregate_spt_src,
                                                        max_magnitude=self.para.max_magnitude)
                # skip the src spt that contains not enough valid matched tgt spt
                if torch.sum(mask) == 0:
                    continue
                if search_type == 'src':
                    dists = torch.cdist(self.data_interim.feat_aggregate_spt_src[i].unsqueeze(0),
                                        self.data_interim.feat_aggregate_spt_tgt[mask])
                elif search_type == 'tgt':
                    dists = torch.cdist(self.data_interim.feat_aggregate_spt_tgt[i].unsqueeze(0),
                                        self.data_interim.feat_aggregate_spt_src[mask])
                # if no enough candidates for top-k selection
                if dists.shape[1] < self.para.num_k:
                    idx_origin = torch.nonzero(mask).squeeze().squeeze()
                else:
                    # the indices mean the location after the mask operation, not the location before it
                    _, indices = dists.topk(self.para.num_k, largest=False)
                    # get the indices in original location before the mask operation
                    idx_origin = torch.nonzero(mask).squeeze()[indices].squeeze()
            else:
                # no distance constrain
                if search_type == 'src':
                    dists = torch.cdist(data_interim.feat_spt_src[i].unsqueeze(0),
                                        data_interim.feat_spt_tgt)
                elif search_type == 'tgt':
                    dists = torch.cdist(data_interim.feat_spt_tgt[i].unsqueeze(0),
                                        data_interim.feat_spt_src)

                _, indices = dists.topk(dict_hypara.num_k, largest=False)

                idx_origin = indices.squeeze()

            # save the matched tgt indices to src superpoints, e.g., [1813, 5]
            # if idx_origin.dim() == 0:
            #     a = 0
            if search_type == 'src':
                self.data_interim.src2tgt_coarse_match_idx[i, :idx_origin.numel()] = idx_origin
            elif search_type == 'tgt':
                self.data_interim.src2tgt_coarse_match_idx_mutual[i, :idx_origin.numel()] = idx_origin

            ####################
            if self.visualize.visualize_patch and i <= self.visualize.num_of_visualize_samples:
                # visualization
                dict_temp = edict(dict())
                if search_type == 'src':
                    dict_temp.coord_spt_src_single = self.data_interim.coord_aggregate_spt_src[i]
                elif search_type == 'tgt':
                    dict_temp.coord_spt_tgt_single = self.data_interim.coord_aggregate_spt_tgt[i]

                # dict_temp.indices = indices
                dict_temp.idx_origin = idx_origin
                dict_temp.i = i
                self._spt_matching_visualization(dict_temp, search_type=search_type)

    def _neighborhood_searching(self, spt_src_cen, spt_tgt_centroid, max_magnitude=10):
        spt_norm = torch.linalg.norm(spt_src_cen - spt_tgt_centroid[:, :3], axis=1)
        mask = spt_norm <= max_magnitude
        return mask

    def _compute_unique_spt_matches(self):
        """ Compute unique spt matches, output local, global spt matching indices, optional? spt matching coord """
        if self.verbose:
            self.logging.info('Computing unique spt matches based on the most reliable pts matches')

        if self.method.unique_selection_type == 'isometry_euclidean':
            # select the one with most point matches after mutual selection
            spt_match_idx_src_refine_sorted_local = self.data_output.spt_matches_idx_refine_sorted[:, 0]
            spt_match_idx_tgt_refine_sorted_local = self.data_output.spt_matches_idx_refine_sorted[:, 1]

            # only consider the case for src
            temp_idx = torch.unique(spt_match_idx_src_refine_sorted_local).to(self.device)
            spt_matches_unique = torch.zeros((temp_idx.shape[0], 2), dtype=torch.int64).to(self.device)
            # two columns, first: src indices of unique matches; second: num. of reliable pts matches
            spt_matches_unique[:, 0] = temp_idx
            # store the isometry Euclidean error
            spt_matches_iso_error = torch.full((temp_idx.numel(),), float('inf'), dtype=torch.float64).to(self.device)
            # store the tgt indices of unique matches, may happen that initial 0 is conflict with real 0
            spt_matches_idx_refine_unique = torch.zeros(temp_idx.numel(), dtype=torch.int64).to(self.device)

            self.data_interim.idx_pts_in_spt_match = []

            # select the one has most mutual selected matches
            spt_match_idx_src_refine_sorted_global = self.data_interim.spt_idx_src_temp[0][
                spt_match_idx_src_refine_sorted_local]
            spt_match_idx_tgt_refine_sorted_global = self.data_interim.spt_idx_tgt_temp[0][
                spt_match_idx_tgt_refine_sorted_local]
            # for all established matches, in the end save unique matches
            for i in tqdm(range(0, self.data_output.spt_matches_sorted.shape[0]), position=0, leave=True):
                pts_in_curr_patch_idx_src = (self.data_input.pts2spt_idx_src.reshape(-1) ==
                                             spt_match_idx_src_refine_sorted_global[i])
                pts_in_curr_patch_idx_tgt = (self.data_input.pts2spt_idx_tgt.reshape(-1) ==
                                             spt_match_idx_tgt_refine_sorted_global[i])

                # not used
                # pts_in_curr_patch_coord_src = self.data_input.pts_coord_src[pts_in_curr_patch_idx_src, :]
                # pts_in_curr_patch_coord_tgt = self.data_input.pts_coord_tgt[pts_in_curr_patch_idx_tgt, :]

                pts_in_curr_patch_feat_src = self.data_input.pts_feat_src[pts_in_curr_patch_idx_src, :]
                pts_in_curr_patch_feat_tgt = self.data_input.pts_feat_tgt[pts_in_curr_patch_idx_tgt, :]

                # based on knn, mutual selection to generate reliable matches
                if self.method.fine_matching_type == 'torch_cdist':
                    # Compute pairwise distances between all source and target features
                    dists = torch.cdist(pts_in_curr_patch_feat_src, pts_in_curr_patch_feat_tgt)

                    # Find the index of the nearest neighbor for each source feature
                    _, patch_labels_src_to_tgt = dists.topk(1, dim=1, largest=False)
                    # Find the index of the nearest neighbor for each target feature
                    _, patch_labels_tgt_to_src = dists.topk(1, dim=0, largest=False)

                    # Remove the singleton dimensions
                    patch_labels_src_to_tgt = patch_labels_src_to_tgt.squeeze(1)
                    patch_labels_tgt_to_src = patch_labels_tgt_to_src.squeeze(0)

                    patch_corr_idx = torch.cat(
                        (torch.arange(0, len(patch_labels_src_to_tgt)).to(self.device).unsqueeze(0),
                         patch_labels_src_to_tgt.unsqueeze(0)), dim=0).T
                    patch_corr_idx_mutual = torch.cat((
                        patch_labels_tgt_to_src.unsqueeze(0),
                        torch.arange(0, len(patch_labels_tgt_to_src)).to(self.device).unsqueeze(0)), dim=0).T

                    # mutual selection already remove most wrong matches
                    patch_labels, idx_src2tgt, idx_tgt2src = get_intersect_tensor(patch_corr_idx, patch_corr_idx_mutual,
                                                                                  self.device)

                    # collect the valid pts matches
                    # need the indice from 180 to 86, why?
                    idx_pts_in_curr_patch_global_src = torch.nonzero(pts_in_curr_patch_idx_src).squeeze()
                    idx_pts_in_curr_patch_global_tgt = torch.nonzero(pts_in_curr_patch_idx_tgt).squeeze()
                    idx_pts_in_curr_patch_global_src = [idx_pts_in_curr_patch_global_src[patch_labels[:, 0]]
                                                        if idx_pts_in_curr_patch_global_src.numel() > 1 else idx_pts_in_curr_patch_global_src.unsqueeze(
                        0)][0]
                    idx_pts_in_curr_patch_global_tgt = [idx_pts_in_curr_patch_global_tgt[patch_labels[:, 1]]
                                                        if idx_pts_in_curr_patch_global_tgt.numel() > 1 else idx_pts_in_curr_patch_global_tgt.unsqueeze(
                        0)][0]

                    # idx_pts_in_curr_patch_global_src = torch.nonzero(pts_in_curr_patch_idx_src).squeeze()[patch_labels[:, 0]]
                    # idx_pts_in_curr_patch_global_tgt = torch.nonzero(pts_in_curr_patch_idx_tgt).squeeze()[patch_labels[:, 1]]

                    idx_pts_in_curr_patch_global = torch.cat((idx_pts_in_curr_patch_global_src.unsqueeze(1),
                                                              idx_pts_in_curr_patch_global_tgt.unsqueeze(1)), dim=1)

                    # just reliable pts match indices
                    self.data_interim.idx_pts_in_spt_match.append(idx_pts_in_curr_patch_global)

                # compute the isometry Euclidean error
                iso_error = self._compute_isometry_euclidean_error(idx_pts_in_curr_patch_global)

                # spt_matches_unique: unique_idx x max. pts matches
                # compare and save the indices and num. of the one with the most reliable matches
                spt_matches_temp = torch.where(
                    spt_matches_unique[:, 0] == self.data_output.spt_matches_idx_refine_sorted[i, 0])

                # in the end, may only spt_matches_idx_refine_unique is further used
                # so it doesn't matter what are recorded in spt_matches_unique and spt_matches_iso_error
                if iso_error < spt_matches_iso_error[spt_matches_temp]:
                    # record the lowest isometry error
                    spt_matches_iso_error[spt_matches_temp] = iso_error
                    # record the indices that have the lowest isometry error
                    spt_matches_idx_refine_unique[spt_matches_temp] = i

            # still the local indices, should assign to self.data_interim.spt_idx_xxx_temp[0]
            self.data_output.spt_match_idx_unique = self.data_output.spt_matches_idx_refine_sorted[
                spt_matches_idx_refine_unique]
            # save iso error for further comparison
            self.data_output.spt_match_iso_error = spt_matches_iso_error
            # just for visualization
            temp_spt_match_idx_unique_global_src = self.data_interim.spt_idx_src_temp[0][
                self.data_output.spt_match_idx_unique[:, 0]]
            temp_spt_match_idx_unique_global_tgt = self.data_interim.spt_idx_tgt_temp[0][
                self.data_output.spt_match_idx_unique[:, 1]]
            self.data_output.spt_match_idx_unique_global = torch.cat((temp_spt_match_idx_unique_global_src[:, None],
                                                                      temp_spt_match_idx_unique_global_tgt[:, None]),
                                                                     dim=1)

            # self.data_interim.spt_match_idx_tgt_unique = self.data_output.spt_match_idx_tgt_refine[spt_matches_idx_refine_unique]
            self.data_output.spt_match_coord_unique = self.data_output.spt_matches_sorted[spt_matches_idx_refine_unique]

            # to recover the original spt matches for getting unique reliable pts matches, for SVD estimation
            self.data_interim.spt_unique_idx = spt_matches_idx_refine_unique
        else:
            # select the one with most point matches after mutual selection
            spt_match_idx_src_refine_sorted_local = self.data_output.spt_matches_idx_refine_sorted[:, 0]
            spt_match_idx_tgt_refine_sorted_local = self.data_output.spt_matches_idx_refine_sorted[:, 1]

            # only consider the case for src
            temp_idx = torch.unique(spt_match_idx_src_refine_sorted_local).to(self.device)
            spt_matches_unique = torch.zeros((temp_idx.shape[0], 2), dtype=torch.int64).to(self.device)
            # two columns, first: src indices of unique matches; second: num. of reliable pts matches
            spt_matches_unique[:, 0] = temp_idx
            # store the tgt indices of unique matches, may happen that initial 0 is conflict with real 0
            spt_matches_idx_refine_unique = torch.zeros(temp_idx.numel(), dtype=torch.int64).to(self.device)

            self.data_interim.idx_pts_in_spt_match = []

            # select the one has most mutual selected matches
            spt_match_idx_src_refine_sorted_global = self.data_interim.spt_idx_src_temp[0][
                spt_match_idx_src_refine_sorted_local]
            spt_match_idx_tgt_refine_sorted_global = self.data_interim.spt_idx_tgt_temp[0][
                spt_match_idx_tgt_refine_sorted_local]
            # for all established matches, in the end save unique matches
            for i in tqdm(range(0, self.data_output.spt_matches_sorted.shape[0]), position=0, leave=True):
                pts_in_curr_patch_idx_src = (self.data_input.pts2spt_idx_src.reshape(-1) ==
                                             spt_match_idx_src_refine_sorted_global[i])
                pts_in_curr_patch_idx_tgt = (self.data_input.pts2spt_idx_tgt.reshape(-1) ==
                                             spt_match_idx_tgt_refine_sorted_global[i])

                # not used
                # pts_in_curr_patch_coord_src = self.data_input.pts_coord_src[pts_in_curr_patch_idx_src, :]
                # pts_in_curr_patch_coord_tgt = self.data_input.pts_coord_tgt[pts_in_curr_patch_idx_tgt, :]

                pts_in_curr_patch_feat_src = self.data_input.pts_feat_src[pts_in_curr_patch_idx_src, :]
                pts_in_curr_patch_feat_tgt = self.data_input.pts_feat_tgt[pts_in_curr_patch_idx_tgt, :]

                # based on knn, mutual selection to generate reliable matches
                if self.method.fine_matching_type == 'torch_cdist':
                    # Compute pairwise distances between all source and target features
                    dists = torch.cdist(pts_in_curr_patch_feat_src, pts_in_curr_patch_feat_tgt)

                    # Find the index of the nearest neighbor for each source feature
                    _, patch_labels_src_to_tgt = dists.topk(1, dim=1, largest=False)
                    # Find the index of the nearest neighbor for each target feature
                    _, patch_labels_tgt_to_src = dists.topk(1, dim=0, largest=False)

                    # Remove the singleton dimensions
                    patch_labels_src_to_tgt = patch_labels_src_to_tgt.squeeze(1)
                    patch_labels_tgt_to_src = patch_labels_tgt_to_src.squeeze(0)

                    patch_corr_idx = torch.cat(
                        (torch.arange(0, len(patch_labels_src_to_tgt)).to(self.device).unsqueeze(0),
                         patch_labels_src_to_tgt.unsqueeze(0)), dim=0).T
                    patch_corr_idx_mutual = torch.cat((
                        patch_labels_tgt_to_src.unsqueeze(0),
                        torch.arange(0, len(patch_labels_tgt_to_src)).to(self.device).unsqueeze(0)), dim=0).T

                    # mutual selection already remove most wrong matches
                    patch_labels, idx_src2tgt, idx_tgt2src = get_intersect_tensor(patch_corr_idx, patch_corr_idx_mutual,
                                                                                  self.device)

                    # collect the valid pts matches
                    # need the indice from 180 to 86, why?
                    idx_pts_in_curr_patch_global_src = torch.nonzero(pts_in_curr_patch_idx_src).squeeze()
                    idx_pts_in_curr_patch_global_tgt = torch.nonzero(pts_in_curr_patch_idx_tgt).squeeze()
                    idx_pts_in_curr_patch_global_src = [idx_pts_in_curr_patch_global_src[patch_labels[:, 0]]
                                                        if idx_pts_in_curr_patch_global_src.numel() > 1 else idx_pts_in_curr_patch_global_src.unsqueeze(
                        0)][0]
                    idx_pts_in_curr_patch_global_tgt = [idx_pts_in_curr_patch_global_tgt[patch_labels[:, 1]]
                                                        if idx_pts_in_curr_patch_global_tgt.numel() > 1 else idx_pts_in_curr_patch_global_tgt.unsqueeze(
                        0)][0]

                    # idx_pts_in_curr_patch_global_src = torch.nonzero(pts_in_curr_patch_idx_src).squeeze()[patch_labels[:, 0]]
                    # idx_pts_in_curr_patch_global_tgt = torch.nonzero(pts_in_curr_patch_idx_tgt).squeeze()[patch_labels[:, 1]]

                    idx_pts_in_curr_patch_global = torch.cat((idx_pts_in_curr_patch_global_src.unsqueeze(1),
                                                              idx_pts_in_curr_patch_global_tgt.unsqueeze(1)), dim=1)

                    # just reliable pts match indices
                    self.data_interim.idx_pts_in_spt_match.append(idx_pts_in_curr_patch_global)

                # spt_matches_unique: unique_idx x max. pts matches
                # compare and save the indices and num. of the one with the most reliable matches
                spt_matches_temp = torch.where(
                    spt_matches_unique[:, 0] == self.data_output.spt_matches_idx_refine_sorted[i, 0])
                curr_num_pts_matches = patch_labels.shape[0]
                if curr_num_pts_matches > spt_matches_unique[spt_matches_temp, 1]:
                    # record the highest pts matches of the unique 1-1 match
                    spt_matches_unique[spt_matches_temp, 1] = curr_num_pts_matches
                    # record the indices that have the most matches
                    spt_matches_idx_refine_unique[spt_matches_temp] = i

            # still the local indices, should assign to self.data_interim.spt_idx_xxx_temp[0]
            self.data_output.spt_match_idx_unique = self.data_output.spt_matches_idx_refine_sorted[
                spt_matches_idx_refine_unique]

            temp_spt_match_idx_unique_global_src = self.data_interim.spt_idx_src_temp[0][
                self.data_output.spt_match_idx_unique[:, 0]]
            temp_spt_match_idx_unique_global_tgt = self.data_interim.spt_idx_tgt_temp[0][
                self.data_output.spt_match_idx_unique[:, 1]]
            self.data_output.spt_match_idx_unique_global = torch.cat((temp_spt_match_idx_unique_global_src[:, None],
                                                                      temp_spt_match_idx_unique_global_tgt[:, None]),
                                                                     dim=1)

            # just for visualization
            # self.data_interim.spt_match_idx_tgt_unique = self.data_output.spt_match_idx_tgt_refine[spt_matches_idx_refine_unique]
            self.data_output.spt_match_coord_unique = self.data_output.spt_matches_sorted[spt_matches_idx_refine_unique]

            # to recover the original spt matches for getting unique reliable pts matches, for SVD estimation
            self.data_interim.spt_unique_idx = spt_matches_idx_refine_unique

        # visualization, visualize 1-to-1 patch matches
        if self.visualize.visualize_patch_after_union_selection:
            if self.verbose:
                self.logging.info(f"Visualizing unique patch matches after unique selection "
                                  f"{self.method.unique_selection_type}... ")
            # visualize refine coarse matches
            # visualize first 10 src - tgt matches after mutula selection, no need for tgt-based
            for spt_unique_i in torch.arange(0, self.visualize.num_of_visualize_samples):
                self._spt_matching_visualization_one2one_match(spt_unique_i)

    def _compute_unique_spt_matches_among_neighbors(self):
        """ Compute unique spt matches among neighbors
        because the unique spt matches are not so precise """

        global idx_pts_in_curr_patch_global
        if self.verbose:
            self.logging.info(f"Computing unique spt matches among neighbors ")

        # select the knn of spt_unique_matches_tgt, and cat with current tgt idx
        spt_match_idx_unique_global = self.data_output.spt_match_idx_unique_global
        spt_match_coord_unique_tgt = self.data_output.spt_match_coord_unique[:, 3:6]
        # compute geometric distances

        temp_dists = torch.cdist(spt_match_coord_unique_tgt, self.data_interim.coord_aggregate_spt_tgt)
        # Find the index of the nearest neighbor for each source feature, local indices
        _, patch_labels_tgt_to_neighbors = temp_dists.topk(self.para.num_close_neighbor, dim=1, largest=False)

        # Preallocate tensors for efficiency
        # spt_matches_iso_error_among_neighbors = copy.deepcopy(self.data_output.spt_match_iso_error)
        # spt_matches_idx_unique_among_neighbors_local = copy.deepcopy(self.data_output.spt_match_idx_unique)

        spt_matches_iso_error_among_neighbors = self.data_output.spt_match_iso_error.clone()
        spt_matches_idx_unique_among_neighbors_local = self.data_output.spt_match_idx_unique.clone()
        for num_unique_i in tqdm(torch.arange(0, self.data_output.spt_match_idx_unique.shape[0])):
            ###############
            if self.method.close_neighbor_selection_type == 'isometry_euclidean':
                # check if the global indices equal to the one derived from local indices
                pts_in_curr_patch_idx_src = (self.data_input.pts2spt_idx_src.reshape(-1) ==
                                             spt_match_idx_unique_global[num_unique_i, 0])
                pts_in_curr_patch_feat_src = self.data_input.pts_feat_src[pts_in_curr_patch_idx_src, :]

                for close_neigh_i in torch.arange(0, self.para.num_close_neighbor):
                    spt_match_idx_unique_tgt_neighbor_i_local = patch_labels_tgt_to_neighbors[
                        num_unique_i, close_neigh_i]
                    # make use the spt_idx_tgt_temp is just for one iteration
                    spt_match_idx_unique_tgt_neighbor_i_global = self.data_interim.spt_idx_tgt_temp[0][
                        spt_match_idx_unique_tgt_neighbor_i_local]
                    pts_in_curr_patch_idx_tgt = (self.data_input.pts2spt_idx_tgt.reshape(-1) ==
                                                 spt_match_idx_unique_tgt_neighbor_i_global)

                    pts_in_curr_patch_feat_tgt = self.data_input.pts_feat_tgt[pts_in_curr_patch_idx_tgt, :]

                    # based on knn, mutual selection to generate reliable matches
                    if self.method.fine_matching_type == 'torch_cdist':
                        # Compute pairwise distances between all source and target features
                        temp_dists = torch.cdist(pts_in_curr_patch_feat_src, pts_in_curr_patch_feat_tgt)

                        # Find the index of the nearest neighbor for each source feature
                        _, patch_labels_src_to_tgt = temp_dists.topk(1, dim=1, largest=False)
                        # Find the index of the nearest neighbor for each target feature
                        _, patch_labels_tgt_to_src = temp_dists.topk(1, dim=0, largest=False)

                        # Remove the singleton dimensions
                        patch_labels_src_to_tgt = patch_labels_src_to_tgt.squeeze(1)
                        patch_labels_tgt_to_src = patch_labels_tgt_to_src.squeeze(0)

                        patch_corr_idx = torch.cat(
                            (torch.arange(0, len(patch_labels_src_to_tgt)).to(self.device).unsqueeze(0),
                             patch_labels_src_to_tgt.unsqueeze(0)), dim=0).T
                        patch_corr_idx_mutual = torch.cat((
                            patch_labels_tgt_to_src.unsqueeze(0),
                            torch.arange(0, len(patch_labels_tgt_to_src)).to(self.device).unsqueeze(0)), dim=0).T

                        # mutual selection already remove most wrong matches
                        patch_labels, idx_src2tgt, idx_tgt2src = get_intersect_tensor(patch_corr_idx,
                                                                                      patch_corr_idx_mutual,
                                                                                      self.device)

                        # collect the valid pts matches
                        # need the indice from 180 to 86, why?
                        idx_pts_in_curr_patch_global_src = torch.nonzero(pts_in_curr_patch_idx_src).squeeze()
                        idx_pts_in_curr_patch_global_tgt = torch.nonzero(pts_in_curr_patch_idx_tgt).squeeze()
                        idx_pts_in_curr_patch_global_src = [idx_pts_in_curr_patch_global_src[patch_labels[:, 0]]
                                                            if idx_pts_in_curr_patch_global_src.numel() > 1 else idx_pts_in_curr_patch_global_src.unsqueeze(
                            0)][0]
                        idx_pts_in_curr_patch_global_tgt = [idx_pts_in_curr_patch_global_tgt[patch_labels[:, 1]]
                                                            if idx_pts_in_curr_patch_global_tgt.numel() > 1 else idx_pts_in_curr_patch_global_tgt.unsqueeze(
                            0)][0]

                        # idx_pts_in_curr_patch_global_src = torch.nonzero(pts_in_curr_patch_idx_src).squeeze()[patch_labels[:, 0]]
                        # idx_pts_in_curr_patch_global_tgt = torch.nonzero(pts_in_curr_patch_idx_tgt).squeeze()[patch_labels[:, 1]]

                        idx_pts_in_curr_patch_global = torch.cat((idx_pts_in_curr_patch_global_src.unsqueeze(1),
                                                                  idx_pts_in_curr_patch_global_tgt.unsqueeze(1)), dim=1)

                        # just reliable pts match indices
                        # self.data_interim.idx_pts_in_spt_match.append(idx_pts_in_curr_patch_global)

                    # compute the isometry Euclidean error
                    iso_error = self._compute_isometry_euclidean_error(idx_pts_in_curr_patch_global)

                    # in the end, may only spt_matches_idx_refine_unique is further used
                    # so it doesn't matter what are recorded in spt_matches_unique and spt_matches_iso_error
                    if iso_error < spt_matches_iso_error_among_neighbors[num_unique_i]:
                        # record the lowest isometry error
                        spt_matches_iso_error_among_neighbors[num_unique_i] = iso_error
                        # record the indices that have the lowest isometry error
                        spt_matches_idx_unique_among_neighbors_local[num_unique_i, 1] = (
                            spt_match_idx_unique_tgt_neighbor_i_local)

                        # update pts matches, useful for svd estimation
                        temp_src_idx = self.data_interim.spt_unique_idx
                        # has to verify this, if no changes for indices, the indices should be the same
                        self.data_interim.idx_pts_in_spt_match[
                            temp_src_idx[num_unique_i]] = idx_pts_in_curr_patch_global

        # replace the previous local indices
        self.data_output.spt_match_idx_unique = spt_matches_idx_unique_among_neighbors_local
        temp_spt_match_idx_unique_global_src = self.data_interim.spt_idx_src_temp[0][
            spt_matches_idx_unique_among_neighbors_local[:, 0]]
        temp_spt_match_idx_unique_global_tgt = self.data_interim.spt_idx_tgt_temp[0][
            spt_matches_idx_unique_among_neighbors_local[:, 1]]
        self.data_output.spt_match_idx_unique_global = torch.cat((temp_spt_match_idx_unique_global_src[:, None],
                                                                  temp_spt_match_idx_unique_global_tgt[:,
                                                                  None]), dim=1)
        # spt_matches_iso_error_among_neighbors replace the previous iso error
        self.data_output.spt_match_iso_error = spt_matches_iso_error_among_neighbors
        # replace the previous spt match coord.
        self.data_output.spt_match_coord_unique[:, 3:6] = self.data_interim.coord_aggregate_spt_tgt[
            spt_matches_idx_unique_among_neighbors_local[:, 1]]

        if self.method.remove_spt_match_with_iso_error:
            mask_iso_error = spt_matches_iso_error_among_neighbors < self.para.iso_error_threshold
            self.data_output.spt_match_iso_error = self.data_output.spt_match_iso_error[mask_iso_error]
            self.data_output.spt_match_coord_unique = self.data_output.spt_match_coord_unique[mask_iso_error]
            self.data_output.spt_match_idx_unique = self.data_output.spt_match_idx_unique[mask_iso_error]
            self.data_output.spt_match_idx_unique_global = self.data_output.spt_match_idx_unique_global[mask_iso_error]
            self.data_interim.spt_unique_idx = self.data_interim.spt_unique_idx[mask_iso_error]

        # visualization, visualize 1-to-1 patch matches
        if self.visualize.visualize_patch_after_union_close_neighbor_selection:
            if self.verbose:
                self.logging.info(" Visualizing unique patch matches after close neighbor selection... ")
            # visualize refine coarse matches
            # visualize first 10 src - tgt matches after mutula selection, no need for tgt-based
            for spt_unique_i in torch.arange(0, self.visualize.num_of_visualize_samples):
                self._spt_matching_visualization_one2one_match(spt_unique_i)

    def _compute_isometry_euclidean_error(self, idx_pts_in_curr_patch_global):
        pts_coord_in_curr_patch_src = self.data_input.pts_coord_src[idx_pts_in_curr_patch_global[:, 0]]
        pts_coord_in_curr_patch_tgt = self.data_input.pts_coord_tgt[idx_pts_in_curr_patch_global[:, 1]]

        # Calculate pairwise distances in Tensor1
        distances_1 = torch.linalg.norm(
            pts_coord_in_curr_patch_src[:, np.newaxis, :] - pts_coord_in_curr_patch_src[np.newaxis, :, :], axis=2)
        # Calculate pairwise distances in Tensor2
        distances_2 = torch.linalg.norm(
            pts_coord_in_curr_patch_tgt[:, np.newaxis, :] - pts_coord_in_curr_patch_tgt[np.newaxis, :, :], axis=2)

        if self.method.isometry_error_type == 'mean_squared_deviation':
            # Compute the squared differences between corresponding distances
            squared_differences = (distances_2 - distances_1) ** 2

            # Calculate the Mean Squared Deviation
            deviation = torch.mean(squared_differences)
        elif self.method.isometry_error_type == 'supremum_deviation':
            # Compute the absolute differences between corresponding distances
            absolute_differences = torch.abs(distances_2 - distances_1)
            # Find the maximum of these absolute differences
            deviation = torch.max(absolute_differences)
        elif self.method.isometry_error_type == 'l1_norm_deviation':
            # Compute the absolute differences between corresponding distances
            absolute_differences = torch.abs(distances_2 - distances_1)
            # Sum these absolute differences
            deviation = torch.sum(absolute_differences)
        elif self.method.isometry_error_type == 'hausdorff_distance':
            distances_1_to_2 = torch.norm(pts_coord_in_curr_patch_src[:, None] - pts_coord_in_curr_patch_tgt[None, :],
                                          dim=-1)
            distances_2_to_1 = torch.norm(pts_coord_in_curr_patch_tgt[:, None] - pts_coord_in_curr_patch_src[None, :],
                                          dim=-1)
            # Calculate Hausdorff distances
            hausdorff_1_to_2 = torch.max(torch.min(distances_1_to_2, dim=1)[0])
            hausdorff_2_to_1 = torch.max(torch.min(distances_2_to_1, dim=1)[0])
            ##########

            # The Hausdorff distance is the maximum of the two directions
            deviation = torch.max(hausdorff_1_to_2, hausdorff_2_to_1)
        elif self.method.isometry_error_type == 'distortion':
            # Calculate pairwise distances in Tensor1
            distances_1 = torch.norm(pts_coord_in_curr_patch_src[:, None] - pts_coord_in_curr_patch_src[None, :],
                                     dim=-1)
            # Calculate pairwise distances in Tensor2
            distances_2 = torch.norm(pts_coord_in_curr_patch_tgt[:, None] - pts_coord_in_curr_patch_tgt[None, :],
                                     dim=-1)

            # Compute the ratios of corresponding distances
            ratios = distances_2 / distances_1
            deviation_type = 'mean'
            if deviation_type == 'mean':
                # Mean distortion deviation
                deviation = torch.mean(torch.abs(ratios - 1))
            elif deviation_type == 'max':
                # Maximum distortion deviation
                deviation = torch.max(torch.abs(ratios - 1))
        elif self.method.isometry_error_type == 'energy_function':
            # Calculate squared distances in Tensor1
            squared_distances_1 = torch.sum(
                (pts_coord_in_curr_patch_src[:, None] - pts_coord_in_curr_patch_src[None, :]) ** 2, dim=-1)

            # Calculate squared distances in Tensor2
            squared_distances_2 = torch.sum(
                (pts_coord_in_curr_patch_tgt[:, None] - pts_coord_in_curr_patch_tgt[None, :]) ** 2, dim=-1)

            # Compute the energy function deviation
            deviation = torch.mean((squared_distances_2 - squared_distances_1) ** 2)

        return deviation

    # one to multiple or multiple to one
    def _spt_matching_visualization(self, dict_temp, search_type):
        """
        Visualize using spt matching.
        :input: dict_temp.idx_origin, the corresponding indices of the current spt point
        :input: dict_temp.i, the index of the current spt point
        :return:
        """
        # indices = dict_temp.indices
        idx_origin = dict_temp.idx_origin
        i = dict_temp.i
        small_region = self.visualize.small_region
        # only use 20 x 20 m base pcd for accelerate visualization

        if search_type == 'src':
            coord_spt_src_single = dict_temp.coord_spt_src_single
            small_base_pts_src, small_base_pts_tgt = self._crop_small_point_cloud_for_visualization(
                current_spt_pts_src=coord_spt_src_single,
                current_spt_pts_tgt=self.data_interim.coord_aggregate_spt_tgt[
                    idx_origin[0] if idx_origin.numel() > 1 else idx_origin.item()],
                small_region=small_region)

            pts_idx_in_curr_spt_match_src = (
                    self.data_interim.spt_idx_src_temp[1] == self.data_interim.spt_idx_src_temp[0][i])
            pts_in_curr_spt_match_src = self.data_input.pts_coord_src[pts_idx_in_curr_spt_match_src]
            # visualize multiple tgt_patch for src_patch, limit: no mutual consideration
            pts_in_curr_spt_match_tgt = []
            # find corresponding k tgt patch
            # len(idx_origin) may be smaller than len(indices[0]) if num. of feat. < k
            for tgt_idx_i in np.arange(idx_origin.numel()):
                pts_idx_in_curr_spt_match_tgt = self.data_interim.spt_idx_tgt_temp[1] == \
                                                self.data_interim.spt_idx_tgt_temp[0][
                                                    idx_origin[
                                                        tgt_idx_i] if idx_origin.numel() > 1 else idx_origin.item()]
                pts_in_curr_spt_match_tgt_i = self.data_input.pts_coord_tgt[pts_idx_in_curr_spt_match_tgt, :3]
                pts_in_curr_spt_match_tgt.append(pts_in_curr_spt_match_tgt_i)

            # visualize patch matches
            if ((self.visualize.visualize_patch and self.visualize.visualize_patch_one_src_to_multiple_tgt) or
                    self.visualize.visualize_patch_after_mutual_selection):
                visualize_patch_match(small_base_pts_src, small_base_pts_tgt, pts_in_curr_spt_match_src,
                                      pts_in_curr_spt_match_tgt, offset=[0, 50, 0],
                                      window_name=f'spt matches with distance constraint {self.method.dist_constraint}, '
                                                  f'{search_type}, num. of k: {idx_origin.numel()}'
                                                  f'spt match local indices: src: [{i}] - tgt: {idx_origin.tolist()}')
        elif search_type == 'tgt':
            coord_spt_tgt_single = dict_temp.coord_spt_tgt_single
            small_base_pts_src, small_base_pts_tgt = self._crop_small_point_cloud_for_visualization(
                current_spt_pts_src=self.data_interim.coord_aggregate_spt_src[idx_origin[0] if idx_origin.numel() > 1
                else idx_origin.item()],
                current_spt_pts_tgt=coord_spt_tgt_single,
                small_region=small_region)

            pts_idx_in_curr_spt_match_tgt = self.data_interim.spt_idx_tgt_temp[1] == \
                                            self.data_interim.spt_idx_tgt_temp[0][i]
            pts_in_curr_spt_match_tgt = self.data_input.pts_coord_tgt[pts_idx_in_curr_spt_match_tgt, :3]
            # visualize multiple src_patch for tgt_patch, with mutual consideration
            pts_in_curr_spt_match_src = []
            # find corresponding k tgt patch
            # len(idx_origin) may be smaller than len(indices[0]) if num. of feat. < k
            for src_idx_i in np.arange(idx_origin.numel()):
                pts_idx_in_curr_spt_match_src = self.data_interim.spt_idx_src_temp[1] == \
                                                self.data_interim.spt_idx_src_temp[0][
                                                    idx_origin[
                                                        src_idx_i] if idx_origin.numel() > 1 else idx_origin.item()]
                pts_in_curr_spt_match_src_i = self.data_input.pts_coord_src[pts_idx_in_curr_spt_match_src, :3]
                pts_in_curr_spt_match_src.append(pts_in_curr_spt_match_src_i)

            # visualize patch matches
            visualize_patch_match(small_base_pts_src, small_base_pts_tgt, pts_in_curr_spt_match_src,
                                  pts_in_curr_spt_match_tgt, offset=[0, 50, 0],
                                  # src_color=[0, 0.839, 1.], tgt_color=[0.921, 0.569, 0],
                                  window_name=f'spt matches with distance constraint {self.method.dist_constraint}, '
                                              f'{search_type}, num. of k: {idx_origin.numel()}'
                                              f'spt match local indices: src: {idx_origin.tolist()} - tgt: [{i}]')

    def _spt_matching_visualization_one2one_match(self, i):
        """
        Visualize using spt matching.
        :input: dict_temp.idx_origin, the corresponding indices of the current spt point
        :input: dict_temp.i, the index of the current spt point
        :return:
        """
        # indices = dict_temp.indices
        small_region = self.para.small_region
        # only use 20 x 20 m base pcd for accelerate visualization

        small_base_pts_src, small_base_pts_tgt = self._crop_small_point_cloud_for_visualization(
            current_spt_pts_src=self.data_output.spt_match_coord_unique[i, :3],
            current_spt_pts_tgt=self.data_output.spt_match_coord_unique[i, 3:6],
            small_region=small_region)

        # spt_match_idx_unique_global = self.data_interim.spt_idx_src_temp[0][self.data_interim.spt_match_idx_unique[i][0]]
        pts_idx_in_curr_spt_match_src = (
                self.data_interim.spt_idx_src_temp[1] == self.data_output.spt_match_idx_unique_global[i][0])
        pts_in_curr_spt_match_src = self.data_input.pts_coord_src[pts_idx_in_curr_spt_match_src]

        pts_idx_in_curr_spt_match_tgt = (
                self.data_interim.spt_idx_tgt_temp[1] == self.data_output.spt_match_idx_unique_global[i][1])
        pts_in_curr_spt_match_tgt = self.data_input.pts_coord_tgt[pts_idx_in_curr_spt_match_tgt]

        visualize_patch_match(small_base_pts_src, small_base_pts_tgt, pts_in_curr_spt_match_src,
                              pts_in_curr_spt_match_tgt, offset=[0, 50, 0],
                              window_name=f'spt matches after unique selection, spt match local indices: '
                                          f'{self.data_output.spt_match_idx_unique[i]}, '
                                          f'iso error after xxx selection: {self.data_output.spt_match_iso_error[i]:.3f}')

    def _crop_small_point_cloud_for_visualization(self, current_spt_pts_src, current_spt_pts_tgt, small_region=20):
        """
        Visualize using small region instead of the entire point cloud.
        """
        dists_src = torch.cdist(current_spt_pts_src.unsqueeze(0), self.data_input_3d.src_pts[:])

        small_base_pts_src_idx = dists_src <= small_region
        small_base_pts_src_idx = small_base_pts_src_idx.squeeze().cpu()
        small_base_pts_src = self.data_input_3d.src_pts[small_base_pts_src_idx, :3]

        # spt_tgt_centroid[ind_origin[0][curr_tgt_idx].cpu()
        dists_tgt_temp = torch.cdist(current_spt_pts_tgt.unsqueeze(0), self.data_input_3d.tgt_pts[:, :3])

        small_base_pts_tgt_idx = dists_tgt_temp <= small_region
        small_base_pts_tgt_idx = small_base_pts_tgt_idx.squeeze().cpu()
        small_base_pts_tgt_temp = self.data_input_3d.tgt_pts[small_base_pts_tgt_idx, :3]

        small_base_pts_tgt = small_base_pts_tgt_temp
        return small_base_pts_src, small_base_pts_tgt

    def _sort_tensor2d(self, indices, return_tensor=True):
        """ Sort tensor or array according to 2d indices"""
        if not isinstance(indices, list):
            indices = indices.tolist()
        # Pair each index with its original position
        indexed_indices = [(idx, i) for i, idx in enumerate(indices)]

        # Sort the pairs by the index values
        sorted_indexed_indices = sorted(indexed_indices, key=lambda x: (x[0][0], x[0][1]))

        # Separate the sorted indices and the original positions
        indices_sorted = [idx for idx, _ in sorted_indexed_indices]
        original_positions = [i for _, i in sorted_indexed_indices]

        indices_sorted = np.asarray(indices_sorted)
        original_positions = np.asarray(original_positions)

        if return_tensor:
            indices_sorted = array2tensor(indices_sorted)
            original_positions = array2tensor(original_positions)

        return indices_sorted, original_positions
