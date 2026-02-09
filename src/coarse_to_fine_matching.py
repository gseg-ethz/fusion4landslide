# ---------------------------------------------------------------------------- #
# The general structure of the proposed fusion method, keep most of details in
# coarse_to-fine_matching_base.py
# ---------------------------------------------------------------------------- #
from src.coarse_to_fine_matching_base import Coarse2Fine_Base
import torch
import faiss   # pip install faiss-gpu
import open3d as o3d

def merge_correspondences_by_priority_cpu(corres_list):
    """
    Merge multiple levels of 3D correspondences by priority order.

    Args:
        corres_list (list of torch.Tensor): List of (N,6) tensors, where each row is (x, y, z, ...).

    Returns:
        torch.Tensor: Merged (M,6) tensor with unique points prioritized by level.
    """
    merged_xyz = set()  # Set to store unique (x, y, z) coordinates
    merged_corres = []  # List to store the final result

    for corres in corres_list:  # Process each level in priority order (level 0, level 1, level 2)
        xyz = corres[:, :3]  # Extract the XYZ coordinates
        xyz_tuples = set(
            map(tuple, xyz.cpu().numpy()))  # Convert the XYZ coordinates to tuples for set-based deduplication

        # Only keep points from the current level that are not already in merged_xyz
        new_indices = [i for i, p in enumerate(xyz_tuples) if p not in merged_xyz]

        if new_indices:  # If there are new points to add
            merged_corres.append(corres[new_indices])  # Append the unique correspondences from the current level
            merged_xyz.update(xyz_tuples)  # Update the set with the newly added points

    # Concatenate the unique correspondences from all levels
    return torch.cat(merged_corres, dim=0) if merged_corres else torch.empty((0, 6), device=corres_list[0].device)


# TODO: combine all nn search types, and take one of them based on the input variable
def merge_correspondences_by_priority_with_distance_threshold(
    corres_list, distance_threshold=1e-3, search_type="faiss"
):
    """
    Merge multiple levels of 3D correspondences by priority, considering distance threshold for duplicates.

    Args:
        corres_list (list of torch.Tensor): List of (N, 6) tensors, where each row is (x, y, z, ...).
        distance_threshold (float): The distance threshold to consider two points as duplicates.
        search_type (str): Search method: "cdist", "kdtree", or "faiss".

    Returns:
        torch.Tensor: Merged (M, 6) tensor with unique points prioritized by level.
    """
    merged_xyz = []
    merged_corres = []

    # Step 1: Merge level 0 first (highest priority)
    xyz_level0 = corres_list[0][:, :3].cpu().numpy().astype("float32")
    merged_xyz.append(xyz_level0)
    merged_corres.append(corres_list[0])

    # for different search methods,
    # "faiss" is good for 1M+ points run on gpu, "torch.cdist" is good for small number of points (10k),
    # "kdtree" is good for 10k - 500k points, but only run on cpu
    if search_type == "faiss":
        # Billion-scale similarity search with {GPUs}, 2019
        # link: https://github.com/facebookresearch/faiss?tab=readme-ov-file
        # FAISS Index
        dim = 3
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        index.add(xyz_level0)

    elif search_type == "kdtree":
        # Open3D KDTree
        kdtree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(xyz_level0))

    def is_duplicate(new_xyz):
        """Check for duplicates using selected search method."""
        if search_type == "cdist":
            merged_xyz_tensor = torch.tensor(
                torch.cat(merged_xyz, dim=0), dtype=torch.float32
            )
            D = torch.cdist(torch.tensor(new_xyz), merged_xyz_tensor)
            return (D.min(dim=1)[0] < distance_threshold).cpu()

        elif search_type == "kdtree":
            valid_mask = []
            for i in range(new_xyz.shape[0]):
                _, _, distances = kdtree.search_knn_vector_3d(new_xyz[i], 1)
                valid_mask.append(distances[0] > distance_threshold ** 2)
            return torch.tensor(valid_mask)

        elif search_type == "faiss":
            D, _ = index.search(new_xyz, 1)
            return torch.tensor(D[:, 0] < distance_threshold ** 2)

        return torch.zeros(new_xyz.shape[0], dtype=torch.bool)

    # Step 2: Process other levels
    for level in range(1, len(corres_list)):
        xyz_level = corres_list[level][:, :3].cpu().numpy().astype("float32")
        valid_mask = ~is_duplicate(xyz_level)
        new_xyz = xyz_level[valid_mask.numpy()]
        new_corres = corres_list[level][valid_mask]

        if search_type == "faiss":
            index.add(new_xyz)
        elif search_type == "kdtree":
            kdtree = o3d.geometry.KDTreeFlann(
                o3d.utility.Vector3dVector(torch.cat([torch.tensor(merged_xyz), torch.tensor(new_xyz)], dim=0).numpy())
            )

        merged_xyz.append(new_xyz)
        merged_corres.append(new_corres)

    return torch.cat(merged_corres, dim=0) if merged_corres else torch.empty((0, 6), device=corres_list[0].device)


def merge_correspondences_by_priority_with_distance_threshold_cdist(corres_list, distance_threshold=1e-3):
    """
    Merge multiple levels of 3D correspondences by priority, considering distance threshold for duplicates.
    This version uses GPU batch operations for efficient distance calculation.

    Args:
        corres_list (list of torch.Tensor): List of (N, 6) tensors, where each row is (x, y, z, ...).
        distance_threshold (float): The distance threshold to consider two points as duplicates.
        batch_size (int): The batch size for processing large point clouds in chunks.

    Returns:
        torch.Tensor: Merged (M, 6) tensor with unique points prioritized by level.
    """
    merged_xyz = []  # List to store unique (x, y, z) coordinates
    merged_corres = []  # List to store the final result

    # Step 1: First, merge level 0 (most detailed) points
    xyz_level0 = corres_list[0][:, :3].cpu()  # Extract the XYZ coordinates from level 0
    merged_xyz.append(xyz_level0)  # Add all level 0 points
    merged_corres.append(corres_list[0])  # Keep all correspondences from level 0

    # Helper function to compute distances and check duplicates in batches
    def is_duplicate_batch(new_xyz, current_xyz, threshold, batch_size):
        num_new_points = new_xyz.size(0)
        num_existing_points = current_xyz.size(0)

        # Preallocate result
        duplicates = torch.zeros(num_new_points, dtype=torch.bool, device=new_xyz.device)

        # Iterate over batches of new points
        for i in range(0, num_new_points, batch_size):
            new_batch = new_xyz[i:i + batch_size]  # Current batch of new points
            dist_batch = torch.cdist(new_batch,
                                     current_xyz)  # Compute pairwise distances (batch_size, num_existing_points)

            # Find if any distance is smaller than the threshold
            min_dist, _ = dist_batch.min(dim=1)  # Find the minimum distance for each new point
            duplicates[i:i + batch_size] = min_dist < threshold  # Mark duplicates based on distance threshold

        return duplicates

    batch_size = 1024
    # Step 2: Then, process level 1 points, add only if not already in merged_xyz
    xyz_level1 = corres_list[1][:, :3].cpu()  # Extract the XYZ coordinates from level 1
    if len(merged_xyz) > 0:
        merged_xyz_tensor = torch.cat(merged_xyz, dim=0)  # Combine all previously merged points
        duplicates = is_duplicate_batch(xyz_level1, merged_xyz_tensor, distance_threshold, batch_size)
        new_points = xyz_level1[~duplicates]  # Only keep non-duplicate points
        new_corres = corres_list[1][~duplicates]  # Correspondences for new points
    else:
        new_points = xyz_level1
        new_corres = corres_list[1]

    merged_xyz.append(new_points)  # Add new non-duplicate points to merged_xyz
    merged_corres.append(new_corres)  # Keep new correspondences

    # Step 3: Finally, process level 2 points, add only if not already in merged_xyz
    xyz_level2 = corres_list[2][:, :3].cpu()  # Extract the XYZ coordinates from level 2
    if len(merged_xyz) > 0:
        merged_xyz_tensor = torch.cat(merged_xyz, dim=0)  # Combine all previously merged points
        duplicates = is_duplicate_batch(xyz_level2, merged_xyz_tensor, distance_threshold, batch_size)
        new_points = xyz_level2[~duplicates]  # Only keep non-duplicate points
        new_corres = corres_list[2][~duplicates]  # Correspondences for new points
    else:
        new_points = xyz_level2
        new_corres = corres_list[2]

    merged_xyz.append(new_points)  # Add new non-duplicate points to merged_xyz
    merged_corres.append(new_corres)  # Keep new correspondences

    # Step 4: Return the final merged correspondences
    return torch.cat(merged_corres, dim=0) if merged_corres else torch.empty((0, 6), device=corres_list[0].device)


class Coarse2Fine(Coarse2Fine_Base):
    # inherit from the base class
    def __init__(self, config):
        super(Coarse2Fine, self).__init__(config)
        self.debugging = config.debugging

    def implement_c2f_matching(self):

        if self.method.use_2d_matches and not (self.method.coarse_matching_only_3d and self.method.fine_matching_only_3d):
            self.global_matches_from_2d_with_different_types()
            # self.implement_2d_matching()
            # self.implement_2d_matching_mutual()
        else:
            if self.verbose:
                self.logging.info('Skip 2d matching!')

        self._voxel_subsampling()

        self.implement_partition()
        self.load_partition()
        if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
            # stack the outputs of different levels of superpoint partition
            self.data_interim.idx_spt2pts_src_multiple = []
            self.data_interim.idx_spt2pts_tgt_multiple = []
            for level_current in self.method.level_of_superpoint:
                self.data_interim.idx_pts2spt_src = self.data_interim.idx_pts2spt_src_multiple[level_current - 1]
                self.data_interim.idx_pts2spt_tgt = self.data_interim.idx_pts2spt_tgt_multiple[level_current - 1]
                self.prepare_pts2spt_dict()
                self.data_interim.idx_spt2pts_src_multiple.append(self.data_interim.idx_spt2pts_src)
                self.data_interim.idx_spt2pts_tgt_multiple.append(self.data_interim.idx_spt2pts_tgt)
        else:
            self.prepare_pts2spt_dict()

        # reduce numbers when debugging
        if self.debugging.use_debugging:
            self.start_debugging('reduce_num_spt')

        ### 3d
        if self.method.coarse_matching_only_2d and self.method.fine_matching_only_2d:
            if self.verbose:
                self.logging.info('Skip computing point features!')
        else:
            self.compute_point_feat()

            self.global_matches_from_3d()

        # TODO: implement the following functions
        if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
            # stack the outputs of different levels of superpoint partition
            self.data_output.spt_corres_src_multiple = []
            self.data_output.spt_corres_tgt_multiple = []
            for level_current in self.method.level_of_superpoint:
                self.data_interim.idx_pts2spt_src = self.data_interim.idx_pts2spt_src_multiple[level_current - 1]
                self.data_interim.idx_pts2spt_tgt = self.data_interim.idx_pts2spt_tgt_multiple[level_current - 1]
                self.data_interim.idx_spt2pts_src = self.data_interim.idx_spt2pts_src_multiple[level_current - 1]
                self.data_interim.idx_spt2pts_tgt = self.data_interim.idx_spt2pts_tgt_multiple[level_current - 1]
                self.coarse_matching_with_different_types()
                # multiple outputs from coarse matching
                self.data_output.spt_corres_src_multiple.append(self.data_output.spt_corres_src)
                self.data_output.spt_corres_tgt_multiple.append(self.data_output.spt_corres_tgt)
        else:
            self.coarse_matching_with_different_types()

        # fine matching
        if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
            # stack the outputs of different levels of superpoint partition
            self.data_output.corres_3d_refine_apply_icp_multiple = []
            if self.method.output_tgt2src:
                self.data_output.corres_3d_refine_apply_icp_tgt2src_multiple = []
            self.data_output.corres_3d_refine_apply_icp_discrete_multiple = []
            for level_current in self.method.level_of_superpoint:
                self.data_output.spt_corres_src = self.data_output.spt_corres_src_multiple[level_current - 1]
                self.data_output.spt_corres_tgt = self.data_output.spt_corres_tgt_multiple[level_current - 1]
                self.fine_matching_with_different_types()
                # multiple outputs from fine matching
                self.data_output.corres_3d_refine_apply_icp_multiple.append(self.data_output.corres_3d_refine_apply_icp)
                if self.method.output_tgt2src:
                    self.data_output.corres_3d_refine_apply_icp_tgt2src_multiple.append(self.data_output.corres_3d_refine_apply_icp_tgt2src)
                self.data_output.corres_3d_refine_apply_icp_discrete_multiple.append(self.data_output.corres_3d_refine_apply_icp_discrete)
        else:
            self.fine_matching_with_different_types()

        # TODO: better to put this in the base class
        # merge correspondences
        if self.method.partition_type == 'superpoint' and isinstance(self.method.level_of_superpoint, list):
            if self.verbose:
                self.logging.info('Start merging correspondences...')
            # deal with multiple outputs from fine matching
            self.data_output.corres_3d_refine_apply_icp = merge_correspondences_by_priority_with_distance_threshold(self.data_output.corres_3d_refine_apply_icp_multiple)
            if self.method.output_tgt2src:
                self.data_output.corres_3d_refine_apply_icp_tgt2src = merge_correspondences_by_priority_with_distance_threshold(self.data_output.corres_3d_refine_apply_icp_tgt2src_multiple)
            self.data_output.corres_3d_refine_apply_icp_discrete = merge_correspondences_by_priority_with_distance_threshold(self.data_output.corres_3d_refine_apply_icp_discrete_multiple)

        # save results to path
        if self.data_output.corres_3d_refine_apply_icp != []:
            self.save_process_dvf()

    def start_debugging(self, debug_stage):
        """ Start fast debugging with small samples """
        if debug_stage == 'reduce_num_spt':
            spt_idx_src = list(self.data_input.spt_idx_src)
            spt_idx_tgt = list(self.data_input.spt_idx_tgt)

            spt_idx_src[0] = spt_idx_src[0][:self.debugging.num_spt]
            spt_idx_tgt[0] = spt_idx_tgt[0][:self.debugging.num_spt]

            spt_idx_src[2] = len(spt_idx_src[0])
            spt_idx_tgt[2] = len(spt_idx_tgt[0])

            self.data_interim.spt_idx_src_temp = tuple(spt_idx_src)
            self.data_interim.spt_idx_tgt_temp = tuple(spt_idx_tgt)

            if self.verbose:
                self.logging.info(f'Debug with smaller number of superpoints {len(self.data_interim.spt_idx_src_temp)}')
