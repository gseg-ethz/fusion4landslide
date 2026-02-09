import os
import os.path as osp
import time
import open3d as o3d
from src.data_loader import Preprocess_Dataset
from utils.common import dir_exist, read_pcd
from utils.o3d_tools import array2tensor
import torch
import numpy as np
import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from src.functions import transform_point_cloud, compute_c2c
import re
import hnswlib
from cpp_core.supervoxel_segmentation.build import supervoxel as svl_segmentation
from collections import Counter

class Deformation_Analyze(object):
    def __init__(self, config, src_tile_overlap_path, tgt_tile_overlap_path):
        # self.preprocess can be used out of this class
        # self.preprocess = self._preprocess_data() only used inside this class for different inside functions

        self.config = config
        self.logging = config.logging
        self.verbose = config.verbose
        self.voxel_size = config.voxel_size
        self.points_per_batch = config.points_per_batch
        self.device = self.config.device
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        # self.src_tile_overlap_path = src_tile_overlap_path
        # self.tgt_tile_overlap_path = tgt_tile_overlap_path
        self.src_tile_overlap_pcd = read_pcd(src_tile_overlap_path)
        self.tgt_tile_overlap_pcd = read_pcd(tgt_tile_overlap_path)

        # use overlap pcd for analysis
        self.src_tile_non_overlap_path = src_tile_overlap_path
        self.tgt_tile_non_overlap_path = tgt_tile_overlap_path
        self.src_tile_non_overlap_pcd = read_pcd(self.src_tile_non_overlap_path)
        self.tgt_tile_non_overlap_pcd = read_pcd(self.tgt_tile_non_overlap_path)
        # use non overlap pcd for analysis
        # self.src_tile_non_overlap_path = (src_tile_overlap_path.replace('_overlap', '').replace('overlap', 'non_overlap'))
        # self.tgt_tile_non_overlap_path = (tgt_tile_overlap_path.replace('_overlap', '').replace('overlap', 'non_overlap'))
        # self.src_tile_non_overlap_pcd = read_pcd(self.src_tile_non_overlap_path)
        # self.tgt_tile_non_overlap_pcd = read_pcd(self.tgt_tile_non_overlap_path)

        self.tile_id = config.tile_id
        # self.tile_id = re.findall(r'\d+', os.path.basename(src_tile_overlap_path))[0]

        self.output_path = osp.join(config.output_dir, config.output_folder)

        self.src_tile_feat = None
        self.tgt_tile_feat = None
        self.correspondences = None

        self.feat_compute = config.feat_compute
        self.pcd_segment = config.pcd_segment
        self.outlier_removal = config.outlier_removal

        self.sv_type = None
        self.segment_type = self.config.segment_type
        self.save_interim = self.config.save_interim
        self.small_patch_removal = self.config.small_patch_removal

        self.corr_path = osp.join(self.output_path, 'correspondences', f'corr_tile_{self.tile_id}.npz')

    @property
    def _feat_path(self) -> str:
        return osp.join(self.output_path, 'features', f'features_tile_{self.tile_id}.npz')

    @property
    def _segment_path(self) -> str:
        if self.segment_type == 'supervoxel':
            segment_folder = 'svl_segment'
        elif self.segment_type == 'superpoint':
            segment_folder = osp.join('spt_segment', self.config.spt_color_level)
        else:
            raise NotImplementedError
        return osp.join(self.output_path, segment_folder, f'segment_tile_{self.tile_id}.txt')

    def _feat_desc_nn(self, input):
        if self.feat_compute:
            return self.config.feat_desc_nn(input)

    def _outlier_removal_nn(self):
        if self.outlier_removal:
            return self.config.outlier_removal_nn

    def compute_features(self):
        """
        Compute features using either local patch-based or fully-convolutional methods
        options: fpfh, gfm, local-patch-based, coarse-to-fine methods
        :return:
        """
        feat_compute = self.config.feat_compute
        if not feat_compute:
            if not osp.exists(self._feat_path):
                raise FileNotFoundError(f"The feature path '{self._feat_path}' is not found")
            self._load_features()

        elif self.config.feat_type == 'DIPs':
            src_tile_feat, tgt_tile_feat = [], []
            # 10 used in the paper
            neigh_radius = np.sqrt(3) * (10 * self._compute_median_resolution())
            # neigh_radius = np.sqrt(3) * (3 * self._compute_median_resolution())

            # If running into the GPU memory problems reduce the number of points in a batch (default is 2000).
            # which point cloud is used for feature computation, non-overlap or overlap one?
            src_batch = Preprocess_Dataset(self.src_tile_non_overlap_pcd, self.src_tile_overlap_pcd,
                                           self.points_per_batch, neigh_radius)
            tgt_batch = Preprocess_Dataset(self.tgt_tile_non_overlap_pcd, self.tgt_tile_overlap_pcd,
                                           self.points_per_batch, neigh_radius)

            src_batch = torch.utils.data.DataLoader(src_batch, batch_size=self.batch_size,
                                                    num_workers=self.num_workers, shuffle=False,  drop_last=False)
            tgt_batch = torch.utils.data.DataLoader(tgt_batch, batch_size=self.batch_size,
                                                    num_workers=self.num_workers, shuffle=False, drop_last=False)

            for src_i in tqdm(src_batch, position=0, leave=True):
                src_i = src_i.squeeze(0).cuda()
                # (1, 1000, 3, 256) --> (1, 1000, 64)
                src_i, _, _ = self._feat_desc_nn(src_i)

                src_tile_feat.append(src_i)
            for tgt_i in tqdm(tgt_batch, position=0):
                tgt_i = tgt_i.squeeze(0).cuda()
                tgt_i, _, _ = self._feat_desc_nn(tgt_i)

                tgt_tile_feat.append(tgt_i)

            self.src_tile_feat = torch.cat(src_tile_feat, dim=0)
            self.tgt_tile_feat = torch.cat(tgt_tile_feat, dim=0)

            if self.verbose:
                self.logging.info('Compute features using local patch-based methods is done!')

            if self.save_interim:
                dir_exist(osp.dirname(self._feat_path))
                save_type = '.npz'
                if save_type == '.npz':
                    np.savez_compressed(
                        self._feat_path,
                        src_feat=self.src_tile_feat.cpu(),
                        tgt_feat=self.tgt_tile_feat.cpu()
                    )
                if self.verbose:
                    self.logging.info('Save features npz file to %s.', self._feat_path)
        else:
            raise NotImplementedError
            # self.logging.error('Feature descriptor type %s is not supported', self.config.feat_type)

        return None

    def _load_features(self):
        features = np.load(self._feat_path)
        self.src_tile_feat = array2tensor(features['src_feat']).to(self.device)
        self.tgt_tile_feat = array2tensor(features['tgt_feat']).to(self.device)

        if self.verbose:
            self.logging.info(f'Skip feature computation for tile {self.tile_id}. '
                              f'Features will be loaded from: {self._feat_path}')
        return None

    def implement_segmentation(self):
        svl_idx = []
        if not self.pcd_segment:
            if not osp.exists(self._segment_path):
                raise FileNotFoundError(f"The segmentation result path '{self._segment_path}' is not found")
            svl_idx = self._load_segmentation_results()[:, -1]

        elif self.segment_type == 'supervoxel':
            """
            Segmentation using supervoxel method,
            Computes boundary perserving supervoxels (i.e. local patches of geometrically coherent points) 
            based on the method proposed in [2].
            [2] Lin, Y., et al.: Toward better boundary preserved supervoxel segmentation for 3D point clouds. 
            ISPRS journal of photogrammetry and remote sensing, 2018.
            Approximate supervoxel radius is defined for each tile independently based on the median point cloud resolution
            I changes please consider that this has to be a reasonable value (i.e. size of the patch that moves as rigid body)
            """
            svl_radius = np.max((np.sqrt(3) * (10 * self._compute_median_resolution()), self.voxel_size,))
            # keep the same with F2S3
            if 'Rockfall_Simulator' in self.output_path:
                svl_radius = 0.1

            svl_idx = svl_segmentation.computeSupervoxel(self.src_tile_non_overlap_path, self.config.n_normals,
                                                         svl_radius, self._segment_path)
            if self.verbose:
                # self.logging.info('----------------------------------------------------------------------')
                self.logging.info(f'Save segmentation to: {self._segment_path}.')

        elif self.segment_type == 'superpoint':
            self.logging.info('Load the trained weight of superpoint segmentation from offline spt')
            svl_idx = self._load_segmentation_results()[:, -2]
        elif self.segment_type == 'dbscan':
            """
            Segmentation using DBSCAN method,
            https://github.com/yanconglin/ICP-Flow/blob/main/utils_cluster.py"""
            return None
        elif self.segment_type == 'hdbscan':
            """
            Segmentation using HDBSCAN method,
            https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
            https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#sphx-glr-auto-examples-cluster-plot-hdbscan-py

            """
            return None
        else:
            raise NotImplementedError('Segmentation type %s is not supported', self.segment_type)

        svl_idx = np.asarray(svl_idx).reshape(-1, 1)

        # Extract the indices of individual supervoxels
        svl_type_raw = np.unique(svl_idx)

        # Count the occurrences of each index
        svl_idx = svl_idx.flatten()
        counts = Counter(svl_idx)
        # Only keep indices that occur more than 10 times
        if self.small_patch_removal:
            valid_indices = {idx for idx, count in counts.items() if count > 10}
        else:
            valid_indices = {idx for idx, count in counts.items() if count > 1}

        if self.verbose:
            # self.logging.info('----------------------------------------------------------------------')
            self.logging.info(f'Remove supervoxels that contain less than 10 points, {len(valid_indices)} of {len(counts)}')

        # Find the valid indices and store them
        svl_type = []
        for idx in tqdm(svl_type_raw, position=0, leave=True):
            if idx in valid_indices:
                svl_type.append(np.where(svl_idx == idx)[0])

        self.svl_type = svl_type
        # self.valid_indices = np.hstack(svl_type)

    def _load_segmentation_results(self):
        segment_results = np.loadtxt(self._segment_path)

        if self.verbose:
            self.logging.info(f'Skip segmentation for tile {self.tile_id}. '
                              f'Segmentation results will be loaded from: {self._segment_path}')
        return segment_results

    def correspondence_searching(self):
        """
        Perform the correspondence searching using knn, or skipped when using coarse-to-fine methods.
        source: [f2s3]
        Approximate correspondence search in the feature space using the hierarchical navigable small world graph method proposed in [4].

        [4] Malkov, Y. A. et al.: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI, 2018.
        :return:
        """
        correspondence_searching = self.config.correspondence_searching
        if not correspondence_searching:
            if not osp.exists(self.corr_path):
                if osp.exists(self.corr_path.replace('.npz', '.txt')):
                    self.corr_path = self.corr_path.replace('.npz', '.txt')
                else:
                    raise FileNotFoundError(f"The correspondence path '{self.corr_path}' is not found")
            self._load_correspondences()
            if self.verbose:
                # self.logging.info('----------------------------------------------------------------------')
                self.logging.info(f'Skip correspondence searching. Load it from path')
        else:
            if self.verbose:
                # self.logging.info('----------------------------------------------------------------------')
                self.logging.info(f'Start correspondence searching ...')
            # Intitialize the library, specify the space, the type of the vector and add data points
            p = hnswlib.Index(space='l2', dim=64)  # possible options are l2, cosine or ip
            p.init_index(max_elements=self.tgt_tile_feat.shape[0], ef_construction=self.config.efC, M=self.config.M)
            p.set_ef(self.config.efS)
            p.set_num_threads(self.config.num_threads)
            p.add_items(self.tgt_tile_feat.cpu().numpy())

            # Query the elements for themselves and measure recall:
            # knn based correspondence searching
            labels, distances = p.knn_query(self.src_tile_feat.cpu().numpy(), k=1)

            # Save the correspondences for subsequent steps
            self.correspondences = np.concatenate((np.array(self.src_tile_non_overlap_pcd.points),
                                                   np.array(self.tgt_tile_non_overlap_pcd.points)[labels.reshape(-1), :]), axis=1)
            magnitude = np.linalg.norm(self.correspondences[:, :3] - self.correspondences[:, 3:6], axis=1)
            interim_dvfms = np.hstack((self.correspondences[:, :3], magnitude[:, None]))
            dir_exist(osp.join(self.output_path, 'results'))
            np.savetxt(osp.join(self.output_path, 'results', f'f2s3_dvfms_without_pruning_of_tile_{self.tile_id}.txt'), interim_dvfms)

            max_mag_visualize = 5
            interim_dvfms[0, 3] = 0
            interim_dvfms[1, 3] = max_mag_visualize
            np.savetxt(osp.join(self.output_path, 'results', f'f2s3_dvfms_without_pruning_of_tile_{self.tile_id}_visualize_0_5.txt'), interim_dvfms)

            if self.save_interim:
                # np.savez(osp.join(self.config.corr_dir, f'corr_tile_{self.tile_id}.npz'), corr=self.correspondences)
                np.savez(self.corr_path, corr=self.correspondences)

    def _load_correspondences(self):
        add_img_corr = False
        if add_img_corr:
            if osp.splitext(self.corr_path)[1] == '.npz':
                corr = np.load(self.corr_path)
            else:
                corr = np.loadtxt(self.corr_path)
            corr_img_dir = osp.dirname(self.corr_path)
            corr_img_name = osp.splitext(osp.basename(self.corr_path))[0]
            corr_img = np.loadtxt(osp.join(corr_img_dir, f"{corr_img_name}_img.txt"))
            indices_valid_2d = np.loadtxt(osp.join(corr_img_dir, "indices_tile_0_2d_valid_img.txt"))
            corr_img = array2tensor(corr_img).to(self.device)
            self.correspondences = array2tensor(corr['corr']).to(self.device)
            self.correspondences[indices_valid_2d, :] = corr_img
        else:
            if osp.splitext(self.corr_path)[1] == '.npz':
                corr = np.load(self.corr_path)
            else:
                corr = np.loadtxt(self.corr_path)
            self.correspondences = array2tensor(corr['corr']).to(self.device)

    def correspondence_pruning(self):
        """
        Perform correspondence pruning, using segmentation for local patch rigidity checking,
        mean magnitude suppression for each tile
        :return:
        """
        # start_time = time.time()
        inlier_idx = []
        save_coords = []

        # if self.verbose:
        #     supervoxel_iter = tqdm(self.supervoxels, ncols=90)
        # else:
        #     supervoxel_iter = self.supervoxels

        if self.verbose:
            # self.logging.info('----------------------------------------------------------------------')
            self.logging.info(f'Start correspondence pruning ...')

        for svl in tqdm(self.svl_type, position=0):
            svl_data = torch.tensor(self.correspondences[svl, :]).cuda()
            # svl_data_scaled = np.divide(svl_data, np.max(np.abs(svl_data)))
            svl_data_scaled = torch.divide(svl_data, torch.max(torch.abs(svl_data)))

            filtering_output = self._outlier_removal_nn().filter_input(
                svl_data_scaled.unsqueeze(0).unsqueeze(0).float(),
                svl_data.unsqueeze(0).float(), self.config)

            svl_coords = svl_data

            if filtering_output['robust_estimate'] and self.config.refine_results:
                x1t = transform_point_cloud(
                    svl_data[:, 0:3].float(),
                    filtering_output['rot_est'],
                    filtering_output['trans_est']).cpu().numpy()

                svl_coords = svl_coords.cpu().numpy()
                svl_coords[:, 3:6] = x1t

                idx = np.ones(svl_coords.shape[0])

            else:
                idx = (filtering_output['scores'].reshape(-1) > 0.99999).cpu().numpy()

            inlier_idx.append(idx)
            save_coords.append(svl_data)
            # time.sleep(0.01)

        # torch.cuda.empty_cache()
        # gc.collect()

        if inlier_idx:
            inlier_idx = np.concatenate(inlier_idx, axis=0)
            inlier_idx = np.where(inlier_idx > 0.5)[0].reshape(-1)

            save_coords = torch.concatenate(save_coords, axis=0)
            save_coords = save_coords.cpu().numpy()

        # Filter the outliers based on the predicted scores, may have error when using 0_level_colors
        filtered_results = save_coords[inlier_idx, :]
        filtered_magnitudes = np.linalg.norm(filtered_results[:, 3:6] - filtered_results[:, 0:3], axis=1)

        if self.verbose:
            self.logging.info('{} points out of {} were classified as inlier'.format(filtered_results.shape[0],
                                                                                         save_coords.shape[0]))

        save_path_output = os.path.join(self.output_path, 'results')

        dir_exist(save_path_output)

        final_results = np.concatenate((filtered_results, filtered_magnitudes.reshape(-1, 1)), axis=1)
        mask = final_results[:, 6] <= self.config.max_disp_magnitude
        final_results = final_results[mask]

        np.savetxt(os.path.join(save_path_output, 'f2s3_dvfs_of_tile_{}.txt'.format(self.tile_id)), final_results[:, :6])
        np.savetxt(os.path.join(save_path_output, 'f2s3_dvfms_of_tile_{}.txt'.format(self.tile_id)),
                   final_results[:, [0, 1, 2, 6]])

        max_mag_visualize = 5
        if final_results.shape[0] > 2:
            final_results[0, 6] = 0
            final_results[1, 6] = max_mag_visualize
            np.savetxt(osp.join(save_path_output, f'f2s3_dvfms_of_tile_{self.tile_id}_visualize_0_5.txt'), final_results[:, [0, 1, 2, 6]])

        # ###################
        # # find the indices of final output src_pts in raw src pcd
        # raw_src_pcd_path = os.path.join(self.config.data_dir, self.config.src_name)
        # raw_src_pcd = o3d.io.read_point_cloud(raw_src_pcd_path)
        # from scipy.spatial import cKDTree
        #
        # s_tree = cKDTree(raw_src_pcd.points)
        #
        # distances, indices = s_tree.query(final_results[:, :3], k=1, workers=-1)
        # np.savetxt(os.path.join(save_path_output, 'diplacement_indices_{}.txt'.format(self.tile_id)),
        #            indices)
        # ###################

        # If maximum magnitude parameter is set, filter all points with larger magnitude estimates
        if self.config.max_disp_magnitude > 0:
            max_magnitude_idx = np.where(filtered_magnitudes < self.config.max_disp_magnitude)[0].reshape(-1)

            filtered_results = filtered_results[max_magnitude_idx, :]
            filtered_magnitudes = filtered_magnitudes[max_magnitude_idx]
            inlier_idx = inlier_idx[max_magnitude_idx].reshape(-1)

        # If filtered by magnitude is selected also filter very large motion inside a tile
        if self.config.filter_median_magnitude:
            if self.verbose:
                self.logging.info(
                    'Filtering the displacement vectors based on the mean magnitude of the displacement')

            # Compute the median magnitude
            median_mag = np.median(filtered_magnitudes)

            if self.verbose:
                self.logging.info(
                    'Median magnitude {}, all displacementes above {} m will be removed'.format(median_mag,
                                                                                                    30 * median_mag))

            mag_inlier = np.where(filtered_magnitudes < 30 * median_mag)[0]
            filtered_results = filtered_results[mag_inlier, :]
            filtered_magnitudes = filtered_magnitudes[mag_inlier]

            save_path_filtered_mag = os.path.join(save_path_output, 'filtered_by_magnitude')

            dir_exist(save_path_filtered_mag)

            np.savetxt(os.path.join(save_path_filtered_mag, 'f2s3_dvfms_filtered_by_median_mag_of_tile_{}.txt'.format(self.tile_id)),
                       np.concatenate((filtered_results[:, :3], filtered_magnitudes.reshape(-1, 1)), axis=1))

            # If selected combine the inliers estimated by out method with the C2C estimates for the outliers
            if self.config.fill_gaps_c2c:
                c2c_displacements = compute_c2c(save_coords[:, 0:3],
                                                np.asarray(self.tgt_tile_non_overlap_pcd.points)).reshape(-1)

                save_path_c2c = os.path.join(save_path_output, 'combined_with_c2c')

                dir_exist(save_path_c2c)

                c2c_displacements[inlier_idx[mag_inlier]] = filtered_magnitudes

                np.savetxt(os.path.join(save_path_c2c, 'f2s3_dvfms_combined_with_c2c_of_tile_{}.txt'.format(self.tile_id)),
                           np.concatenate((save_coords[:, 0:3], c2c_displacements.reshape(-1, 1)), axis=1))

        # If selected combine the inliers estimated by out method with the C2C estimates for the outliers
        elif self.config.fill_gaps_c2c:
            c2c_displacements = compute_c2c(save_coords[:, 0:3],
                                            np.asarray(self.tgt_tile_non_overlap_pcd.points)).reshape(-1)

            save_path_c2c = os.path.join(save_path_output, 'combined_with_c2c')
            if not os.path.exists(save_path_c2c):
                os.makedirs(save_path_c2c)

            c2c_displacements[inlier_idx] = filtered_magnitudes

            np.savetxt(os.path.join(save_path_c2c, 'f2s3_dvfms_combined_with_c2c_of_tile_{}.txt'.format(self.tile_id)),
                       np.concatenate((save_coords[:, 0:3], c2c_displacements.reshape(-1, 1)), axis=1))

        return None

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
        neigh.fit(np.array(self.src_tile_non_overlap_pcd.points))
        dist01, _ = neigh.kneighbors(np.array(self.src_tile_non_overlap_pcd.points), return_distance=True)

        # Resolution of the source point cloud
        resolution_s = np.median(dist01[:, -1])

        # Compute the point cloud resolution of the target point cloud (k=2 as the closest point is the point itself)
        neigh.fit(np.array(self.tgt_tile_non_overlap_pcd.points))
        dist01, _ = neigh.kneighbors(np.array(self.tgt_tile_non_overlap_pcd.points), return_distance=True)

        # Resolution of the target point cloud
        resolution_t = np.median(dist01[:, -1])

        tile_st_resolution = max(resolution_s, resolution_t)

        return tile_st_resolution