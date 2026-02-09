# ---------------------------------------------------------------------------- #
# Reproduce of Piecewise ICP for comparison
# ---------------------------------------------------------------------------- #
import copy
import os
import sys
import laspy
import open3d as o3d
import numpy as np
import os.path as osp
from utils.common import dir_exist

center_source = []
center_target = []


def Piecewise_ICP(cfg):
    """
    Implementation of the paper 'Identification of stable surfaces within point clouds for areal deformation monitoring, 2016'
    Code reproduced from Helena Laasch's Master thesis 'Registration of TLS Point Clouds for Long-Range Geomonitoring Applications, 2022'
    Modified by Zhaoyi Wang, October 15, 2024

    This function calculates the piecewise ICP registration.
    :param log_file: path of logfile
    :param file_source: path of source point cloud
    :param file_target: path of target point cloud
    :param trans_init: initial transformation matrix
    :param smax: maximum size of octree cell
    :param number_points_min: minimum number of points in a octree cell
    :param threshold: max. corresponding point pair distance for ICP
    :param results: folder to save results
    """

    # file_source, file_target, trans_init, smax, number_points_min, threshold, results, dataset = None

    file_source = osp.join(cfg.src_tile_overlap_path)
    file_target = osp.join(cfg.tgt_tile_overlap_path)
    trans_init = np.eye(3)
    smax = cfg.smax
    number_points_min = cfg.number_points_min
    threshold = cfg.threshold
    results = osp.join(cfg.output_root, 'results')
    dir_exist(results)

    # Function how to traverse through octree
    def f_traverse(node, node_info):
        early_stop = False

        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                # we only want to process nodes / spatial regions with enough points
                early_stop = len(node.indices) < 250
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
                if (len(node.indices) >= number_points_min):
                    if source_bool:
                        # extract point cloud of octree cell
                        tmp_pcl = pcl_source.select_by_index(node.indices)
                        # calculate mean of octree cell points
                        center = np.asarray(tmp_pcl.points).mean(axis=0)
                        # save centroid to list
                        global center_source
                        center_source.append(center)
                    else:
                        # extract point cloud of octree cell
                        tmp_pcl = pcl_target.select_by_index(node.indices)
                        # calculate mean of octree cell points
                        center = np.asarray(tmp_pcl.points).mean(axis=0)
                        # save centroid to list
                        global center_target
                        center_target.append(center)
        # early stopping: if True, traversal of children of the current node will be skipped
        return early_stop

    # dir_exist(results)
    cfg.logging.info('Start processing the current tile')

    # Import point cloud to be registered
    # las_source = laspy.read(file_source)
    pcl_source = o3d.io.read_point_cloud(file_source)
    # pcl_source_wt = o3d.geometry.PointCloud()
    # pcl_source_wt.points = o3d.utility.Vector3dVector(las_source.xyz)
    cfg.logging.info('Start processing the current tile')

    # Import reference point cloud
    # las_target = laspy.read(file_target)
    pcl_target = o3d.io.read_point_cloud(file_target)

    # get Bounding Box of both point clouds
    bb_target = pcl_target.get_axis_aligned_bounding_box()
    bb_source = pcl_source.get_axis_aligned_bounding_box()
    bb_target_min = bb_target.get_min_bound()
    bb_target_max = bb_target.get_max_bound()
    bb_source_min = bb_source.get_min_bound()
    bb_source_max = bb_source.get_max_bound()
    pcl_bb = o3d.geometry.PointCloud()
    pcl_bb.points = o3d.utility.Vector3dVector(np.array([bb_target_min, bb_target_max, bb_source_min, bb_source_max]))
    bounding_box = pcl_bb.get_axis_aligned_bounding_box()
    bounding_box_points = bounding_box.get_box_points()

    # add bounding_box points to both point clouds --> octree same cells
    bb = o3d.geometry.PointCloud()
    bb.points = o3d.utility.Vector3dVector(bounding_box_points)
    pcl_target += bb
    pcl_source += bb

    # Calculate octree depth
    max_distance = bounding_box.get_max_extent()
    depth = int(np.ceil(np.log2(max_distance / smax)))
    cfg.logging.info("Octree depth: " + str(depth))
    # with open(log_file, "a") as text_file:
    #     text_file.write("Octree depth: " + str(depth) + "\n")

    # define octree
    octree_target = o3d.geometry.Octree(max_depth=depth)
    octree_target.convert_from_point_cloud(pcl_target, size_expand=0.0)
    octree_source = o3d.geometry.Octree(max_depth=depth)
    octree_source.convert_from_point_cloud(pcl_source, size_expand=0.0)
    cfg.logging.info("Pointcloud converted to octree")

    # Get Centers of the octree cells
    global center_source
    global center_target
    center_source = []
    center_target = []
    source_bool = True
    octree_source.traverse(f_traverse)
    cfg.logging.info("Centroids found in source point cloud")

    source_bool = False
    octree_target.traverse(f_traverse)
    cfg.logging.info("Centroids found in target point cloud")

    # Define kdtree
    pcl_center_target = o3d.geometry.PointCloud()
    pcl_center_target.points = o3d.utility.Vector3dVector(np.asarray(center_target))
    points_center_source = np.asarray(center_source)
    center_tree = o3d.geometry.KDTreeFlann(pcl_center_target)
    cfg.logging.info("KDTree defined")

    # Find corresping centroids
    matching_centroids = []
    for i in np.arange(np.shape(points_center_source)[0]):
        [_, idx, _] = center_tree.search_knn_vector_3d(points_center_source[i, :], 1)
        matching_centroids.append(np.concatenate(
            (np.array([points_center_source[i, :]]), np.asarray(pcl_center_target.select_by_index(idx).points)),
            axis=1)[0, :])
    matching_centroids = np.asarray(matching_centroids)
    cfg.logging.info("Corresponding centroids found")

    # Distance between centroids
    dist_centroid = np.linalg.norm(matching_centroids[:, 0:3] - matching_centroids[:, 3:6], axis=1)

    dist_mean = np.mean(dist_centroid)
    dist_std = np.std(dist_centroid)
    dist_threshold = dist_mean + dist_std
    # dist_threshold = 0.5

    # find stable centroids
    stable_centroids = matching_centroids[np.where(dist_centroid <= dist_threshold)]
    unstable_centroids = matching_centroids[np.where(dist_centroid > dist_threshold)]

    cfg.logging.info("Centroid pairs are categorized as stable. (" + str(stable_centroids.shape[0]) + "/" + str(
        dist_centroid.size) + ", " + str(np.round(stable_centroids.shape[0] / dist_centroid.size * 100, 2)) + "%)")

    # extract points of stable octree cells
    pcl_stable_source = o3d.geometry.PointCloud()
    pcl_stable_target = o3d.geometry.PointCloud()
    stable_centroids_source = np.unique(stable_centroids[:, 0:3], axis=0)
    stable_centroids_target = np.unique(stable_centroids[:, 3:6], axis=0)
    for centroid in stable_centroids_source:
        point_idx_source = octree_source.locate_leaf_node(centroid)[0].indices
        pcl_stable_source += pcl_source.select_by_index(point_idx_source)

    piecewise_stable_dvfs = np.hstack((np.asarray(pcl_stable_source.points), np.asarray(pcl_stable_source.points)))
    piecewise_stable_mag = np.linalg.norm(piecewise_stable_dvfs[:, :3] - piecewise_stable_dvfs[:, 3:6], axis=1)
    piecewise_stable_dvfms = np.hstack((piecewise_stable_dvfs[:, :3], piecewise_stable_mag[:, None]))
    ####################
    # unstable part
    # pcl_unstable_source = o3d.geometry.PointCloud()
    pcl_unstable_target = o3d.geometry.PointCloud()
    # unstable_centroids_source = np.unique(unstable_centroids[:, 0:3], axis=0)
    # unstable_centroids_target = np.unique(unstable_centroids[:, 3:6], axis=0)

    unstable_centroids_source = unstable_centroids[:, 0:3]
    unstable_centroids_target = unstable_centroids[:, 3:6]
    unstable_dev = unstable_centroids_target - unstable_centroids_source
    pcl_unstable_source_target_all = []
    for i, centroid in enumerate(unstable_centroids_source):
        point_idx_source = octree_source.locate_leaf_node(centroid)[0].indices
        pcl_unstable_source = np.asarray(pcl_source.select_by_index(point_idx_source).points)
        pcl_unstable_target_from_source = pcl_unstable_source + unstable_dev[i, :]
        pcl_unstable_source_target = np.hstack((pcl_unstable_source, pcl_unstable_target_from_source))
        pcl_unstable_source_target_all.append(pcl_unstable_source_target)
        # pcl_unstable_source += pcl_source.select_by_index(point_idx_source)

    piecewise_unstable_dvfs = np.vstack(pcl_unstable_source_target_all)
    piecewise_unstable_mag = np.linalg.norm(piecewise_unstable_dvfs[:, :3] - piecewise_unstable_dvfs[:, 3:6], axis=1)
    piecewise_unstable_dvfms = np.hstack((piecewise_unstable_dvfs[:, :3], piecewise_unstable_mag[:, None]))

    piecewise_dvfms = np.vstack((piecewise_stable_dvfms, piecewise_unstable_dvfms))
    piecewise_dvfs = np.vstack((piecewise_stable_dvfs, piecewise_unstable_dvfs))
    np.savetxt(osp.join(results, f'piecewise_icp_dvfms_of_tile_{cfg.tile_id}.txt'), piecewise_dvfms)
    np.savetxt(osp.join(results, f'piecewise_icp_dvfs_of_tile_{cfg.tile_id}.txt'), piecewise_dvfs)

    piecewise_dvfms_visualize = copy.deepcopy(piecewise_dvfms)
    piecewise_dvfms_visualize[0, 3] = 0
    if cfg.dataset == 'rockfall':
        piecewise_dvfms_visualize[1, 3] = 0.06
    elif cfg.dataset == 'brienz_tls':
        piecewise_dvfms_visualize[1, 3] = 5
    elif cfg.dataset == 'mattertal':
        piecewise_dvfms_visualize[1, 3] = 10
    else:
        piecewise_dvfms_visualize[1, 3] = 10
    np.savetxt(osp.join(results, f'piecewise_dvfms_visualize_of_tile_{cfg.tile_id}.txt'), piecewise_dvfms_visualize)

    cfg.logging.info("Points in the source point cloud are categorized as stable. (" + str(
        np.asarray(pcl_stable_source.points).shape[0]) + "/" + str(
        np.asarray(pcl_source.points).shape[0]) + ", " + str(
        np.round(np.asarray(pcl_stable_source.points).shape[0] / np.asarray(pcl_source.points).shape[0] * 100,
                 2)) + "%)")

    for centroid in stable_centroids_target:
        point_idx_target = octree_target.locate_leaf_node(centroid)[0].indices
        pcl_stable_target += pcl_target.select_by_index(point_idx_target)

    for centroid in unstable_centroids_target:
        point_idx_target = octree_target.locate_leaf_node(centroid)[0].indices
        pcl_unstable_target += pcl_target.select_by_index(point_idx_target)
    cfg.logging.info("Points in the target point cloud are categorized as stable. (" + str(
        np.asarray(pcl_stable_target.points).shape[0]) + "/" + str(np.asarray(pcl_target.points).shape[0]) + ", " + str(
        np.round(np.asarray(pcl_stable_target.points).shape[0] / np.asarray(pcl_target.points).shape[0] * 100,
                 2)) + "%)")
    return None
