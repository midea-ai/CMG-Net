import numpy as np
import copy
import cv2
import trimesh.transformations as tra
from scipy.spatial import cKDTree

def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False,
                                convert_to_internal_coords=False):
    """
    Various preprocessing of the point cloud (downsampling, centering, coordinate transforms)

    Arguments:
        input_pc {np.ndarray} -- Nx3 input point cloud
        num_point {int} -- downsample to this amount of points

    Keyword Arguments:
        pc_mean {np.ndarray} -- use 3x1 pre-computed mean of point cloud  (default: {None})
        return_mean {bool} -- whether to return the point cloud mean (default: {False})
        use_farthest_point {bool} -- use farthest point for downsampling (slow and suspectible to outliers) (default: {False})
        convert_to_internal_coords {bool} -- Convert from opencv to internal coordinates (x left, y up, z front) (default: {False})

    Returns:
        [np.ndarray] -- num_pointx3 preprocessed point cloud
    """
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()

    if convert_to_internal_coords:
        pc[:, :2] *= -1

    if pc_mean is None:
        pc_mean = np.mean(pc, 0)

    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    else:
        return pc


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.

    Arguments:
        trans {np.ndarray} -- 4x4 transform.

    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def reject_median_outliers(data, m=0.4, z_only=False):
    """
    Reject outliers with median absolute distance m

    Arguments:
        data {[np.ndarray]} -- Numpy array such as point cloud

    Keyword Arguments:
        m {[float]} -- Maximum absolute distance from median in m (default: {0.4})
        z_only {[bool]} -- filter only via z_component (default: {False})

    Returns:
        [np.ndarray] -- Filtered data without outliers
    """
    if z_only:
        d = np.abs(data[:, 2:3] - np.median(data[:, 2:3]))
    else:
        d = np.abs(data - np.median(data, axis=0, keepdims=True))

    return data[np.sum(d, axis=1) < m]


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates

      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """

    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0, 2])
    normalized_y = (y.astype(np.float32) - K[1, 2])

    world_x = normalized_x * depth[y, x] / K[0, 0]
    world_y = normalized_y * depth[y, x] / K[1, 1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y, x, :]

    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)


def estimate_normals_cam_from_pc(pc_cam, max_radius=0.05, k=12):
    """
    Estimates normals in camera coords from given point cloud.

    Arguments:
        pc_cam {np.ndarray} -- Nx3 point cloud in camera coordinates

    Keyword Arguments:
        max_radius {float} -- maximum radius for normal computation (default: {0.05})
        k {int} -- Number of neighbors for normal computation (default: {12})

    Returns:
        [np.ndarray] -- Nx3 point cloud normals
    """
    tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0] + 1)
    _, ndx = tree.query(pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=-1)  # num_points x k

    for c, idcs in enumerate(ndx):
        idcs[idcs == pc_cam.shape[0]] = c
        ndx[c, :] = idcs
    neighbors = np.array([pc_cam[ndx[:, n], :] for n in range(k)]).transpose((1, 0, 2))
    pc_normals = vectorized_normal_computation(pc_cam, neighbors)
    return pc_normals


def vectorized_normal_computation(pc, neighbors):
    """
    Vectorized normal computation with numpy

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        neighbors {np.ndarray} -- Nxkx3 neigbours

    Returns:
        [np.ndarray] -- Nx3 normal directions
    """
    diffs = neighbors - np.expand_dims(pc, 1)  # num_point x k x 3
    covs = np.matmul(np.transpose(diffs, (0, 2, 1)), diffs)  # num_point x 3 x 3
    covs /= diffs.shape[1] ** 2
    # takes most time: 6-7ms
    eigen_values, eigen_vectors = np.linalg.eig(covs)  # num_point x 3, num_point x 3 x 3
    orders = np.argsort(-eigen_values, axis=1)  # num_point x 3
    orders_third = orders[:, 2]  # num_point
    directions = eigen_vectors[np.arange(pc.shape[0]), :, orders_third]  # num_point x 3
    dots = np.sum(directions * pc, axis=1)  # num_point
    directions[dots >= 0] = -directions[dots >= 0]
    return directions


def center_pc_convert_cam(cam_poses, batch_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud

    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds
    :returns: (cam_poses, batch_data) converted
    """
    # OpenCV OpenGL conversion
    for j in range(len(cam_poses)):
        cam_poses[j, :3, 1] = -cam_poses[j, :3, 1]
        cam_poses[j, :3, 2] = -cam_poses[j, :3, 2]
        cam_poses[j] = inverse_transform(cam_poses[j])

    pc_mean = np.mean(batch_data, axis=1, keepdims=True)
    batch_data[:, :, :3] -= pc_mean[:, :, :3]
    cam_poses[:, :3, 3] -= pc_mean[:, 0, :3]

    return cam_poses, batch_data
