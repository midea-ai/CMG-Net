import os
import sys
from PIL import Image
import numpy as np
import copy
import cv2
import trimesh.transformations as tra
from scipy.spatial import cKDTree
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import provider
import trimesh
from copy import deepcopy
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from object_utils import Objectness
from mesh_utils import Object
from transformer import Transform, Rotation
from pc_utils import farthest_points, distance_by_translation_point, regularize_pc_point_count, \
    estimate_normals_cam_from_pc


class PointCloudReader:
    """
    Class to load scenes, render point clouds and augment them during training

    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch

    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_samples {int} -- training scenes (default: {None})
        num_test_samples {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """

    def __init__(
            self,
            root_folder,
            obj_related_path,
            batch_size=3,
            raw_num_points=20000,
            estimate_normals=True,
            caching=True,
            scene_obj_paths=None,
            scene_obj_transforms=None,
            num_scene=None,
            use_farthest_point=False,
            intrinsics=None,
            distance_range=(0.9, 1.3),
            elevation=(30, 150),
            pc_augm_config=None,
            depth_augm_config=None
    ):
        self._root_folder = root_folder
        self._obj_related_path = obj_related_path
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._caching = caching
        self._num_scene = num_scene
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point
        self._scene_obj_scales = 1
        self._scene_obj_paths = scene_obj_paths
        self._scene_obj_transforms = scene_obj_transforms
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config

        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        self._cam_orientations = []
        self._elevation = np.array(elevation) / 180.
        for az in np.linspace(0, np.pi * 2, 30):
            for el in np.linspace(self._elevation[0], self._elevation[1], 10):
                self._cam_orientations.append(tra.euler_matrix(0, -el, az))
        self._coordinate_transform = tra.euler_matrix(np.pi / 2, 0, 0).dot(tra.euler_matrix(0, np.pi / 2, 0))

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """

        distance = self._distance_range[0] + np.random.rand() * (self._distance_range[1] - self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance
        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        cam_pose[2, 3] += self._renderer._table_dims[2] / 2
        cam_pose[:3, :2] = -cam_pose[:3, :2]
        return cam_pose

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud

        Returns:
            np.ndarray -- augmented point cloud
        """

        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'],
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :],
                                             sigma=self._pc_augm_config['sigma'],
                                             clip=self._pc_augm_config['clip'])[0]

        return pc[:, :3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config

        Arguments:
            depth {np.ndarray} -- depth map

        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma * np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth, (kernel, kernel), 0)
            depth[depth_copy == 0] = depth_copy[depth_copy == 0]

        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal

        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = farthest_points(pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def get_scene_batch(self, scene_idx=None, return_segmap=False, save=False):
        """
        Render a batch of scene point clouds

        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})

        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        dims = 6 if self._estimate_normals else 3
        batch_data = np.empty((self._batch_size, self._raw_num_points, dims), dtype=np.float32)
        cam_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)

        ps = []
        for p in self._scene_obj_paths[scene_idx]:
            names = p.split('#')
            related_name = names[0]
            ps.append(related_name)
        obj_paths = [
            os.path.join(self._root_folder, self._obj_related_path, p, 'google_16k/model_watertight_5000def.obj') for p
            in ps]
        mesh_scales = [1 for i in range(len(obj_paths))]
        obj_trafos = self._scene_obj_transforms[scene_idx]

        self.change_scene(obj_paths, mesh_scales, obj_trafos, visualize=False)

        batch_segmap, batch_obj_pcs = [], []
        for i in range(self._batch_size):
            # 0.005s
            pc_cam, pc_normals, camera_pose, depth = self.render_random_scene(estimate_normals=self._estimate_normals)

            if return_segmap:
                segmap, _, obj_pcs = self._renderer.render_labels(depth, obj_paths, mesh_scales, render_pc=True)
                batch_obj_pcs.append(obj_pcs)
                batch_segmap.append(segmap)

                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(depth)
                plt.show()
                plt.figure()
                plt.imshow(segmap)
                plt.show()

            batch_data[i, :, 0:3] = pc_cam[:, :3]
            if self._estimate_normals:
                batch_data[i, :, 3:6] = pc_normals[:, :3]
            cam_poses[i, :, :] = camera_pose

        if save:
            K = np.array([[616.36529541, 0, 310.25881958], [0, 616.20294189, 236.59980774], [0, 0, 1]])
            data = {'depth': depth, 'K': K, 'camera_pose': camera_pose, 'scene_idx': scene_idx}
            if return_segmap:
                data.update(segmap=segmap)
            np.savez('results/{}_acronym.npz'.format(scene_idx), data)
        # cam_poses = torch.FloatTensor(cam_poses)
        # batch_data = torch.FloatTensor(batch_data)

        if return_segmap:
            return batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs
        else:
            return batch_data, cam_poses, scene_idx

    def render_random_scene(self, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations

        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})

        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._cam_orientations))
            camera_orientation = self._cam_orientations[viewing_index]
            camera_pose = self.get_cam_pose(camera_orientation)

        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        _, depth, _, camera_pose = self._renderer.render(in_camera_pose, render_pc=False)
        depth = self._augment_depth(depth)

        pc = self._renderer._to_pointcloud(depth)
        pc = regularize_pc_point_count(pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc = self._augment_pc(pc)

        # view_pc = trimesh.points.PointCloud(pc)
        # view_pc.show()
        pc_normals = estimate_normals_cam_from_pc(pc[:, :3]) if estimate_normals else []

        return pc, pc_normals, camera_pose, depth

    def change_object(self, cad_path, cad_scale):
        """
        Change object in pyrender scene

        Arguments:
            cad_path {str} -- path to CAD model
            cad_scale {float} -- scale of CAD model
        """

        self._renderer.change_scene([cad_path], [cad_scale], [np.eye(4)])

    def change_scene(self, obj_paths, obj_scales, obj_transforms, visualize=False):
        """
        Change pyrender scene

        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models

        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        """
        self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        if visualize:
            self._visualizer.change_scene(obj_paths, obj_scales, obj_transforms)

    def __del__(self):
        print('********** terminating renderer **************')


class SceneRenderer:
    def __init__(self, intrinsics=None, fov=np.pi / 6, caching=True, viewing_mode=False):
        """Renders depth with given intrinsics during training.

        Keyword Arguments:
            intrinsics {str} -- camera name from 'kinect_azure', 'realsense' (default: {None})
            fov {float} -- field of view, ignored if inrinsics is not None (default: {np.pi/6})
            caching {bool} -- whether to cache object meshes (default: {True})
            viewing_mode {bool} -- visualize scene (default: {False})
        """

        self._fov = fov

        self._scene = pyrender.Scene()
        self._table_dims = [1.0, 1.2, 0.6]
        self._table_pose = np.eye(4)
        self._viewer = viewing_mode
        if viewing_mode:
            self._viewer = pyrender.Viewer(
                self._scene,
                use_raymond_lighting=True,
                run_in_thread=True)

        self._intrinsics = intrinsics
        if self._intrinsics == 'realsense':
            # 326.3168640136719, 242.61068725585938, 615.66357421875, 614.5852661132812
            self._fx = 615.66357421875
            self._fy = 614.5852661132812
            self._cx = 326.3168640136719
            self._cy = 242.61068725585938
            self._znear = 0.04
            self._zfar = 1000
            self._height = 480
            self._width = 640
        elif self._intrinsics == 'kinect_azure':
            self._fx = 631.54864502
            self._fy = 631.20751953
            self._cx = 638.43517329
            self._cy = 366.49904066
            self._znear = 0.04
            self._zfar = 20
            self._height = 720
            self._width = 1280

        self._add_table_node()
        self._init_camera_renderer()

        self._current_context = None
        self._cache = {} if caching else None
        self._caching = caching

    def _init_camera_renderer(self):
        """
        If not in visualizing mode, initialize camera with given intrinsics
        """

        if self._viewer:
            return

        if self._intrinsics in ['kinect_azure', 'realsense']:
            camera = pyrender.IntrinsicsCamera(self._fx, self._fy, self._cx, self._cy, self._znear, self._zfar)
            self._camera_node = self._scene.add(camera, pose=np.eye(4), name='camera')
            self.renderer = pyrender.OffscreenRenderer(viewport_width=self._width,
                                                       viewport_height=self._height,
                                                       point_size=1.0)
        else:
            camera = pyrender.PerspectiveCamera(yfov=self._fov, aspectRatio=1.0,
                                                znear=0.001)  # do not change aspect ratio FOV就是相机视锥体的两端的夹角
            self._camera_node = self._scene.add(camera, pose=tra.euler_matrix(np.pi, 0, 0), name='camera')
            self.renderer = pyrender.OffscreenRenderer(400, 400)

    def _add_table_node(self):
        """
        Adds table mesh and sets pose
        """
        if self._viewer:
            return
        obj_path = '/home/test/program_grasp/grasp_tmp/dataset/urdfs/setup/3d-model.obj'
        obj = Object(obj_path)
        obj.rescale(0.03)
        tmesh = obj.mesh

        for facet in tmesh.facets:
            tmesh.visual.face_colors[facet] = np.asarray([255, 255, 204, 255])
            # tmesh.visual.face_colors[facet] = trimesh.visual.random_color()
        mesh = pyrender.Mesh.from_trimesh(tmesh, smooth=False)

        T_x = Transform.from_matrix(
            np.asarray([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
        )
        pose = (Transform(Rotation.identity(), [0, 0, -0.704]) * T_x).as_matrix()

        # table_mesh = trimesh.creation.box(self._table_dims)
        # # trimesh.visual.color.ColorVisuals(table_mesh, )
        # # for facet in table_mesh.facets:
        # #     table_mesh.visual.face_colors[facet] = [255, 0, 0, 255]
        # mesh = pyrender.Mesh.from_trimesh(table_mesh)
        # pose = Transform(Rotation.identity(), [0, 0, 1])

        table_node = pyrender.Node(mesh=mesh, name='table')
        self._scene.add_node(table_node)
        self._scene.set_pose(table_node, pose)

    def _load_object(self, path, scale):
        """
        Load a mesh, scale and center it

        Arguments:
            path {str} -- path to mesh
            scale {float} -- scale of the mesh

        Returns:
            dict -- contex with loaded mesh info
        """
        if (path, scale) in self._cache:
            return self._cache[(path, scale)]
        obj = Objectness(path)
        obj.rescale(scale)

        tmesh = obj.mesh
        tmesh_mean = np.mean(tmesh.vertices, 0)
        tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        lbs = np.min(tmesh.vertices, 0)
        ubs = np.max(tmesh.vertices, 0)
        object_distance = np.max(ubs - lbs) * 5

        mesh = pyrender.Mesh.from_trimesh(tmesh)

        context = {
            'name': path + '_' + str(scale),
            'tmesh': copy.deepcopy(tmesh),
            'distance': object_distance,
            'node': pyrender.Node(mesh=mesh, name=path + '_' + str(scale)),
            'mesh_mean': np.expand_dims(tmesh_mean, 0),
        }

        self._cache[(path, scale)] = context

        return self._cache[(path, scale)]

    def change_scene(self, obj_paths, obj_scales, obj_transforms):
        """Remove current objects and add new ones to the scene

        Arguments:
            obj_paths {list} -- list of object mesh paths
            obj_scales {list} -- list of object scales
            obj_transforms {list} -- list of object transforms
        """
        if self._viewer:
            self._viewer.render_lock.acquire()
        for n in self._scene.get_nodes():
            if n.name not in ['table', 'camera', 'parent']:
                self._scene.remove_node(n)

        if not self._caching:
            self._cache = {}

        for p, t, s in zip(obj_paths, obj_transforms, obj_scales):
            object_context = self._load_object(p, s)
            object_context = deepcopy(object_context)

            self._scene.add_node(object_context['node'])
            self._scene.set_pose(object_context['node'], t)

        if self._viewer:
            self._viewer.render_lock.release()

    def _to_pointcloud(self, depth):
        """Convert depth map to point cloud

        Arguments:
            depth {np.ndarray} -- HxW depth map

        Returns:
            np.ndarray -- Nx4 homog. point cloud
        """
        fy = self._fy
        fx = self._fx
        height = self._height
        width = self._width
        cx = self._cx
        cy = self._cy

        mask = np.where(depth > 0.2)

        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - cx)
        normalized_y = (y.astype(np.float32) - cy)

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

    def render(self, pose, render_pc=True):
        """Render object or scene in camera pose

        Arguments:
            pose {np.ndarray} -- 4x4 camera pose

        Keyword Arguments:
            render_pc {bool} -- whether to convert depth map to point cloud (default: {True})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- HxWx3 color, HxW depth, Nx4 point cloud, 4x4 camera pose
        """

        transferred_pose = pose.copy()
        self._scene.set_pose(self._camera_node, transferred_pose)

        self.add_light(transferred_pose)

        color, depth = self.renderer.render(self._scene)

        self._scene.remove_node(self.direc_l_node)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, transferred_pose

    def render_labels(self, full_depth, obj_paths, obj_scales, render_pc=False):
        """Render instance segmentation map

        Arguments:
            full_depth {np.ndarray} -- HxW depth map
            obj_paths {list} -- list of object paths in scene
            obj_scales {list} -- list of object scales in scene

        Keyword Arguments:
            render_pc {bool} -- whether to return object-wise point clouds (default: {False})

        Returns:
            [np.ndarray, list, dict] -- integer segmap with 0=background, list of
                                        corresponding object names, dict of corresponding point clouds
        """

        scene_object_nodes = []
        for n in self._scene.get_nodes():
            if n.name not in ['camera', 'parent']:
                n.mesh.is_visible = False
                if n.name != 'table':
                    scene_object_nodes.append(n)

        obj_names = [path + '_' + str(scale) for path, scale in zip(obj_paths, obj_scales)]

        pcs = {}
        output = np.zeros(full_depth.shape, np.uint8)
        for n in scene_object_nodes:
            n.mesh.is_visible = True

            depth = self.renderer.render(self._scene)[1]
            mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0
            )
            if not np.any(mask):
                continue
            if np.any(output[mask] != 0):
                raise ValueError('wrong label')

            indices = [i + 1 for i, x in enumerate(obj_names) if x == n.name]
            for i in indices:
                if not np.any(output == i):
                    print('')
                    output[mask] = i
                    break

            n.mesh.is_visible = False

            if render_pc:
                pcs[i] = self._to_pointcloud(depth * mask)

        for n in self._scene.get_nodes():
            if n.name not in ['camera', 'parent']:
                n.mesh.is_visible = True

        return output, ['BACKGROUND'] + obj_names, pcs

    def add_light(self, light_pose):
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        self.direc_l_node = self._scene.add(direc_l, pose=light_pose)
