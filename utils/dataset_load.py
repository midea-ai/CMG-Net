import os.path
import sys
import numpy as np
import scipy
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PRO_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(PRO_DIR, 'utils'))

view_dataset_name = 'views7_dataset_info.pkl'

class graspTmpDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000, use_normal=False, use_height=False, augment=False,
                 data_root=None, num_pos_contacts=8000):
        assert (num_points <= 50000)
        self.split_set = split_set
        self.num_points = num_points
        self.use_color = use_normal
        self.use_height = use_height
        self.augment = augment
        self.data_root = data_root
        self.num_pos_contacts = num_pos_contacts

        self.point_cloud_list = []
        self.cam_poses = []
        self.finger_infos = []
        self.pc_seg_infos = []
        self.pc_seg_objs = []

        pickle_filename = os.path.join(ROOT_DIR, '../', self.data_root, view_dataset_name)
        with open(pickle_filename, 'rb') as f:
            self.data_infos = pickle.load(f)
        if split_set == 'test':
            self.data_infos = self.data_infos[:100]
        else:
            self.data_infos = self.data_infos[0:5]  # TODO xiugai
        pbar = tqdm(self.data_infos)
        for data_info_path in pbar:
            data_info_path = os.path.join(PRO_DIR, data_info_path)
            with open(data_info_path, 'rb') as f:
                data_info = pickle.load(f)
                pc_normal_file_name = os.path.split(data_info['pc_normal_file_name'])[-1]
                pc_seg_file_name = os.path.split(data_info['pc_seg_file_name'])[-1]
                points_dir = os.path.join(PRO_DIR, 'dataset', 'views_7', pc_normal_file_name)
                pc_seg_dir = os.path.join(PRO_DIR, 'dataset', 'views_7', pc_seg_file_name)
                self.point_cloud_list.append(np.load(points_dir))
                self.pc_seg_infos.append(np.load(pc_seg_dir))
                self.cam_poses.append(data_info['cam_pose'])
                self.finger_infos.append(data_info['finger_info'])
                self.pc_seg_objs.append(data_info['seg_index'])

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        point_cloud = self.point_cloud_list[idx]
        cam_pose = self.cam_poses[idx]
        finger_info = self.finger_infos[idx]
        pc_seg_info = self.pc_seg_infos[idx]
        pc_seg_obj = self.pc_seg_objs[idx]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # Loads fixed amount of contact grasp data per scene into CPU/GPU memory

        grasp_transforms = finger_info['grasp_transforms']
        scene_id = finger_info['scene_id']
        contact_points = finger_info['contact_points']
        contact_suc = np.ones_like(contact_points[:, 0])
        grasp_transformer = finger_info['grasp_transforms']
        contact_face_normals = finger_info['contact_face_normals']
        finger_idx = finger_info['finger_idxs']
        finger_vector = finger_info['finger_vectors']
        dof = finger_info['dofs']
        joint0s = finger_info['all_joints']
        # joint = finger_info['joints']
        joint0 = finger_info['all_joints'][:, 1]
        joint1 = finger_info['all_joints'][:, 4]
        joint2 = finger_info['all_joints'][:, 6]

        grasp_transforms_rot = grasp_transforms[:, 0:3, 0:3]
        rot = scipy.spatial.transform.Rotation.from_matrix(grasp_transforms_rot)
        grasp_transforms_eulor = rot.as_euler('zxy')
        grasp_transforms_tra = grasp_transforms[:, 0:3, 3]
        grasp_transforms_eulor_all = np.concatenate((grasp_transforms_eulor, grasp_transforms_tra), 1)

        pos_idcs = np.where(contact_suc > 0)[0]
        pos_contact_points = contact_points[pos_idcs]

        if self.num_pos_contacts > len(pos_contact_points):
            pos_sampled_contact_idcs = np.arange(len(pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(pos_contact_points)),
                                                                    self.num_pos_contacts - len(pos_contact_points),
                                                                    replace=True)
            pos_sampled_contact_idcs = np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(pos_contact_points)), self.num_pos_contacts,
                                                        replace=False)
        pos_contact_points = pos_contact_points[pos_sampled_contact_idcs, :]
        pos_contact_normal = contact_face_normals[pos_sampled_contact_idcs, :]
        pos_pc_seg_info = pc_seg_info
        dof = dof[pos_sampled_contact_idcs]
        joint0 = joint0[pos_sampled_contact_idcs]
        joint1 = joint1[pos_sampled_contact_idcs]
        joint2 = joint2[pos_sampled_contact_idcs]

        all_joints = joint0s[pos_sampled_contact_idcs, :]
        finger_idx = finger_idx[pos_sampled_contact_idcs]
        finger_vector = finger_vector[pos_sampled_contact_idcs, :]
        grasp_transforms = grasp_transforms[pos_sampled_contact_idcs, :]
        grasp_transforms_eulor_all = grasp_transforms_eulor_all[pos_sampled_contact_idcs, :]

        data_info = {}
        data_info['grasp_transforms'] = grasp_transforms
        data_info['grasp_transforms_eulor'] = grasp_transforms_eulor_all
        data_info['input_point_clouds'] = point_cloud
        data_info['cam_pose'] = cam_pose
        data_info['scene_idcs'] = finger_info['scene_id']
        data_info['pos_contact_points'] = pos_contact_points
        data_info['pos_contact_face_normals'] = pos_contact_normal
        data_info['finger_idxs'] = finger_idx
        data_info['finger_vector'] = finger_vector
        data_info['dof'] = dof
        data_info['joint0s'] = all_joints
        data_info['joint0'] = joint0
        data_info['joint1'] = joint1
        data_info['joint2'] = joint2
        data_info['pc_seg'] = pos_pc_seg_info
        if self.split_set == 'test':
            data_info['object_ids'] = finger_info['object_ids']
            data_info['object_transforms'] = finger_info['object_transforms']
            data_info['pc_seg_obj_idx'] = pc_seg_obj

        return data_info
