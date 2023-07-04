import numpy as np
import torch.nn.functional as F
import math
import trimesh
import glob
import torch
import os
import sys

ROOT_DIR = (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from transformer import Transform, rotation_matrix, normal_vector


# All lengths are in mm and rotations in radians

# np.array([1, 3])
def get_angle_between_two_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    cos_angle = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(cos_angle)
    return angle


class BarrettLayer(torch.nn.Module):
    def __init__(self, device='cpu'):

        # The forward kinematics equations implemented here are from https://support.barrett.com/wiki/Hand/280/KinematicsJointRangesConversionFactors
        super().__init__()
        self.device = device
        self.Aw = torch.tensor(0.001 * 25, dtype=torch.float32, device=device)
        self.A1 = torch.tensor(0.001 * 50, dtype=torch.float32, device=device)
        self.A2 = torch.tensor(0.001 * 70, dtype=torch.float32, device=device)
        self.A3 = torch.tensor(0.001 * 50, dtype=torch.float32, device=device)
        self.Dw = torch.tensor(0.001 * 76, dtype=torch.float32, device=device)
        self.Dw_knuckle = torch.tensor(0.001 * 42, dtype=torch.float32, device=device)
        self.D3 = torch.tensor(0.001 * 9.5, dtype=torch.float32, device=device)
        self.phi2 = torch.tensor(0, dtype=torch.float32, device=device)
        self.phi3 = torch.tensor(0 * math.radians(42), dtype=torch.float32, device=device)
        self.pi_0_5 = torch.tensor(math.pi / 2, dtype=torch.float32, device=device)
        self.meshes = self.load_meshes()
        self.palm = self.meshes["palm_280"]
        self.knuckle = self.meshes["knuckle"]
        self.finger = self.meshes["finger"]
        self.finger_tip = self.meshes["finger_tip"]

        self.circle_center = Transform.from_list([0.0, 0.0, 0.0, 1.0, 0.0362, -0.0286, 0.0]).as_matrix()
        self.finger_circle_dis = torch.norm(torch.tensor(self.circle_center[0:3, 3]))
        self.angle_finger_circle = torch.tensor(
            get_angle_between_two_vectors(self.circle_center[0:3, 3].reshape(3), np.asarray([1, 0, 0])),
            dtype=torch.float32, device=self.device
        )

        # r and j are used to calculate the forward kinematics for the barrett Hand's different fingers
        self.r = [-1, 1, 0]
        self.j = [1, 1, -1]

    def load_meshes(self):
        mesh_path = os.path.join(ROOT_DIR, 'gripper_models/barrett_hand/*')
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            triangle_areas = trimesh.triangles.area(mesh.triangles)
            vert_area_weight = []
            for i in range(mesh.vertices.shape[0]):
                vert_neighour_face = np.where(mesh.faces == i)[0]
                vert_area_weight.append(1000000 * triangle_areas[vert_neighour_face].mean())
            temp = torch.ones(mesh.vertices.shape[0], 1).float()
            meshes[name] = [
                mesh
            ]
        return meshes

    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 7)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        batch_size = pose.shape[0]
        rot_z_90 = torch.eye(4, device=self.device)

        rot_z_90[1, 1] = -1
        rot_z_90[2, 3] = -0.001 * 79
        rot_z_90 = rot_z_90.repeat(batch_size, 1, 1)
        pose = torch.matmul(pose, rot_z_90)
        palm_vertices = self.palm.repeat(batch_size, 1, 1)
        palm_vertices = torch.matmul(pose,
                                     palm_vertices.transpose(2, 1)).transpose(
            1, 2)[:, :, :3]
        # The second dimension represents the number of joints for the fingers which are stored as: finger1 joint1, finger1 joint2, ..., finger 3 joint 2
        joints = torch.zeros((batch_size, 6, 4, 4), device=self.device)

        all_knuckle_vertices = torch.zeros(
            (batch_size, 2, self.knuckle.shape[0], 3), device=self.device)

        knuckle_vertices = self.knuckle.repeat(batch_size, 1, 1)

        all_finger_vertices = torch.zeros(
            (batch_size, 3, self.finger.shape[0], 3), device=self.device)
        all_finger_tip_vertices = torch.zeros(
            (batch_size, 3, self.finger_tip.shape[0], 3), device=self.device)

        finger_vertices = self.finger.repeat(batch_size, 1, 1)
        finger_tip_vertices = self.finger_tip.repeat(batch_size, 1, 1)
        for i in range(3):
            Tw1 = self.forward_kinematics(
                self.r[i] * self.Aw, torch.tensor(0, dtype=torch.float32, device=self.device), self.Dw,
                self.r[i] * theta[:, 0] - (math.pi / 2) * self.j[i],
                batch_size)
            T12 = self.forward_kinematics(self.A1, torch.tensor(math.pi / 2, dtype=torch.float32, device=self.device),
                                          0, self.phi2 + theta[:, i + 1],
                                          batch_size)
            T23 = self.forward_kinematics(self.A2, torch.tensor(math.pi, dtype=torch.float32, device=self.device), 0,
                                          self.phi3 - theta[:, i + 4], batch_size)

            if i == 0 or i == 1:
                Tw_knuckle = self.forward_kinematics(
                    self.r[(i + 1) % 2] * self.Aw, torch.tensor(0, dtype=torch.float32, device=self.device),
                    self.Dw_knuckle,
                    -1 * (self.r[i] * theta[:, 0] - (math.pi / 2) * self.j[i]),
                    batch_size)
                pose_to_Tw_knuckle = torch.matmul(pose, Tw_knuckle)
                all_knuckle_vertices[:, i] = torch.matmul(pose_to_Tw_knuckle,
                                                          knuckle_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
            pose_to_T12 = torch.matmul(torch.matmul(pose, Tw1), T12)
            all_finger_vertices[:, i] = torch.matmul(
                pose_to_T12,
                finger_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
            pose_to_T23 = torch.matmul(pose_to_T12, T23)
            all_finger_tip_vertices[:, i] = torch.matmul(
                pose_to_T23,
                finger_tip_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
            joints[:, 2 * i] = pose_to_T12
            joints[:, 2 * i + 1] = pose_to_T23
        return palm_vertices, all_knuckle_vertices, all_finger_vertices, all_finger_tip_vertices, joints

    def forward_kinematics(self, A, alpha, D, theta, batch_size, point_num):  # : * :
        '''
        :param A: tenser constant
        :param alpha:
        :param D:
        :param theta: B * N
        :param batch_size:
        :param point_num:
        :return: B * N * 4 * 4
        '''
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, point_num, 4, 4), device=self.device)
        l_1_to_l[:, :, 0, 0] = c_theta
        l_1_to_l[:, :, 0, 1] = -s_theta
        l_1_to_l[:, :, 0, 3] = A
        l_1_to_l[:, :, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, :, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, :, 1, 2] = -s_alpha
        l_1_to_l[:, :, 1, 3] = -s_alpha * D
        l_1_to_l[:, :, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, :, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, :, 2, 2] = c_alpha
        l_1_to_l[:, :, 2, 3] = c_alpha * D
        l_1_to_l[:, :, 3, 3] = 1
        return l_1_to_l

    def inverse(self, transform):
        rotation = torch.inverse(transform[:, :, 0:3, 0:3])
        translation = -transform[:, :, 0:3, 3:4]
        translation = torch.matmul(rotation, translation)
        add = torch.reshape(torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device), (1, 1, 1, 4)) \
            .repeat([transform.shape[0], transform.shape[1], 1, 1])
        T = torch.zeros((transform.shape[0], transform.shape[1], 4, 4))
        T[:, :, 0:3, 0:3] = rotation
        T[:, :, 0:3, 3:4] = translation
        T[:, :, 3:4, :] = add
        return T

    def XY_To_Pose(self, contact, contact_normal, x, y, dof, joint0, finger_idx, r=0.012):
        '''

        :param contact: B * N * 3
        :param contact_normal: B * N * 3
        :param x: B * N
        :param y: B * N
        :param dof: B * N
        :param joint0: B * N * 1
        :param finger_idx: int
        :param r: int
        :return:
        '''
        B, N, _ = contact.shape
        B_frame_z = F.normalize(contact_normal, p=2, dim=2)
        B_frame_y = F.normalize(torch.cat((
            torch.zeros_like(B_frame_z[:, :, 0:1]),
            -B_frame_z[:, :, 2:3],
            B_frame_z[:, :, 1:2]
        ), dim=2), p=2, dim=2)
        B_frame_x = F.normalize(
            torch.cross(B_frame_y, B_frame_z)
            , p=2, dim=2
        )
        A_point = contact
        B_point = A_point + B_frame_z * r
        T_w_B = torch.zeros((B_frame_z.shape[0], B_frame_z.shape[1], 4, 4))
        T_w_B[:, :, 0:3, 0:1] = torch.unsqueeze(B_frame_x, -1)
        T_w_B[:, :, 0:3, 1:2] = torch.unsqueeze(B_frame_y, -1)
        T_w_B[:, :, 0:3, 2:3] = torch.unsqueeze(B_frame_z, -1)
        T_w_B[:, :, 0:3, 3:4] = torch.unsqueeze(B_point, -1)
        add = torch.reshape(
            torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device), (1, 1, 1, 4)
        ).repeat([B_frame_z.shape[0], B_frame_z.shape[1], 1, 1])
        T_w_B[:, :, 3:4, :] = add

        test_x = x
        test_y = y
        x_2 = torch.pow(test_x, 2)
        y_2 = torch.pow(test_y, 2)
        z = torch.sqrt(1 - x_2 - y_2)
        filter_z = torch.zeros((B, N))
        pose_mask = torch.isnan(z)
        for i in range(B):
            for j in range(N):
                if pose_mask[i][j] == 0:
                    filter_z[i][j] = z[i][j]
                else:
                    filter_z[i][j] = 0
        z = filter_z.cuda()

        det_B = torch.reshape((
                torch.mul(torch.unsqueeze(test_x, -1), B_frame_x) +
                torch.mul(torch.unsqueeze(test_y, -1), B_frame_y) +
                torch.mul(torch.unsqueeze(z, -1), B_frame_z)
        ), (contact.shape[0], contact.shape[1], 3, 1)) * self.finger_circle_dis
        C = T_w_B[:, :, 0:3, 3:4].to(self.device) + det_B.to(self.device)

        CB = -det_B
        C1_frame_x = F.normalize(CB, p=2, dim=2)
        C1_frame_z = F.normalize(torch.cross(CB, torch.unsqueeze(B_frame_z, -1), dim=2), p=2, dim=2)
        C1_frame_y = F.normalize(torch.cross(C1_frame_z, C1_frame_x), p=2, dim=2)
        C1_point = C
        T_w_C1 = torch.zeros((B_frame_z.shape[0], B_frame_z.shape[1], 4, 4))
        T_w_C1[:, :, 0:3, 0:1] = C1_frame_x
        T_w_C1[:, :, 0:3, 1:2] = C1_frame_y
        T_w_C1[:, :, 0:3, 2:3] = C1_frame_z
        T_w_C1[:, :, 0:3, 3:4] = C1_point
        T_w_C1[:, :, 3:4, :] = add

        Rotation_C1_C = torch.from_numpy(rotation_matrix(np.asarray([0, 0, 1]).reshape(3), self.angle_finger_circle))
        T_C1_C = torch.eye(4, 4)
        T_C1_C[0:3, 0:3] = Rotation_C1_C

        T_w_C = torch.matmul(T_w_C1, T_C1_C)

        B, N = finger_idx.shape
        finger_r = torch.zeros((B, N))
        finger_j = torch.zeros((B, N))
        for i in range(B):
            for j in range(N):
                finger_r[i][j] = self.r[finger_idx[i][j]]
                finger_j[i][j] = self.j[finger_idx[i][j]]

        finger_r = finger_r.to(self.device)
        finger_j = finger_j.to(self.device)
        T_base_1 = self.forward_kinematics(
            # self.r[i] * self.Aw,
            finger_r * self.Aw,
            torch.tensor(0, dtype=torch.float32, device=self.device),
            self.Dw,
            # self.r[i] * joint0[:, :, 0] - (math.pi / 2) * self.j[i],
            finger_r * dof[:, :, 0] - (math.pi / 2) * finger_j,
            batch_size=contact.shape[0],
            point_num=contact.shape[1]
        )

        T_1_2 = self.forward_kinematics(
            self.A1,
            torch.tensor(math.pi / 2, dtype=torch.float32, device=self.device),
            0,
            self.phi2 + joint0[:, :, 0],
            batch_size=contact.shape[0],
            point_num=contact.shape[1]
        )
        T_2_3 = self.forward_kinematics(
            self.A2,
            torch.tensor(math.pi, dtype=torch.float32, device=self.device),
            0,
            self.phi3 - joint0[:, :, 0] / 3,
            batch_size=contact.shape[0],
            point_num=contact.shape[1]
        )

        T_base_3 = torch.matmul(T_base_1, torch.matmul(T_1_2, T_2_3))
        T_3_base = self.inverse(T_base_3)
        T_w_base = torch.matmul(T_w_C, T_3_base)
        use_pose = T_w_base.to(self.device)

        return use_pose

    def from_X_Y_to_base(self, torch_pos_contact_points, torch_pos_contact_face_normals,
                         torch_xs, torch_ys, torch_dofs, torch_joints, finger_idx, r=0.012):

        A_points = torch.squeeze(torch_pos_contact_points).numpy()
        A_B_list = torch.squeeze(torch_pos_contact_face_normals).numpy()
        B_x_list = torch.squeeze(torch_xs).numpy()
        B_y_list = torch.squeeze(torch_ys).numpy()
        tmp = list(torch.squeeze(torch_joints).numpy())
        joints = [tmp[0], tmp[1], tmp[4], tmp[6], tmp[2], tmp[5], tmp[7]]

        index = finger_idx
        from method.mesh_utils import BarrettGripper
        barrett = BarrettGripper()

        test_B_frame_z = normal_vector(A_B_list.reshape(1, 3))
        test_B_frame_y = normal_vector(np.asarray([0, -test_B_frame_z[0][2], test_B_frame_z[0][1]]).reshape(1, 3))
        test_B_frame_x = normal_vector(np.cross(test_B_frame_y, test_B_frame_z))
        assert round(np.linalg.norm(test_B_frame_z, axis=1)[0], 2) == 1
        B_point = A_points + test_B_frame_z * r
        # B_point = B_points[index].reshape(1, 3)
        test_B_x = B_x_list
        test_B_y = B_y_list
        test_B_z = np.sqrt(1 - np.linalg.norm(test_B_x) ** 2 - np.linalg.norm(test_B_y) ** 2)
        # det_B = (B_frame_x * B_x + B_frame_y * B_y + B_frame_z * B_z) * 0.01
        det_B = (
                        test_B_x * test_B_frame_x + test_B_y * test_B_frame_y + test_B_z * test_B_frame_z) * barrett.finger_circle_dis
        C = B_point + det_B

        CB = -det_B
        test_C_frame_x = normal_vector(CB)
        test_C_frame_z = normal_vector(np.cross(CB, test_B_frame_z))
        test_C_frame_y = normal_vector(np.cross(test_C_frame_z, test_C_frame_x))
        test_C_Transform = C
        test_C = np.concatenate((test_C_frame_x.T, test_C_frame_y.T, test_C_frame_z.T,
                                 test_C_Transform.reshape(3, 1)), axis=1)
        test_C_frame = np.concatenate((test_C, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)

        Rotation_C1_C = rotation_matrix(np.asarray([0, 0, 1]).reshape(3), barrett.angle_finger_circle)
        Transform_C1_C = np.concatenate((Rotation_C1_C, np.asarray([0, 0, 0]).reshape(3, 1)), axis=1)
        Transform_C1_C = np.concatenate((Transform_C1_C, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)

        C_frame = np.dot(test_C_frame, Transform_C1_C)
        Tw1 = barrett.forward_kinematics(
            barrett.r[index] * barrett.Aw,
            0,
            barrett.Dw,
            barrett.r[index] * joints[0] - (math.pi / 2) * barrett.j[index]
        )
        T12 = barrett.forward_kinematics(
            barrett.A1,
            math.pi / 2,
            0,
            barrett.phi2 + joints[index + 1]
        )
        T23 = barrett.forward_kinematics(
            barrett.A2,
            math.pi,
            0,
            barrett.phi3 - joints[index + 4]
        )
        T_base_3 = np.dot(Tw1, np.dot(T12, T23))
        base_3 = Transform.from_matrix(T_base_3)
        T_3_base = base_3.inverse().as_matrix()
        T_w_base = np.dot(C_frame, T_3_base)
        Tran_2_3 = Transform.from_matrix(T23)
        Tran_3_2 = Tran_2_3.inverse().as_matrix()
        T_w_base = torch.reshape(torch.from_numpy(T_w_base), (1, 1, 4, 4))
        return T_w_base
