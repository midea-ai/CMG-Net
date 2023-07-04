# -*- coding: utf-8 -*-
"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import os
import numpy as np
import trimesh
import glob as glob
import math

from transformer import Transform

class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename, force='mesh')
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """Set longest of all three lengths in Cartesian space.

        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.

        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)

# np.array([1, 3])
def get_angle_between_two_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    cos_angle = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(cos_angle)
    return angle

class BarrettGripper(object):

    def __init__(self, root_folder=os.path.dirname(os.path.abspath(__file__))):

        # The forward kinematics equations implemented here are from https://support.barrett.com/wiki/Hand/280/KinematicsJointRangesConversionFactors
        self.Aw = 0.001 * 25
        self.A1 = 0.001 * 50
        self.A2 = 0.001 * 70
        self.A3 = 0.001 * 50
        self.Dw = 0.001 * 76
        self.Dw_knuckle = 0.001 * 42
        self.D3 = 0.001 * 9.5
        self.phi2 = 0
        self.phi3 = 0 * math.radians(42)
        self.pi_0_5 = math.pi / 2
        self.meshes = self.load_meshes()  # vertices, faces, vertice to face area, normals
        self.palm = self.meshes["palm_280"]
        self.knuckle = self.meshes["knuckle"]
        self.finger = self.meshes["finger"]
        self.finger_tip = self.meshes["finger_tip"]

        self.circle_center = Transform.from_list([0.0, 0.0, 0.0, 1.0, 0.0362, -0.0286, 0.0]).as_matrix()
        self.finger_circle_dis = np.linalg.norm(self.circle_center[0:3, 3])
        self.angle_finger_circle = get_angle_between_two_vectors(self.circle_center[0:3, 3].reshape(3), np.asarray([1, 0, 0]))

        # r and j are used to calculate the forward kinematics for the barrett Hand's different fingers
        self.r = [-1, 1, 0]
        self.j = [1, 1, -1]

    def load_meshes(self):
        mesh_path = "./gripper_models/barrett_hand/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            # print(name)
            triangle_areas = trimesh.triangles.area(mesh.triangles)
            vert_area_weight = []
            for i in range(mesh.vertices.shape[0]):
                vert_neighour_face = np.where(mesh.faces == i)[0]
                vert_area_weight.append(1000000*triangle_areas[vert_neighour_face].mean())
            temp = np.ones((mesh.vertices.shape[0], 1))
            meshes[name] = mesh
        # for i in meshes:
        #     meshes[i].show()
        return meshes

    def forward_kinematics(self, A, alpha, D, theta):
        c_theta = math.cos(theta)  # theta is radian
        s_theta = math.sin(theta)
        c_alpha = math.cos(alpha)
        s_alpha = math.sin(alpha)
        l_1_to_l = np.zeros((4, 4))
        l_1_to_l[0, 0] = c_theta
        l_1_to_l[0, 1] = -s_theta
        l_1_to_l[0, 3] = A
        l_1_to_l[1, 0] = s_theta * c_alpha
        l_1_to_l[1, 1] = c_theta * c_alpha
        l_1_to_l[1, 2] = -s_alpha
        l_1_to_l[1, 3] = -s_alpha * D
        l_1_to_l[2, 0] = s_theta * s_alpha
        l_1_to_l[2, 1] = c_theta * s_alpha
        l_1_to_l[2, 2] = c_alpha
        l_1_to_l[2, 3] = c_alpha * D
        l_1_to_l[3, 3] = 1
        return l_1_to_l

    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 7)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

            finger 1(-x), finger 2(+x), finger 3
       """
        pose = pose.as_matrix()
        self.palm.apply_transform(pose)
        self.knuckle_list = [self.knuckle.copy(), self.knuckle.copy(), 0]

        self.finger_list = [self.finger.copy(), self.finger.copy(), self.finger.copy()]
        self.finger_tip_list = [self.finger_tip.copy(), self.finger_tip.copy(), self.finger_tip.copy()]
        self.pose_to_T23_list = []
        self.pose_to_T12_list = []

        self.circle_center_list = [self.circle_center, self.circle_center, self.circle_center]
        hand = [self.palm]
        hand_expert_fingers = [self.palm]

        for i in range(3):
            Tw1 = self.forward_kinematics(
                self.r[i] * self.Aw,
                0,
                self.Dw,
                self.r[i] * theta[0] - (math.pi / 2) * self.j[i]
            )
            T12 = self.forward_kinematics(
                self.A1,
                math.pi / 2,
                0,
                self.phi2 + theta[i + 1]
            )
            T23 = self.forward_kinematics(
                self.A2,
                math.pi,
                0,
                self.phi3 - theta[i + 4]
            )
            if i is 0 or i is 1:
                Tw_knuckle = self.forward_kinematics(
                    self.r[(i+1) % 2] * self.Aw,
                    0,
                    self.Dw_knuckle,
                    -1*(self.r[i] * theta[0] - (math.pi / 2) * self.j[i])
                )
                pose_to_Tw_knuckle = np.dot(pose, Tw_knuckle)
                if i is 0:
                    self.knuckle_list[0].apply_transform(pose_to_Tw_knuckle)
                if i is 1:
                    self.knuckle_list[1].apply_transform(pose_to_Tw_knuckle)
            pose_to_T12 = np.dot(np.dot(pose, Tw1), T12)
            self.finger_list[i].apply_transform(pose_to_T12)
            self.pose_to_T12_list.append(pose_to_T12)
            pose_to_T23 = np.dot(pose_to_T12, T23)
            self.finger_tip_list[i].apply_transform(pose_to_T23)
            self.pose_to_T23_list.append(pose_to_T23)
            pose_to_circle_center = np.dot(pose_to_T23, self.circle_center)
            self.circle_center_list[i] = pose_to_circle_center

            if i is 0 or i is 1:
                fingers = trimesh.util.concatenate([self.knuckle_list[i], self.finger_list[i], self.finger_tip_list[i]])
                fingers_expert_tip = trimesh.util.concatenate([self.knuckle_list[i], self.finger_list[i]])
            else:
                fingers = trimesh.util.concatenate([self.finger_list[i], self.finger_tip_list[i]])
                fingers_expert_tip = self.finger_list[i]
            hand.append(fingers)
            hand_expert_fingers.append(fingers_expert_tip)

        self.hand = trimesh.util.concatenate(hand)
        self.hand_expert_fingers = trimesh.util.concatenate(hand_expert_fingers)

