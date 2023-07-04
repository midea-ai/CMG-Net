import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from backbone_old import Pointnet2Backbone
# from backbone import Pointnet2Backbone

from modules import grasp_pre_eulor, pre_joint


class Grasp_baseline(nn.Module):
    def __init__(self, input_feature_dim=3, dof_num=10, training=True):
        super(Grasp_baseline, self).__init__()
        self.is_training = training
        self.backbone = Pointnet2Backbone(3)
        self.pose_module = grasp_pre_eulor()
        self.pre_dof = pre_joint(6, 'dof')
        self.pre_joint0 = pre_joint(7, '0')
        self.pre_joint1 = pre_joint(7, '1')
        self.pre_joint2 = pre_joint(7, '2')


    def forward(self, pc):
        end_points = {}
        # xyz = pc.transpose(1, 2).contiguous()
        seed_features, seed_xyz, end_points = self.backbone(pc, end_points)
        end_points = self.pose_module(seed_xyz, seed_features, end_points)
        end_points = self.pre_dof(seed_xyz, seed_features, end_points)
        end_points = self.pre_joint0(seed_xyz, seed_features, end_points)
        end_points = self.pre_joint1(seed_xyz, seed_features, end_points)
        end_points = self.pre_joint2(seed_xyz, seed_features, end_points)
        return end_points
