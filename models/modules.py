import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


class grasp_pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(256 + 3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 2 + 3 + 2, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, seed_xyz, seed_feature, end_points):
        B, num_seed, _ = seed_xyz.size()
        feature = torch.cat((seed_xyz.transpose(2, 1).contiguous(), seed_feature), 1)
        feature = F.relu(self.bn1(self.conv1(feature)), inplace=True)
        feature = F.relu(self.bn2(self.conv2(feature)), inplace=True)
        feature = self.conv3(feature)
        sem_score = feature[:, :2, :]
        finger_score = feature[:, 2:5, :]
        xs = feature[:, 5, :]
        ys = feature[:, 6, :]
        end_points['pre_xs'] = xs
        end_points['pre_ys'] = ys
        end_points['pre_sem_score'] = sem_score
        end_points['pre_finger_score'] = finger_score

        return end_points


class grasp_pre_eulor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(256 + 3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 8, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, seed_xyz, seed_feature, end_points):
        B, num_seed, _ = seed_xyz.size()
        feature = torch.cat((seed_xyz.transpose(2, 1).contiguous(), seed_feature), 1)
        feature = F.relu(self.bn1(self.conv1(feature)), inplace=True)
        feature = F.relu(self.bn2(self.conv2(feature)), inplace=True)
        feature = self.conv3(feature)
        sem_score = feature[:, :2, :]
        # finger_score = feature[:, 2:5, :]
        # xs = feature[:, 5, :]
        # ys = feature[:, 6, :]
        # end_points['pre_xs'] = xs
        # end_points['pre_ys'] = ys
        end_points['pre_sem_score'] = sem_score
        # end_points['pre_finger_score'] = finger_score
        end_points['pred_eulor'] = feature[:, 2:, :]

        return end_points


class pre_joint(nn.Module):
    def __init__(self, num_joint, joint_name):
        super().__init__()
        self.num_joint = num_joint
        self.joint_name = joint_name
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_joint * 2, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, seed_xyz, seed_feature, end_points):
        B, num_seed, _ = seed_xyz.size()
        feature = F.relu(self.bn1(self.conv1(seed_feature)), inplace=True)
        feature = F.relu(self.bn2(self.conv2(feature)), inplace=True)
        feature = self.conv3(feature)

        end_points['pre_joint_' + self.joint_name + '_cls'] = feature[:, :self.num_joint, ]
        end_points['pre_joint_' + self.joint_name + '_res'] = feature[:, self.num_joint:, :]

        return end_points
