import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import scipy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from barrett_layer import BarrettLayer
from nn_distance import huber_loss, nn_distance

GP_weight = 1
NUM_JOINT_CLS = 7
NUM_DOF_CLS = 6
epsilon = 1e-7

def get_loss_new(end_points, args):
    gp_mask = torch.where(end_points['label_grasp_success_pc'].unsqueeze(-1) > 0)
    cls_loss, end_points = compute_objectness_loss(end_points, args)
    dof_loss, end_points = compute_degree_loss(end_points['pre_joint_dof_cls'], end_points['pre_joint_dof_res'],
                                               end_points['label_dofs'], 'dof', end_points)
    joint0_loss, end_points = compute_degree_loss(end_points['pre_joint_0_cls'], end_points['pre_joint_0_res'],
                                                  end_points['label_joint0'], '0', end_points)
    joint1_loss, end_points = compute_degree_loss(end_points['pre_joint_1_cls'], end_points['pre_joint_1_res'],
                                                  end_points['label_joint1'], '1', end_points)
    joint2_loss, end_points = compute_degree_loss(end_points['pre_joint_2_cls'], end_points['pre_joint_2_res'],
                                                  end_points['label_joint2'], '2', end_points)
    if args.fully_pose_loss:
        pose_loss, end_points = fully_pose_loss(end_points)
    else:
        pose_loss, end_points = compute_pose_simple(end_points, args)
    # pose_loss, end_points = com
    total_loss = args.cls_weight * cls_loss + args.dof_weight * dof_loss + args.joint_weight * joint0_loss + \
                 args.joint_weight * joint1_loss + args.joint_weight * joint2_loss + args.pose_weight * pose_loss
    end_points['total_loss'] = total_loss
    return total_loss, end_points


def get_loss_eulor(end_points, args):
    gp_mask = torch.where(end_points['label_grasp_success_pc'].unsqueeze(-1) > 0)
    cls_loss, end_points = compute_objectness_loss(end_points, args)
    dof_loss, end_points = compute_degree_loss(end_points['pre_joint_dof_cls'], end_points['pre_joint_dof_res'],
                                               end_points['label_dofs'], 'dof', end_points)
    joint0_loss, end_points = compute_degree_loss(end_points['pre_joint_0_cls'], end_points['pre_joint_0_res'],
                                                  end_points['label_joint0'], '0', end_points)
    joint1_loss, end_points = compute_degree_loss(end_points['pre_joint_1_cls'], end_points['pre_joint_1_res'],
                                                  end_points['label_joint1'], '1', end_points)
    joint2_loss, end_points = compute_degree_loss(end_points['pre_joint_2_cls'], end_points['pre_joint_2_res'],
                                                  end_points['label_joint2'], '2', end_points)
    # if args.fully_pose_loss:
    #     pose_loss, end_points = fully_pose_loss(end_points)
    # else:
    #     pose_loss, end_points = compute_pose_simple(end_points, args)
    eulor_loss, end_points = compute_eulor_loss(end_points, args)
    # pose_loss, end_points = com
    total_loss = args.cls_weight * cls_loss + args.dof_weight * dof_loss + args.joint_weight * joint0_loss + \
                 args.joint_weight * joint1_loss + args.joint_weight * joint2_loss + args.pose_weight * eulor_loss
    end_points['total_loss'] = total_loss
    return total_loss, end_points


def compute_objectness_loss(end_points, args):
    pred_grasp_score = end_points['pre_sem_score']
    label_if_grasp = end_points['label_grasp_success_pc']
    fp3_inds = end_points['sa2_inds'].long().cuda()
    B, N = fp3_inds.size()

    # if_gp_mask = torch.where(label_if_grasp > -1)
    # gp_mask = torch.where(label_if_grasp > 0)
    gp_loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    gp_loss = gp_loss_criterion(pred_grasp_score, label_if_grasp.long())
    end_points['objectness_loss'] = gp_loss

    objectness_pred = torch.argmax(pred_grasp_score, 1)
    tp, tn, fp, fn = calculate_classification_statistics(objectness_pred, label_if_grasp)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    end_points['grasp_positive_pred_acc'] = (objectness_pred == label_if_grasp.long()).float().mean()

    end_points['grasp_positive_pred_prec'] = precision
    end_points['grasp_positive_pred_recall'] = recall

    find_positive_pc = torch.sum(objectness_pred)
    actually_positive_pc = torch.sum(label_if_grasp)
    if 'pre_finger_score' in end_points:
        # finger index pred loss
        finger_criterion = nn.CrossEntropyLoss(reduction='mean')
        pre_finger_score = end_points['pre_finger_score']
        label_finger_score = end_points['grouped_finger_idx'].cuda()
        finger_index_loss = args.finger_cls_weight * finger_criterion(pre_finger_score, label_finger_score.long())
        end_points['finger_index_loss'] = finger_index_loss

        finger_index_pred = torch.argmax(pre_finger_score, 1)
        end_points['finger_index_pred_acc'] = (finger_index_pred == label_finger_score.long()).float().mean()

        cls_loss = gp_loss + finger_index_loss

        return cls_loss, end_points
    cls_loss = gp_loss

    return cls_loss, end_points


def compute_degree_loss(pre_cls, pre_res, label_rad, name, end_points):
    if name == 'dof':
        NUM_JOINT_CLS = 6
        tmp_scope = 90 / NUM_JOINT_CLS
    else:
        NUM_JOINT_CLS = 7
        tmp_scope = 140 / NUM_JOINT_CLS
    label_grasp = end_points['label_grasp_success_pc']
    B, _, N = end_points['pre_sem_score'].shape

    label_joint_deg = torch.rad2deg_(label_rad)
    label_joint_deg = torch.clamp(label_joint_deg, 0, tmp_scope * NUM_JOINT_CLS - 1e-4)
    label_joint_cls = (label_joint_deg / tmp_scope).floor().long()
    label_joint_res = label_joint_deg - (label_joint_cls.float() * tmp_scope)
    label_joint_normalized_res = label_joint_res / tmp_scope

    criterion_joint_cls = nn.CrossEntropyLoss(reduction='none')
    # joint_cls_loss = criterion_joint_cls(
    #     (pre_cls.transpose(2, 1).contiguous()[gp_mask]),
    #     (label_joint_cls.unsqueeze(-1)[gp_mask]).long())
    joint_cls_loss = criterion_joint_cls(pre_cls, label_joint_cls)
    joint_cls_loss = torch.sum(joint_cls_loss * label_grasp) / (torch.sum(label_grasp) + 1e-6)

    joint_cls_one_hot = torch.cuda.FloatTensor(B, N, NUM_JOINT_CLS).zero_()
    joint_cls_one_hot.scatter_(2, label_joint_cls.unsqueeze(-1), 1)

    joint_residual_normalized_loss = huber_loss(
        torch.sum(pre_res.transpose(2, 1) * joint_cls_one_hot, -1) - label_joint_normalized_res,
        delta=1.0)  # (B,K)
    joint_residual_normalized_loss = torch.sum(joint_residual_normalized_loss * label_grasp) / (
            torch.sum(label_grasp) + 1e-6)
    joint_loss = joint_cls_loss + 10 * joint_residual_normalized_loss

    end_points['joint_' + name + '_cls_loss'] = joint_cls_loss
    end_points['joint_' + name + '_res_loss'] = 10 * joint_residual_normalized_loss
    end_points['joint_' + name + '_loss'] = joint_loss

    return joint_loss.float(), end_points


def fully_pose_loss(end_points):
    label_grasp = end_points['label_grasp_success_pc']
    pred_pose, label_pose, _ = compute_pose(end_points, 1)
    pred_pose = pred_pose[:, :, 0:3, :].reshape(pred_pose.shape[0], pred_pose.shape[1], -1)
    label_pose = label_pose[:, :, 0:3, :].reshape(pred_pose.shape[0], pred_pose.shape[1], -1)
    pose_loss = torch.mean(huber_loss(pred_pose - label_pose, delta=1.0), -1)
    pose_loss = 10 * torch.sum(pose_loss * label_grasp) / (torch.sum(label_grasp) + 1e-6)
    end_points['pose_loss'] = pose_loss
    return pose_loss, end_points


def compute_pose(end_points, cam_poses=None):
    # if ['label_x'] in end_points:
    dof_scope = 15
    joint_scope = 20
    B, N = end_points['pre_xs'].shape
    if 'fp3_xyz' in end_points:
        contact_points = end_points['sa1_xyz']
        contact_points_normals = end_points['sa1_normal']
    else:
        contact_points = end_points['sa2_xyz']
        contact_points_normals = end_points['fp2_normals']

    pre_x = end_points['pre_xs']
    pre_y = end_points['pre_ys']
    pre_dofs = pre_joint(end_points['pre_joint_dof_cls'], end_points['pre_joint_dof_res'], dof_scope, dof_scope)
    pre_joint_0s = pre_joint(end_points['pre_joint_0_cls'], end_points['pre_joint_0_res'], joint_scope, joint_scope)
    pre_joint_1s = pre_joint(end_points['pre_joint_1_cls'], end_points['pre_joint_1_res'], joint_scope, joint_scope)
    pre_joint_2s = pre_joint(end_points['pre_joint_2_cls'], end_points['pre_joint_2_res'], joint_scope, joint_scope)
    pred_finger_idx_score = end_points['pre_finger_score']
    pre_finger_idx = torch.argmax(pred_finger_idx_score, 1)
    end_points['pre_finger_idx'] = pre_finger_idx

    pre_dofs_clone = pre_dofs.clone()
    pre_joint_0s_clone = pre_joint_0s.clone()
    pre_joint_1s_clone = pre_joint_1s.clone()
    pre_joint_2s_clone = pre_joint_2s.clone()

    # pred contact point joint
    pred_pose_joint = pre_joint_0s
    tmp_joint1s_idx = torch.nonzero(pre_finger_idx == 1, as_tuple=False)[:, 1]
    pred_pose_joint[:, tmp_joint1s_idx] = pre_joint_1s[:, tmp_joint1s_idx]

    tmp_joint2s_idx = torch.nonzero(pre_finger_idx == 2, as_tuple=False)[:, 1]
    pred_pose_joint[:, tmp_joint2s_idx] = pre_joint_2s[:, tmp_joint2s_idx]

    pred_pose_joint = torch.deg2rad(pred_pose_joint)

    barret_layer = BarrettLayer(contact_points.device)

    pred_pose = barret_layer.XY_To_Pose(contact_points, contact_points_normals, pre_x, pre_y,
                                        torch.deg2rad(pre_dofs).unsqueeze(-1).cuda(),
                                        pred_pose_joint.unsqueeze(-1).cuda(), pre_finger_idx)

    end_points['pred_pose'] = pred_pose
    end_points['pre_dofs'] = torch.deg2rad(pre_dofs_clone)
    end_points['pre_joint0s'] = torch.deg2rad(pre_joint_0s_clone + 5)
    end_points['pre_joint1s'] = torch.deg2rad(pre_joint_1s_clone + 5)
    end_points['pre_joint2s'] = torch.deg2rad(pre_joint_2s_clone + 5)

    # label_pose = inverse_transform(cam_poses)
    # label_pose = torch.matmul(torch.from_numpy(inverse_transform(cam_poses.cpu().numpy())), torch.from_numpy(label_pose))
    if 'label_x' in end_points:
        label_x = end_points['label_x']
        label_y = end_points['label_y']
        label_dofs = torch.deg2rad(end_points['label_dofs'])
        label_joint_0s = torch.deg2rad(end_points['label_joint0'])
        label_joint_1s = torch.deg2rad(end_points['label_joint1'])
        label_joint_2s = torch.deg2rad(end_points['label_joint2'])
        label_finger_idx = end_points['grouped_finger_idx']
        # label contact point joint

        label_pose_joint = copy.deepcopy(label_joint_0s)
        # label_pose_joint = label_joint_0s
        tmp_label_joint1s_idx = torch.nonzero(label_finger_idx == 1, as_tuple=False)[:, 1]
        label_pose_joint[:, tmp_label_joint1s_idx] = label_joint_1s[:, tmp_label_joint1s_idx]

        tmp_label_joint2s_idx = torch.nonzero(label_finger_idx == 2, as_tuple=False)[:, 1]
        label_pose_joint[:, tmp_label_joint2s_idx] = label_joint_2s[:, tmp_label_joint2s_idx]
        label_pose = barret_layer.XY_To_Pose(contact_points, contact_points_normals, label_x.cuda(), label_y.cuda(),
                                             label_dofs.unsqueeze(-1).cuda(), label_pose_joint.unsqueeze(-1).cuda(),
                                             label_finger_idx)
        label_pose_test = end_points['grasp_transforms']

        end_points['label_pose'] = label_pose
        end_points['label_dof_rad'] = label_dofs
        end_points['label_joint0s'] = label_joint_0s
        end_points['label_joint1s'] = label_joint_1s
        end_points['label_joint2s'] = label_joint_2s
        return pred_pose, label_pose, label_pose_test
    return pred_pose, pre_dofs, pre_joint_0s, pre_joint_1s, pre_joint_2s


def compute_pose_eulor(end_points, cam_poses=None):
    # if ['label_x'] in end_points:
    dof_scope = 15
    joint_scope = 20
    # B, N = end_points['pre_xs'].shape
    if 'fp3_xyz' in end_points:
        contact_points = end_points['sa1_xyz']
        contact_points_normals = end_points['sa1_normal']
    else:
        contact_points = end_points['sa2_xyz']
        contact_points_normals = end_points['fp2_normals']

    pre_dofs = pre_joint(end_points['pre_joint_dof_cls'], end_points['pre_joint_dof_res'], dof_scope, dof_scope)
    pre_joint_0s = pre_joint(end_points['pre_joint_0_cls'], end_points['pre_joint_0_res'], joint_scope, joint_scope)
    pre_joint_1s = pre_joint(end_points['pre_joint_1_cls'], end_points['pre_joint_1_res'], joint_scope, joint_scope)
    pre_joint_2s = pre_joint(end_points['pre_joint_2_cls'], end_points['pre_joint_2_res'], joint_scope, joint_scope)
    # pred_finger_idx_score = end_points['pre_finger_score']
    # pre_finger_idx = torch.argmax(pred_finger_idx_score, 1)
    # end_points['pre_finger_idx'] = pre_finger_idx

    grasp_transforms_eulor_all = end_points['pred_eulor'].transpose(2, 1).squeeze(0).detach().cpu().numpy()
    grasp_transforms_eulor_test = scipy.spatial.transform.Rotation.from_euler('zxy',
                                                                              grasp_transforms_eulor_all[:, :3])
    rot_test = grasp_transforms_eulor_test.as_matrix()
    tra_test = np.expand_dims(grasp_transforms_eulor_all[:, 3:], -1)
    add = np.expand_dims(
        np.repeat(np.asarray([0.0, 0.0, 0.0, 1.0]).reshape((1, 4)), grasp_transforms_eulor_all.shape[0], axis=0), 1)
    pred_pose = np.concatenate(
        (np.concatenate((rot_test, tra_test), -1), add), 1
    )

    end_points['pred_pose'] = pred_pose
    end_points['pre_dofs'] = torch.deg2rad(pre_dofs)
    end_points['pre_joint0s'] = torch.deg2rad(pre_joint_0s)
    end_points['pre_joint1s'] = torch.deg2rad(pre_joint_1s)
    end_points['pre_joint2s'] = torch.deg2rad(pre_joint_2s)

    return pred_pose


def pre_joint_f(pre_cls, pre_res, shape1, shape2, cls, degree):
    pre_dof_cls = torch.argmax(pre_cls.transpose(2, 1).contiguous(), -1)
    pre_dof_res = torch.gather(pre_res.transpose(2, 1).contiguous(), 2,
                               pre_dof_cls.unsqueeze(-1))
    pre_dofs = np.zeros((shape1, shape2))
    for i in range(shape1):
        for j in range(shape2):
            pre_dofs[i][j] = class2angle(pre_dof_cls[i][j], pre_dof_res[i][j], cls, degree)
    return pre_dofs


def pre_joint(pre_cls, pre_res, scope, res_deg):
    joint_cls = torch.argmax(pre_cls, dim=1)
    joint_res_norm = torch.gather(pre_res, dim=1, index=joint_cls.unsqueeze(dim=1)).squeeze(1)
    joint_res = joint_res_norm * scope
    # pre_joint_deg = joint_cls.float() * scope + joint_res + res_deg / 3
    pre_joint_deg = joint_cls.float() * scope + joint_res
    # pre_joint_deg = pre_joint_deg + scope / 3
    # pre_joint = torch.deg2rad(pre_joint_deg)
    return pre_joint_deg


def compute_pose_simple(end_points, args):
    B, N = end_points['pre_xs'].shape
    # contact_points = end_points['sa2_xyz']
    # contact_points_normals = end_points['fp2_normals']
    grasp_mask = end_points['label_grasp_success_pc']

    pre_x = end_points['pre_xs']
    pre_y = end_points['pre_ys']
    label_x = end_points['label_x']
    label_y = end_points['label_y']
    # tmp = torch.argmax(end_points['pre_dof_score'], 1)

    pred_pose_simple = torch.cat((pre_x.unsqueeze(-1), pre_y.unsqueeze(-1)), -1)
    label_pose_simple = torch.cat((label_x.unsqueeze(-1), label_y.unsqueeze(-1)), -1).cuda()
    pose_loss = torch.mean(huber_loss(pred_pose_simple - label_pose_simple, delta=1), -1)
    pose_loss = 10 * torch.sum(pose_loss * grasp_mask) / (torch.sum(grasp_mask) + 1e-6)

    end_points['pose_loss'] = pose_loss

    return pose_loss, end_points


def compute_eulor_loss(end_points, args):
    pred_eulor = end_points['pred_eulor']
    label_eulor = end_points['grasp_transforms_eulor'].float()
    pose_loss = F.mse_loss(pred_eulor, label_eulor.transpose(2, 1))
    return pose_loss.float(), end_points


def angle2class(angle, cls, sum_angle):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle % (sum_angle)
    assert (angle >= 0 and angle <= sum_angle)
    angle_per_class = sum_angle / float(cls)
    shifted_angle = (angle + angle_per_class / 2) % (sum_angle)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle.cuda()


def angle2class_batch(angle, cls, sum_angle):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle % (sum_angle)
    # assert (angle >= 0 and angle <= sum_angle)
    angle_per_class = sum_angle / float(cls)
    shifted_angle = (angle + angle_per_class / 2) % (sum_angle)
    class_id = (shifted_angle / angle_per_class).floor()
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, cls, sum_angle, to_label_format=False):
    ''' Inverse function to angle2class '''
    angle_per_class = sum_angle / float(cls)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def calculate_classification_statistics(y_pred, y_true):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    return tp, tn, fp, fn


# def decode_pred(end_points):

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
