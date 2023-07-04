import copy
import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def compute_labels_test(grasp_infos, camera_pose_pl, end_points, batch_ids):
    nsample = 1
    radius = 0.005

    if 'fp3_xyz' in end_points:
        xyz_cam = end_points['sa1_xyz']
        normal_cam = end_points['sa1_normal']
    else:
        xyz_cam = end_points['fp2_xyz']
        normal_cam = end_points['fp2_normals']

    B, N, _ = xyz_cam.shape

    pc_mesh = xyz_cam
    pc_normal = normal_cam

    finger_vector = grasp_infos['finger_vector'][batch_ids].unsqueeze(0)
    pad_homog2 = torch.ones((pc_mesh.shape[0], finger_vector.shape[1], 1)).cuda()
    contact_point_batch_mesh = grasp_infos['pos_contact_points'][batch_ids].unsqueeze(0)
    contact_point_batch_mesh = contact_point_batch_mesh.repeat(pc_mesh.shape[0], 1, 1).cuda()
    contact_point_batch_cam = torch.matmul(torch.cat((contact_point_batch_mesh, pad_homog2), 2),
                                           torch.transpose(camera_pose_pl, 1, 2).float())[:, :, :3]
    dofs = grasp_infos['dofs'][batch_ids].unsqueeze(0)
    joint0 = grasp_infos['joint0s'][batch_ids].unsqueeze(0)
    joint1 = grasp_infos['joint1s'][batch_ids].unsqueeze(0)
    joint2 = grasp_infos['joint2s'][batch_ids].unsqueeze(0)
    finger_idx = grasp_infos['finger_idxs'][batch_ids].unsqueeze(0)
    grasp_transforms = grasp_infos['grasp_transforms'][batch_ids].unsqueeze(0)

    finger_vector_batch = finger_vector.cuda()
    finger_vector_batch_cam = torch.matmul(finger_vector_batch,
                                           torch.transpose(camera_pose_pl[:, :3, :3], 1, 2))[:, :, :3]

    squared_dists_all = torch.sum(
        (torch.unsqueeze(contact_point_batch_cam.cuda(), 1) - torch.unsqueeze(xyz_cam, 2)) ** 2,
        dim=3)  # 计算contact point 和 xyz的距离
    neg_squared_dists_k, close_contact_pt_idcs = torch.topk(-squared_dists_all, k=nsample,
                                                            sorted=False)  # 将距离取反，找到离已知点最近的接触点
    squared_dists_k = -neg_squared_dists_k  # 获得距离

    # Nearest neighbor mapping # less是比大小返回x<y, cast将bool变为float
    grasp_success_labels_pc = torch.lt(torch.mean(squared_dists_k, dim=2),
                                       radius * radius).float()  # (batch_size, num_point)
    close_contact_pt_idcs = torch.squeeze(close_contact_pt_idcs, dim=2)

    finger_idx = finger_idx.repeat(pc_mesh.shape[0], 1)
    joint0 = joint0.repeat(pc_mesh.shape[0], 1)
    joint1 = joint1.repeat(pc_mesh.shape[0], 1)
    joint2 = joint2.repeat(pc_mesh.shape[0], 1)
    dofs = dofs.repeat(pc_mesh.shape[0], 1)
    grasp_transforms = grasp_transforms.repeat(pc_mesh.shape[0], 1, 1, 1)

    grouped_finger_idx = index_points(torch.unsqueeze(finger_idx, 2), close_contact_pt_idcs)  # add
    grouped_finger_vector = index_points(finger_vector_batch_cam, close_contact_pt_idcs)
    grouped_joint0 = index_points(torch.unsqueeze(joint0, 2), close_contact_pt_idcs)
    grouped_joint1 = index_points(torch.unsqueeze(joint1, 2), close_contact_pt_idcs)
    grouped_joint2 = index_points(torch.unsqueeze(joint2, 2), close_contact_pt_idcs)
    grouped_dofs = index_points(torch.unsqueeze(dofs, 2), close_contact_pt_idcs)
    grouped_grasp_transforms = index_points(grasp_transforms, close_contact_pt_idcs)

    pc_normal_copy = copy.deepcopy(pc_normal)
    finger_vector_copy = copy.deepcopy(grouped_finger_vector).float()
    B_frame_z_new = pc_normal_copy
    B_frame_z_norm = F.normalize(B_frame_z_new, p=2, dim=-1)
    B_frame_y_new = torch.stack((torch.zeros((B, N)).cuda(), -B_frame_z_new[:, :, 2], B_frame_z_new[:, :, 1]), -1)
    B_frame_y_norm = F.normalize(B_frame_y_new, p=2, dim=-1)
    B_frame_x_new = torch.cross(B_frame_y_norm, B_frame_z_new)
    assert torch.sum(torch.round(torch.linalg.norm(B_frame_x_new, axis=-1)) != 1) == 0
    B_frame_x_norm = B_frame_x_new

    BC = finger_vector_copy.float()
    B_x = torch.sum(torch.mul(BC, B_frame_x_norm), -1).squeeze(-1)
    B_y = torch.sum(torch.mul(BC, B_frame_y_norm), -1).squeeze(-1)
    B_z = torch.sum(torch.mul(BC, B_frame_z_norm), -1).squeeze(-1)

    grouped_finger_idx = torch.squeeze(grouped_finger_idx, dim=2)  # add
    grouped_joint0 = torch.squeeze(grouped_joint0, dim=2)
    grouped_joint1 = torch.squeeze(grouped_joint1, dim=2)
    grouped_joint2 = torch.squeeze(grouped_joint2, dim=2)
    grouped_dofs = torch.squeeze(grouped_dofs, dim=2)
    end_points['grouped_finger_idx'] = grouped_finger_idx.cuda()
    end_points['label_grasp_success_pc'] = grasp_success_labels_pc.cuda()
    end_points['label_x'] = B_x.cuda()
    end_points['label_y'] = B_y.cuda()
    end_points['label_joint0'] = grouped_joint0.cuda()
    end_points['label_joint1'] = grouped_joint1.cuda()
    end_points['label_joint2'] = grouped_joint2.cuda()
    end_points['label_dofs'] = grouped_dofs.cuda()
    end_points['grasp_transforms'] = grouped_grasp_transforms.cuda()

    return end_points

def compute_labels_new(grasp_infos, camera_pose_pl, end_points):
    nsample = 1
    radius = 0.005

    if 'fp3_xyz' in end_points:
        xyz_cam = end_points['sa1_xyz']
        normal_cam = end_points['sa1_normal']
    else:
        xyz_cam = end_points['fp2_xyz']
        normal_cam = end_points['fp2_normals']

    B, N, _ = xyz_cam.shape

    pc_mesh = xyz_cam
    pc_normal = normal_cam

    finger_vector = grasp_infos['finger_vector']
    pad_homog2 = torch.ones((pc_mesh.shape[0], finger_vector.shape[1], 1)).cuda()
    contact_point_batch_mesh = grasp_infos['pos_contact_points']
    contact_point_batch_cam = torch.matmul(torch.cat((contact_point_batch_mesh, pad_homog2), 2),
                                           torch.transpose(camera_pose_pl, 1, 2).double())[:, :, :3]
    dofs = grasp_infos['dof']
    joint0 = grasp_infos['joint0']
    joint1 = grasp_infos['joint1']
    joint2 = grasp_infos['joint2']
    finger_idx = grasp_infos['finger_idxs']
    grasp_transforms = grasp_infos['grasp_transforms']
    joint0s = grasp_infos['joint0s']

    finger_vector_batch = finger_vector
    finger_vector_batch_cam = torch.matmul(finger_vector_batch,
                                           torch.transpose(camera_pose_pl[:, :3, :3], 1, 2).double())[:, :, :3]

    squared_dists_all = torch.sum(
        (torch.unsqueeze(contact_point_batch_cam.cuda(), 1) - torch.unsqueeze(xyz_cam, 2)) ** 2,
        dim=3)  # 计算contact point 和 xyz的距离
    neg_squared_dists_k, close_contact_pt_idcs = torch.topk(-squared_dists_all, k=nsample,
                                                            sorted=False)  # 将距离取反，找到离已知点最近的接触点
    squared_dists_k = -neg_squared_dists_k  # 获得距离

    # Nearest neighbor mapping # less是比大小返回x<y, cast将bool变为float
    grasp_success_labels_pc = torch.lt(torch.mean(squared_dists_k, dim=2),
                                       radius * radius).float()  # (batch_size, num_point)

    close_contact_pt_idcs = torch.squeeze(close_contact_pt_idcs, dim=2)

    grouped_finger_idx = index_points(torch.unsqueeze(finger_idx, 2), close_contact_pt_idcs)  # add
    grouped_finger_vector = index_points(finger_vector_batch_cam, close_contact_pt_idcs)
    grouped_joint0 = index_points(torch.unsqueeze(joint0, 2), close_contact_pt_idcs)
    grouped_joint1 = index_points(torch.unsqueeze(joint1, 2), close_contact_pt_idcs)
    grouped_joint2 = index_points(torch.unsqueeze(joint2, 2), close_contact_pt_idcs)

    grouped_dofs = index_points(torch.unsqueeze(dofs, 2), close_contact_pt_idcs)

    grouped_grasp_transforms = index_points(grasp_transforms, close_contact_pt_idcs)
    grouped_joint0s = index_points(joint0s, close_contact_pt_idcs)

    pc_normal_copy = copy.deepcopy(pc_normal)
    finger_vector_copy = copy.deepcopy(grouped_finger_vector).float()

    B_frame_z_new = pc_normal_copy
    B_frame_z_norm = F.normalize(B_frame_z_new, p=2, dim=-1)
    B_frame_y_new = torch.stack((torch.zeros((B, N)).cuda(), -B_frame_z_new[:, :, 2], B_frame_z_new[:, :, 1]), -1)
    B_frame_y_norm = F.normalize(B_frame_y_new, p=2, dim=-1)
    B_frame_x_new = torch.cross(B_frame_y_norm, B_frame_z_new)
    assert torch.sum(torch.round(torch.linalg.norm(B_frame_x_new, axis=-1)) != 1) == 0
    B_frame_x_norm = B_frame_x_new

    BC = finger_vector_copy.float()
    B_x = torch.sum(torch.mul(BC, B_frame_x_norm), -1).squeeze(-1)
    B_y = torch.sum(torch.mul(BC, B_frame_y_norm), -1).squeeze(-1)
    B_z = torch.sum(torch.mul(BC, B_frame_z_norm), -1).squeeze(-1)

    grouped_finger_idx = torch.squeeze(grouped_finger_idx, dim=2)  # add
    # grouped_x = torch.squeeze(grouped_x, dim=2)
    # grouped_y = torch.squeeze(grouped_y, dim=2)
    grouped_joint0 = torch.squeeze(grouped_joint0, dim=2)
    grouped_joint1 = torch.squeeze(grouped_joint1, dim=2)
    grouped_joint2 = torch.squeeze(grouped_joint2, dim=2)
    grouped_dofs = torch.squeeze(grouped_dofs, dim=2)
    end_points['grouped_finger_idx'] = grouped_finger_idx
    end_points['label_grasp_success_pc'] = grasp_success_labels_pc
    end_points['label_x'] = B_x
    end_points['label_y'] = B_y
    end_points['label_joint0'] = grouped_joint0
    end_points['label_joint1'] = grouped_joint1
    end_points['label_joint2'] = grouped_joint2
    end_points['label_dofs'] = grouped_dofs
    end_points['grasp_transforms'] = grouped_grasp_transforms
    end_points['joint_all'] = grouped_joint0s

    return end_points