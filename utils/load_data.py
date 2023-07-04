import os
import glob
import pickle
import numpy as np
import torch
import trimesh
from tqdm import tqdm
import copy


def normal_vector(vector):
    vector_norm = np.linalg.norm(vector, axis=1)
    vector = (vector / vector_norm)
    return vector


def load_scene(Data_dir, mode, start, end):
    scene_path = os.path.join(Data_dir, mode, '*')
    scene_infos = []
    scene_files = glob.glob(scene_path)
    scene_files.sort()
    scene_files = scene_files[start: end]
    pbar = tqdm(scene_files)
    for scene_file in pbar:
        scene_output = pickle.load(open(scene_file, 'rb'))
        scene_id = scene_output['scene_id']
        obj_names_in_scene = scene_output['object_ids']
        obj_transforms = scene_output['object_transforms']
        scene_final_grasp = scene_output['grasp_transforms']
        scene_final_contact = scene_output['contact_points']
        scene_final_contact_face_normal = scene_output['contact_face_normals']
        scene_final_finger_vector = scene_output['finger_vectors']
        scene_final_joint = scene_output['joints']
        scene_final_valid_location = scene_output['valid_locations']
        scene_final_result = scene_output['success']
        scene_xs = []
        scene_ys = []
        scene_x_frames = []
        for final_contact_face_normal, final_finger_vector, final_valid_location \
                in zip(scene_final_contact_face_normal, scene_final_finger_vector, scene_final_valid_location):
            x = []
            y = []
            scene_x_frame = []
            for index in range(len(final_valid_location)):
                if not final_valid_location[index]:
                    x.append(0)
                    y.append(0)
                    continue
                B_frame_z = normal_vector(final_contact_face_normal[index].reshape(1, 3))
                B_frame_y = normal_vector(np.asarray([0, -B_frame_z[0][2], B_frame_z[0][1]]).reshape(1, 3))
                B_frame_x = normal_vector(np.cross(B_frame_y, B_frame_z))

                BC = final_finger_vector[index]
                B_x = np.dot(BC, B_frame_x.T)[0]
                B_y = np.dot(BC, B_frame_y.T)[0]
                B_z = np.dot(BC, B_frame_z.T)[0]

                assert round(np.linalg.norm(np.asarray([B_x, B_y, B_z]), axis=0), 2) == 1
                x.append(B_x)
                y.append(B_y)
                scene_x_frame.append(B_frame_x)
            scene_xs.append(x)
            scene_ys.append(y)
            scene_x_frames.append(scene_x_frame)
        scene_xs = np.asarray(scene_xs)
        scene_ys = np.asarray(scene_ys)

        scene_info = {
            'scene_id': scene_id,
            'object_ids': obj_names_in_scene,
            'object_transforms': obj_transforms,
            'grasp_transforms': scene_final_grasp,
            'contact_points': scene_final_contact,
            'contact_face_normals': scene_final_contact_face_normal,
            'joints': scene_final_joint,
            'finger_vector': scene_final_finger_vector,
            'xs': scene_xs,
            'ys': scene_ys,
            'x_frames': scene_x_frames,
            'valid_locations': scene_final_valid_location,
            'success': scene_final_result
        }
        scene_infos.append(scene_info)
        pbar.set_description("Processing %s" % scene_file)

    return scene_infos


def get_finger_infos(scene_infos, fingers_idx):
    map_fingeridx_to_jointidx = {
        0: 1,
        1: 4,
        2: 6
    }
    # jointindx = map_fingeridx_to_jointidx[fingers_idx]
    finger_infos = []
    pbar = tqdm(scene_infos)
    for scene_info in pbar:
        valid_locations = scene_info['valid_locations']
        scene_final_result = scene_info['success']
        scene_id = scene_info['scene_id']
        obj_names_in_scene = scene_info['object_ids']
        obj_transforms = scene_info['object_transforms']

        scene_final_grasp = scene_info['grasp_transforms']
        scene_final_contact = scene_info['contact_points']
        scene_final_contact_face_normal = scene_info['contact_face_normals']
        scene_final_joint = scene_info['joints']
        scene_final_finger_vector = scene_info['finger_vector']
        xs = scene_info['xs']
        ys = scene_info['ys']
        # dofs = scene_info['dofs']

        scene_final_finger_grasp = []
        scene_final_finger_contact = []
        scene_final_finger_contact_face_normal = []
        scene_final_finger_joint = []
        scene_final_finger_finger_vector = []
        scene_final_all_finger_joint = []
        scene_final_finger_idx = []
        finger_xs = []
        finger_ys = []
        finger_dofs = []
        for grasp_index in range(len(valid_locations)):
            for finger_idx in fingers_idx:
                if not scene_final_result[grasp_index]:
                    continue
                if not valid_locations[grasp_index][finger_idx]:
                    continue
                jointindx = map_fingeridx_to_jointidx[finger_idx]
                scene_final_finger_grasp.append(np.expand_dims(scene_final_grasp[grasp_index], axis=0))
                scene_final_finger_contact.append(np.expand_dims(scene_final_contact[grasp_index][finger_idx], axis=0))
                scene_final_finger_contact_face_normal.append(
                    np.expand_dims(scene_final_contact_face_normal[grasp_index][finger_idx], axis=0))
                scene_final_finger_joint.append(np.expand_dims(scene_final_joint[grasp_index][jointindx], axis=0))
                scene_final_all_finger_joint.append(np.expand_dims(scene_final_joint[grasp_index], axis=0))
                scene_final_finger_finger_vector.append(
                    np.expand_dims(scene_final_finger_vector[grasp_index][finger_idx], axis=0))
                scene_final_finger_idx.append(np.expand_dims(finger_idx, axis=0))
                finger_xs.append(np.expand_dims(xs[grasp_index][finger_idx], axis=0))
                finger_ys.append(np.expand_dims(ys[grasp_index][finger_idx], axis=0))
                # finger_dofs.append(np.expand_dims(dofs[grasp_index], axis=0))
                assert scene_final_joint[grasp_index][0] == scene_final_joint[grasp_index][3]  # xiugai
                finger_dofs.append(np.expand_dims(scene_final_joint[grasp_index][0], axis=0))

        scene_final_finger_grasp = np.concatenate(scene_final_finger_grasp, 0)
        scene_final_finger_contact = np.concatenate(scene_final_finger_contact, 0)
        scene_final_finger_contact_face_normal = np.concatenate(scene_final_finger_contact_face_normal, 0)
        scene_final_finger_joint = np.concatenate(scene_final_finger_joint, 0)
        scene_final_all_finger_joint = np.concatenate(scene_final_all_finger_joint, 0)
        joint0 = scene_final_all_finger_joint[:, 1]
        joint1 = scene_final_all_finger_joint[:, 4]
        joint2 = scene_final_all_finger_joint[:, 6]
        scene_final_finger_finger_vector = np.concatenate(scene_final_finger_finger_vector, 0)
        scene_final_finger_idx = np.concatenate(scene_final_finger_idx, 0)
        finger_xs = np.concatenate(finger_xs, 0)
        finger_ys = np.concatenate(finger_ys, 0)
        finger_dofs = np.concatenate(finger_dofs, 0)

        finger_info = {
            'scene_id': scene_id,
            'object_ids': obj_names_in_scene,
            'object_transforms': obj_transforms,
            'grasp_transforms': scene_final_finger_grasp,
            'pos_contact_points': scene_final_finger_contact,
            'contact_face_normals': scene_final_finger_contact_face_normal,
            'joints': scene_final_finger_joint,
            'joint0s': scene_final_all_finger_joint,
            'finger_vector': scene_final_finger_finger_vector,
            'finger_idxs': scene_final_finger_idx,
            'xs': finger_xs,
            'ys': finger_ys,
            'dof': finger_dofs,
            'joint0': joint0,
            'joint1': joint1,
            'joint2': joint2,
        }
        finger_infos.append(finger_info)
        pbar.set_description("Processing %s" % grasp_index)
    return finger_infos


def load_contact_grasps(finger_infos, num_pos_contacts=8000):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory
    """

    grasp_infos = {}
    pos_contact_points = []
    pos_contact_face_normals = []
    pos_grasp_transform = []
    finger_idxs = []
    finger_vector = []
    dof = []
    joint0s = []
    joint1s = []
    joint2s = []

    for i, c in enumerate(finger_infos):
        all_objects = c['object_ids']
        all_contact_points = c['pos_contact_points'].reshape(-1, 3)
        all_contact_suc = np.ones_like(all_contact_points[:, 0])
        all_grasp_transform = c['grasp_transforms'].reshape(-1, 4, 4)
        all_contact_face_normals = c['contact_face_normals']
        all_finger_idxs = c['finger_idxs']
        all_finger_vector = c['finger_vector']
        all_dof = c['dof']
        joint0 = c['joint0']
        joint1 = c['joint1']
        joint2 = c['joint2']

        pos_idcs = np.where(all_contact_suc > 0)[0]
        if len(pos_idcs) == 0:
            continue
        print('the number of objects %d' % len(all_objects))
        print('total positive contact points ', len(pos_idcs))

        all_pos_contact_points = all_contact_points[pos_idcs]

        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points):
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)),
                                                                    num_pos_contacts - len(all_pos_contact_points),
                                                                    replace=True)
            pos_sampled_contact_idcs = np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts,
                                                        replace=False)
        # pos_sampled_contact_idcs = np.asarray(list([0]))
        pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs, :])
        pos_contact_face_normals.append(all_contact_face_normals[pos_sampled_contact_idcs, :])
        pos_grasp_transform.append(all_grasp_transform[pos_sampled_contact_idcs, :])
        finger_idxs.append(all_finger_idxs[pos_sampled_contact_idcs])
        finger_vector.append(all_finger_vector[pos_sampled_contact_idcs, :])
        dof.append(all_dof[pos_sampled_contact_idcs])
        joint0s.append(joint0[pos_sampled_contact_idcs])
        joint1s.append(joint1[pos_sampled_contact_idcs])
        joint2s.append(joint2[pos_sampled_contact_idcs])

    torch_scene_idcs = torch.IntTensor(np.arange(0, len(pos_contact_points)))
    torch_pos_contact_points = torch.FloatTensor(np.array(pos_contact_points))
    torch_pos_contact_face_normals = torch.FloatTensor(np.array(pos_contact_face_normals))
    torch_finger_idxs = torch.IntTensor(np.array(finger_idxs))
    torch_finger_vector = torch.FloatTensor(np.array(finger_vector))
    torch_dofs = torch.FloatTensor(np.array(dof))
    torch_joint0s = torch.FloatTensor(np.array(joint0s))
    torch_joint1s = torch.FloatTensor(np.array(joint1s))
    torch_joint2s = torch.FloatTensor(np.array(joint2s))
    torch_grasp_transforms = torch.FloatTensor(np.array(pos_grasp_transform))

    grasp_infos['scene_idcs'] = torch_scene_idcs
    grasp_infos['pos_contact_points'] = torch_pos_contact_points
    grasp_infos['pos_contact_face_normals'] = torch_pos_contact_face_normals
    grasp_infos['finger_idxs'] = torch_finger_idxs
    grasp_infos['finger_vector'] = torch_finger_vector
    grasp_infos['dofs'] = torch_dofs
    grasp_infos['joint0s'] = torch_joint0s
    grasp_infos['joint1s'] = torch_joint1s
    grasp_infos['joint2s'] = torch_joint2s
    grasp_infos['grasp_transforms'] = torch_grasp_transforms

    return grasp_infos
