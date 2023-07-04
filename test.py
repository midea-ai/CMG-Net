import argparse
import os
import time

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from load_data import load_scene, get_finger_infos, load_contact_grasps
from config_utils import load_config
from render_utils import PointCloudReader
from grasp_tmp import GraspTmp
from pc_utils import center_pc_convert_cam
from losses import get_loss_new, compute_pose
from logger import setup_logger
from dataset_load import graspTmpDataset
from sim_new import ClutterRemovalSim
from transformer import Transform

object_set = 'barrett_object'
gripper_path_for_pybullet = './barrett_hand_description/robots/bh.urdf'


def parse_option():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--input_feature', default=0, type=int, help='backbone input feature dim')
    parser.add_argument('--dof_dim', default=6, type=int, help='pred pose dof dim')
    parser.add_argument('--joint_dim', default=6, type=int, help='pred pose joint dim')

    # Loss
    parser.add_argument('--cls_weight', default=1.0, type=float, help='Loss weight for sem cls and finger index')
    parser.add_argument('--finger_cls_weight', default=1.0, type=float, help='Loss weight for sem cls and finger index')
    parser.add_argument('--pose_weight', default=1.0, type=float, help='Loss weight for pose xy')
    parser.add_argument('--dof_weight', default=1.0, type=float, help='Loss weight for dof loss')
    parser.add_argument('--joint_weight', default=1.0, type=float, help='Loss weight for joint loss')
    parser.add_argument('--fully_pose_loss', action='store_true', help='if use finally pose to compute pose loss')

    # Data
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU during training phase')
    parser.add_argument('--data_root', default='./dataset', help='data root path')
    parser.add_argument('--obj_related_path', default='./urdfs/barrett_object/ycb_meshes', help='object path')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--fingers_idx', type=int, default=[0, 1, 2], nargs='+', help='index of fingers')

    # Training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run')
    parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path',
                        default='/home/midea/wmz/grasp_tmp/checkpoints/39_TGraspNet.pth',
                        help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=50, help='val frequency')

    # others
    parser.add_argument("--local_rank", default=-1, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    return args


def load_checkpoint(args, model):
    # Load checkpoint if there is any
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        save_path = checkpoint.get('save_path', 'none')
        for k in list(state_dict.keys()):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict)
        logger.info(f"{args.checkpoint_path} loaded successfully!!!")

        del checkpoint
        torch.cuda.empty_cache()
    else:
        raise FileNotFoundError
    return save_path

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def main(args):

    # test
    test_scene_start = 0
    test_scene_end = 1
    obj_remain = 1
    config_path = './config/config.yaml'
    global_config = load_config(config_path, save=False)

    # load scene
    contact_infos = load_scene(args.data_root,
                               mode='test',
                               start=test_scene_start,
                               end=test_scene_end)

    finger_infos = get_finger_infos(contact_infos, args.fingers_idx)
    if args.use_height:
        num_input_channel = int(args.use_normal) * 3 + 1
    else:
        num_input_channel = int(args.use_normal) * 3
    model = GraspTmp(num_input_channel)
    logger.info(str(model))
    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        load_checkpoint(args, model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model = model.cuda()
    # grasp_infos = load_contact_grasps(finger_infos)
    flag = 0
    grasp_all_cnt = {}
    grasp_succes_cnt = {}
    quality_all_cnt = {}

    batch_idx = 0
    while batch_idx < len(finger_infos):
        if batch_idx not in grasp_all_cnt.keys():
            quality_all_cnt[batch_idx] = 0
            grasp_all_cnt[batch_idx] = 0
            grasp_succes_cnt[batch_idx] = 0

        object_transforms = finger_infos[batch_idx]['object_transforms']
        object_ids = finger_infos[batch_idx]['object_ids']

        # render the scene
        pcRender = PointCloudReader(
            args.data_root,
            args.obj_related_path,
            batch_size=1,
            raw_num_points=global_config['DATA']['raw_num_points'],
            estimate_normals=global_config['DATA']['input_normals'],
            caching=True,
            scene_obj_paths=[c['object_ids'] for c in finger_infos],
            scene_obj_transforms=[c['object_transforms'] for c in finger_infos],
            num_scene=None,
            use_farthest_point=global_config['DATA']['use_farthest_point'],
            intrinsics=global_config['DATA']['intrinsics'],
            distance_range=global_config['DATA']['view_sphere']['distance_range'],
            elevation=global_config['DATA']['view_sphere']['elevation'],
            pc_augm_config=global_config['DATA']['pc_augm'],
            depth_augm_config=global_config['DATA']['depth_augm']
        )
        batch_data, cam_poses, scene_idx = pcRender.get_scene_batch(scene_idx=batch_idx)
        cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)
        cam_poses = torch.FloatTensor(cam_poses).cuda()
        batch_data = torch.FloatTensor(batch_data)
        begin_time = time.time()
        batch_data = batch_data.cuda()
        end_points = model(batch_data)
        end_time = time.time()

        pred_pose, pre_dofs, pre_joint_0s, pre_joint_1s, pre_joint_2s = compute_pose(end_points)

        pre_dofs = end_points['pre_dofs']
        pre_joint0s = end_points['pre_joint0s']
        pre_joint1s = end_points['pre_joint1s']
        pre_joint2s = end_points['pre_joint2s']
        pre_finger_index = end_points['pre_finger_idx']

        score = F.softmax(end_points['pre_sem_score'], dim=1)[:, 1]
        topk_val, topk_indx = torch.topk(score, 10, 1)
        topk_idx = topk_indx.squeeze(0)

        infact_select = torch.randint(0, 10, [3])
        fingers_idx = torch.index_select(pre_finger_index, 1, topk_idx)[:, infact_select]
        grasp_transforms = torch.index_select(pred_pose, 1, topk_idx)[:, infact_select, :, :]
        dofs = torch.index_select(pre_dofs.cuda(), 1, topk_idx)[:, infact_select]
        joint0s = torch.index_select(pre_joint0s.cuda(), 1, topk_idx)[:, infact_select]
        joint1s = torch.index_select(pre_joint1s.cuda(), 1, topk_idx)[:, infact_select]
        joint2s = torch.index_select(pre_joint2s.cuda(), 1, topk_idx)[:, infact_select]

        inference_time = end_time - begin_time
        print("inference time is {}".format(inference_time))

        map_objecid_id = {}
        for i, object_id in enumerate(object_ids):
            map_objecid_id[i + 1] = object_id

        sim = ClutterRemovalSim(object_set, gripper_path=gripper_path_for_pybullet, gui=True)
        sim.reset(object_transforms, object_ids, vis=True)
        sim.save_state()

        for finger_idx, grasp_transform, dof, joint0, joint1, joint2 in zip(
                fingers_idx.squeeze(0).detach().cpu().numpy(),
                grasp_transforms.squeeze(
                    0).detach().cpu().numpy(),
                dofs.squeeze(0).detach().cpu().numpy(),
                joint0s.squeeze(0).detach().cpu().numpy(),
                joint1s.squeeze(0).detach().cpu().numpy(),
                joint2s.squeeze(0).detach().cpu().numpy()):
            fake_joint = [dof, joint0, joint0 / 3, dof, joint1, joint1 / 3, joint2, joint2 / 3]
            sim.restore_state()
            grasp_transform = torch.matmul(torch.from_numpy(inverse_transform(cam_poses.squeeze(0).cpu().numpy())),
                                           torch.from_numpy(grasp_transform)).cpu().numpy()
            # filter
            v1 = grasp_transform[0:3, 2]
            v2 = np.array([0, 0, 1])
            angle = angle_between(v1, v2)
            angle = np.degrees(angle)
            if angle < 150:
                print("Angle is not valid!")
                continue

            T_refer_barrett = Transform.from_matrix(grasp_transform)

            T_world_refer = Transform.identity()
            T_world_refer.translation = np.array([0.0, 0.0, -0.01])

            T_world_realpose = T_world_refer * T_refer_barrett
            grasp_transform = T_world_realpose.as_matrix()

            # execute
            result, object_record_id, quality = sim.execute_grasp_quality(grasp_transform, fake_joint)
            grasp_all_cnt[batch_idx] += 1
            if int(result):
                index = finger_infos[batch_idx]['object_ids'].index(map_objecid_id[object_record_id])
                del finger_infos[batch_idx]['object_ids'][index]
                del finger_infos[batch_idx]['object_transforms'][index]
                grasp_succes_cnt[batch_idx] += 1
                quality_all_cnt[batch_idx] += quality
                if len(finger_infos[batch_idx]['object_ids']) == obj_remain:
                    flag = 1
                break
            else:
                break
        sim.world.close()
        if flag == 1:
            flag = 0
            print('#########################')
            print('%d finish' % batch_idx)
            print('batch {} grasp success rate is {}'.format(batch_idx,
                                                             grasp_succes_cnt[batch_idx] / grasp_all_cnt[batch_idx]))
            print('#########################')

            f = "record.txt"

            with open(f, "a") as file:
                file.write('scene {} grasp success rate is {} \n'.format(batch_idx + test_scene_start,
                                                         grasp_succes_cnt[batch_idx] / grasp_all_cnt[batch_idx]))

            batch_idx += 1

    print('the average grasp success rate is {}'.format(sum(grasp_succes_cnt.values()) / sum(grasp_all_cnt.values())))

    f = "record.txt"
    with open(f, "a") as file:
        file.write('the average grasp success rate is {} \n'.format(sum(grasp_succes_cnt.values())
                                                                 / sum(grasp_all_cnt.values())))


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


if __name__ == '__main__':
    opt = parse_option()
    logger = setup_logger(output=opt.log_dir, name="eval")
    for i in range(4):
        main(opt)
