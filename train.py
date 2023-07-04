import argparse
import json
import os
import time

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from grasp_tmp import GraspTmp
from label_utils import compute_labels_new
from losses import get_loss_new
from lr_scheduler import get_scheduler
from logger import setup_logger
from dataset_load import graspTmpDataset


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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU during training phase')
    parser.add_argument('--data_root', default='dataset', help='data root path')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 50000]')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

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
    parser.add_argument('--lr_decay_epochs', type=int, default=[28, 34], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--checkpoint_dir', default='checkpoints/debug', help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=50, help='val frequency')

    # others
    parser.add_argument("--local_rank", default=-1, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    return args


def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

        # create Dataset and Dataloader

    TRAIN_DATASET = graspTmpDataset('train', num_points=args.num_point,
                                    use_normal=args.use_normal, use_height=args.use_height,
                                    data_root=args.data_root)
    print(f"train_len:{len(TRAIN_DATASET)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    return train_loader


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    if args.use_height:
        num_input_channel = int(args.use_normal) * 3 + 1
    else:
        num_input_channel = int(args.use_normal) * 3
    model = GraspTmp(num_input_channel)
    if dist.get_rank() == 0:
        logger.info(str(model))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False)
    # model = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False)

    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        load_checkpoint(args, model, optimizer, scheduler)

    for epoch in range(args.start_epoch, args.max_epoch + 1):
        train_loader.sampler.set_epoch(epoch)

        tic = time.time()

        train_one_epoch(epoch, train_loader, model, optimizer, scheduler, args)

        logger.info('epoch {}, total time {:.2f}, '
                    'lr_base {:.5f}'.format(epoch, (time.time() - tic),
                                            optimizer.param_groups[0]['lr']))

        save_dir = os.path.join(ROOT_DIR, args.checkpoint_dir)
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(save_dir, '{}_TGraspNet.pth'.format(epoch)))
            logger.info('save {} epoch model successful'.format(epoch))


def train_one_epoch(epoch, train_loader, model, optimizer, scheduler, args):
    stat_dict = dict()
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        for key in batch_data:
            if 'object' not in key and 'seg not in key':
                batch_data[key] = batch_data[key].cuda(non_blocking=True)

        input_pc = batch_data['input_point_clouds']

        # b_model = time.time()
        end_points = model(input_pc)
        # print("model forward time is {} !".format(time.time() - b_model))

        # b_label = time.time()
        end_points = compute_labels_new(batch_data, batch_data['cam_pose'], end_points)
        # print("compute label time is {} !".format(time.time() - b_label))

        # b_loss = time.time()
        loss, end_points = get_loss_new(end_points, args)
        # print("compute loss time is {} !".format(time.time() - b_loss))

        optimizer.zero_grad()
        loss.backward()
        if args.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        scheduler.step()
        stat_dict['grad_norm'] = grad_total_norm
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % args.print_freq == 0:
            if dist.get_rank() == 0:
                logger.info(' ---- epoch: %03d batch: %03d ----' % (epoch, batch_idx + 1))
                for key in sorted(stat_dict.keys()):
                    logger.info('mean %s: %f' % (key, stat_dict[key] / args.print_freq))
                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0

if __name__ == '__main__':
    import os

    opt = parse_option()

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29501'
    # dist.init_process_group(backend='nccl', rank=0,
    #                         world_size=1)
    # torch.cuda.set_device(0)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(ROOT_DIR, opt.log_dir)
    if not os.path.exists(log_dir) and dist.get_rank() == 0:
        os.mkdir(log_dir)

    logger = setup_logger(output=log_dir, distributed_rank=dist.get_rank(), name="grasp_v1")

    if dist.get_rank() == 0:
        path = os.path.join(log_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        logger.info(str(vars(opt)))

    main(opt)
