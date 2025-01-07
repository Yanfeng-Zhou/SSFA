from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import cv2
from skimage.morphology import skeletonize
from scipy import ndimage
import math

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset import imagefloder_TA
from config.visdom_config.visual_visdom import visdom_initialization_TA, visualization_TA, visual_image_TA
from config.train_test_config.train_test_config import print_train_loss_TA, print_val_loss, print_train_eval, print_val_eval, save_val_best_TA, draw_pred_TA, print_best
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='/mnt/data1/Semi-Topology-Attention/checkpoints')
    parser.add_argument('--path_seg_results', default='/mnt/data1/Semi-Topology-Attention/seg_pred')
    parser.add_argument('-pd', '--path_dataset', default='/mnt/data1/Semi-Topology-Attention/dataset/CBMI_ER')
    parser.add_argument('--dataset_name', default='CBMI_ER', help='CBMI_ER, CREMI, STARE_DRIVE, SNEMI3D')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=100, type=int)
    parser.add_argument('-s', '--step_size', default=25, type=int)
    parser.add_argument('-l', '--lr', default=0.8, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=3, type=float)
    parser.add_argument('--loss', default='CE')
    parser.add_argument('--loss_weight', default=0.0, type=float)
    parser.add_argument('--attention_loss', default='WCE')
    parser.add_argument('--attention_loss_weight', default=1.0, type=float)

    # attention
    parser.add_argument('-su', '--start_update', default=20, type=int)
    parser.add_argument('-um', '--update_method', default='new', help='raw, both, new')
    parser.add_argument('-aw', '--attention_weight', default=0.8)
    # structure
    parser.add_argument('--structure_skl_kernel_size', default=5, type=int)
    parser.add_argument('--structure_level', default=10, type=int)
    parser.add_argument('--structure_range', default=1.5, type=float)
    parser.add_argument('--structure_thr', default=1, type=int)
    # semantic
    parser.add_argument('--semantic_skl_kernel_size', default=5, type=int)
    parser.add_argument('--semantic_level', default=4, type=int)
    parser.add_argument('--semantic_range', default=15, type=int)

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-n', '--network', default='unet_ta_v1', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # trained model save
    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models + '/' + 'TA' + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-um=' + str(args.update_method) + '-su=' + str(args.start_update) + '-aw=' + str(args.attention_weight)+'-lw=' + str(args.loss_weight)+'-alw=' + str(args.attention_loss_weight)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)

    # seg results save
    path_seg_results = args.path_seg_results + '/' + str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + 'TA' + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-um=' + str(args.update_method) + '-su=' + str(args.start_update) + '-aw=' + str(args.attention_weight)+'-lw=' + str(args.loss_weight)+'-alw=' + str(args.attention_loss_weight)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    # vis
    if args.vis and rank == args.rank_index:
        visdom_env = str('TA-' + str(os.path.split(args.path_dataset)[1]) + '-' + args.network + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) + '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) + '-cw=' + str(args.unsup_weight) + '-um=' + str(args.update_method) + '-su=' + str(args.start_update) + '-aw=' + str(args.attention_weight)+'-lw=' + str(args.loss_weight)+'-alw=' + str(args.attention_loss_weight))
        visdom = visdom_initialization_TA(env=visdom_env, port=args.visdom_port)


    data_transforms = data_transform(cfg['SIZE'], cfg['MEAN'], cfg['STD'])

    dataset_train_unsup = imagefloder_TA(
        data_dir=args.path_dataset + '/train_unsup',
        data_transform=data_transforms['train'],
        sup=False,
        num_images=None,
    )
    num_images_unsup = len(dataset_train_unsup)

    dataset_train_sup = imagefloder_TA(
        data_dir=args.path_dataset + '/train_sup',
        data_transform=data_transforms['train'],
        sup=True,
        num_images=num_images_unsup,
    )
    dataset_val = imagefloder_TA(
        data_dir=args.path_dataset + '/val',
        data_transform=data_transforms['val'],
        sup=True,
        num_images=None,
    )

    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup, shuffle=True)
    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model2 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    model1 = model1.cuda()
    model2 = model2.cuda()
    model1 = DistributedDataParallel(model1, device_ids=[args.local_rank])
    model2 = DistributedDataParallel(model2, device_ids=[args.local_rank])
    dist.barrier()

    criterion = segmentation_loss(args.loss).cuda()
    criterion_attention = segmentation_loss(args.attention_loss).cuda()

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)

    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)

    since = time.time()
    count_iter = 0

    best_model = model1
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(3)]
    best_val_eval_list[2] = 1000000

    structure_skl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.structure_skl_kernel_size, args.structure_skl_kernel_size))
    semantic_skl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.semantic_skl_kernel_size, args.semantic_skl_kernel_size))
    semantic_range_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.semantic_range, args.semantic_range))

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0
        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        dist.barrier()

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup = Variable(unsup_index['image'].cuda())
            structure1_train_unsup = Variable(unsup_index['structure1'].cuda())
            structure2_train_unsup = Variable(unsup_index['structure2'].cuda())
            semantic_train_unsup = Variable(unsup_index['semantic'].cuda())
            attention_train_unsup = Variable(unsup_index['attention'].cuda())

            sup_index = next(dataset_train_sup)
            img_train_sup = Variable(sup_index['image'].cuda())
            structure1_train_sup = Variable(sup_index['structure1'].cuda())
            structure2_train_sup = Variable(sup_index['structure2'].cuda())
            semantic_train_sup = Variable(sup_index['semantic'].cuda())
            attention_train_sup = Variable(sup_index['attention'].cuda())
            mask_train_sup = Variable(sup_index['mask'].cuda())

            if args.start_update <= epoch:
                with torch.no_grad():
                    model1.eval()
                    model2.eval()

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    pred_train_unsup1 = model1(img_train_unsup, structure1_train_unsup.unsqueeze(1), structure2_train_unsup.unsqueeze(1), semantic_train_unsup.unsqueeze(1))
                    pred_train_unsup2 = model2(img_train_unsup, structure1_train_unsup.unsqueeze(1), structure2_train_unsup.unsqueeze(1), semantic_train_unsup.unsqueeze(1))

                    max_train1 = torch.max(pred_train_unsup1, dim=1)[1]
                    max_train2 = torch.max(pred_train_unsup2, dim=1)[1]

                    max_train1_np = max_train1.data.cpu().numpy().astype(np.uint8)
                    max_train2_np = max_train2.data.cpu().numpy().astype(np.uint8)

                    structure1_list_unsup = []
                    structure2_list_unsup = []
                    semantic_list_unsup = []
                    attention_list_unsup = []

                    for sample in range(max_train1_np.shape[0]):
                        train_unsup1_pred_sample = max_train1_np[sample, :, :]
                        train_unsup2_pred_sample = max_train2_np[sample, :, :]

                        train_unsup1_skl_sample = skeletonize(train_unsup1_pred_sample, method='lee')
                        train_unsup2_skl_sample = skeletonize(train_unsup2_pred_sample, method='lee')

                        # structure
                        train_unsup1_skl_sample_ = cv2.dilate(train_unsup1_skl_sample, structure_skl_kernel)
                        train_unsup1_dis_sample = ndimage.distance_transform_edt(train_unsup1_pred_sample)
                        train_unsup2_skl_sample_ = cv2.dilate(train_unsup2_skl_sample, structure_skl_kernel)
                        train_unsup2_dis_sample = ndimage.distance_transform_edt(train_unsup2_pred_sample)

                        structure_map_list1 = []
                        structure_map_list2 = []
                        diff_thr_before1 = np.zeros(train_unsup1_pred_sample.shape)
                        diff_thr_before2 = np.zeros(train_unsup2_pred_sample.shape)
                        max_dis1 = train_unsup1_dis_sample.max()
                        max_dis2 = train_unsup2_dis_sample.max()
                        interval1 = (max_dis1 - 1) / (args.structure_level - 1)
                        interval2 = (max_dis2 - 1) / (args.structure_level - 1)

                        for level in range(args.structure_level):

                            thr1 = 1.0 + level * interval1
                            dis_thr1 = train_unsup1_dis_sample.copy()
                            dis_thr1[train_unsup1_dis_sample <= thr1] = 0
                            pred_thr1 = train_unsup1_pred_sample.copy()
                            pred_thr1[dis_thr1 == 0] = 0

                            thr2 = 1.0 + level * interval2
                            dis_thr2 = train_unsup2_dis_sample.copy()
                            dis_thr2[train_unsup2_dis_sample <= thr2] = 0
                            pred_thr2 = train_unsup2_pred_sample.copy()
                            pred_thr2[dis_thr2 == 0] = 0

                            skl_thr1 = skeletonize(pred_thr1, method='lee')
                            skl_thr1_ = cv2.dilate(skl_thr1, structure_skl_kernel)
                            skl_thr2 = skeletonize(pred_thr2, method='lee')
                            skl_thr2_ = cv2.dilate(skl_thr2, structure_skl_kernel)

                            diff_thr1 = train_unsup1_skl_sample.astype(np.float32) - skl_thr1.astype(np.float32)
                            diff_thr1[diff_thr1 == -1] = 1
                            diff_thr1[(train_unsup1_skl_sample_ == 1) & (skl_thr1_ == 1)] = 0
                            diff_thr1 = diff_thr1 - diff_thr_before1
                            diff_thr_before1 = diff_thr1

                            diff_thr2 = train_unsup2_skl_sample.astype(np.float32) - skl_thr2.astype(np.float32)
                            diff_thr2[diff_thr2 == -1] = 1
                            diff_thr2[(train_unsup2_skl_sample_ == 1) & (skl_thr2_ == 1)] = 0
                            diff_thr2 = diff_thr2 - diff_thr_before2
                            diff_thr_before2 = diff_thr2

                            structure_map1 = np.zeros(train_unsup1_pred_sample.shape)
                            x, y = np.nonzero(diff_thr1)
                            for k in range(len(x)):
                                weight_range = math.ceil(args.structure_range * train_unsup1_dis_sample[x[k], y[k]])
                                structure_map1[x[k] - weight_range:x[k] + weight_range,
                                y[k] - weight_range:y[k] + weight_range] = 1
                            structure_map_list1.append(structure_map1)

                            structure_map2 = np.zeros(train_unsup2_pred_sample.shape)
                            x, y = np.nonzero(diff_thr2)
                            for k in range(len(x)):
                                weight_range = math.ceil(args.structure_range * train_unsup2_dis_sample[x[k], y[k]])
                                structure_map2[x[k] - weight_range:x[k] + weight_range,
                                y[k] - weight_range:y[k] + weight_range] = 1
                            structure_map_list2.append(structure_map2)

                        structure_map1_ = np.zeros(train_unsup1_pred_sample.shape)
                        structure_map_list1.reverse()
                        for j in range(len(structure_map_list1)):
                            if j <= args.structure_thr:
                                structure_map1_[structure_map_list1[j] == 1] = 1
                            else:
                                structure_map1_[structure_map_list1[j] == 1] = j + 1 - args.structure_thr

                        structure_map2_ = np.zeros(train_unsup2_pred_sample.shape)
                        structure_map_list2.reverse()
                        for j in range(len(structure_map_list2)):
                            if j <= args.structure_thr:
                                structure_map2_[structure_map_list2[j] == 1] = 1
                            else:
                                structure_map2_[structure_map_list2[j] == 1] = j + 1 - args.structure_thr

                        structure_map1_[structure_map1_ == 0] = 1
                        structure_map2_[structure_map2_ == 0] = 1
                        structure1_list_unsup.append(structure_map1_)
                        structure2_list_unsup.append(structure_map2_)

                        # semantic
                        train_unsup1_skl_sample_ = cv2.dilate(train_unsup1_skl_sample, semantic_skl_kernel)
                        train_unsup2_skl_sample_ = cv2.dilate(train_unsup2_skl_sample, semantic_skl_kernel)

                        diff_sample = train_unsup1_skl_sample.astype(np.float32) - train_unsup2_skl_sample.astype(np.float32)
                        diff_sample[diff_sample == -1] = 1
                        diff_sample[(train_unsup1_skl_sample_ == 1) & (train_unsup2_skl_sample_ == 1)] = 0

                        semantic_map = cv2.dilate(diff_sample.astype(np.uint8), semantic_range_kernel)
                        semantic_map[semantic_map == 1] = args.semantic_level
                        semantic_map[semantic_map == 0] = 1
                        semantic_list_unsup.append(semantic_map)
                        # attention
                        attention_map = np.stack([semantic_map, structure_map1_, structure_map2_], axis=1)
                        attention_map = np.max(attention_map, 1)
                        # attention_map = cv2.GaussianBlur(attention_map, (9, 9), 0, 0)
                        attention_list_unsup.append(attention_map)

                    pred_train_sup1 = model1(img_train_sup, structure1_train_sup.unsqueeze(1), structure2_train_sup.unsqueeze(1), semantic_train_sup.unsqueeze(1))
                    pred_train_sup2 = model2(img_train_sup, structure1_train_sup.unsqueeze(1), structure2_train_sup.unsqueeze(1), semantic_train_sup.unsqueeze(1))

                    pred_train1 = torch.max(pred_train_sup1, dim=1)[1]
                    pred_train2 = torch.max(pred_train_sup2, dim=1)[1]

                    pred_train1_np = pred_train1.data.cpu().numpy().astype(np.uint8)
                    pred_train2_np = pred_train2.data.cpu().numpy().astype(np.uint8)

                    structure1_list_sup = []
                    structure2_list_sup = []
                    semantic_list_sup = []
                    attention_list_sup = []

                    for sample in range(pred_train1_np.shape[0]):
                        train_sup1_pred_sample = pred_train1_np[sample, :, :]
                        train_sup2_pred_sample = pred_train2_np[sample, :, :]

                        train_sup1_skl_sample = skeletonize(train_sup1_pred_sample, method='lee')
                        train_sup2_skl_sample = skeletonize(train_sup2_pred_sample, method='lee')

                        # structure
                        train_sup1_skl_sample_ = cv2.dilate(train_sup1_skl_sample, structure_skl_kernel)
                        train_sup1_dis_sample = ndimage.distance_transform_edt(train_sup1_pred_sample)
                        train_sup2_skl_sample_ = cv2.dilate(train_sup2_skl_sample, structure_skl_kernel)
                        train_sup2_dis_sample = ndimage.distance_transform_edt(train_sup2_pred_sample)

                        structure_map_list1 = []
                        structure_map_list2 = []
                        diff_thr_before1 = np.zeros(train_sup1_pred_sample.shape)
                        diff_thr_before2 = np.zeros(train_sup2_pred_sample.shape)
                        max_dis1 = train_sup1_dis_sample.max()
                        max_dis2 = train_sup2_dis_sample.max()
                        interval1 = (max_dis1 - 1) / (args.structure_level - 1)
                        interval2 = (max_dis2 - 1) / (args.structure_level - 1)

                        for level in range(args.structure_level):

                            thr1 = 1.0 + level * interval1
                            dis_thr1 = train_sup1_dis_sample.copy()
                            dis_thr1[train_sup1_dis_sample <= thr1] = 0
                            pred_thr1 = train_sup1_pred_sample.copy()
                            pred_thr1[dis_thr1 == 0] = 0

                            thr2 = 1.0 + level * interval2
                            dis_thr2 = train_sup2_dis_sample.copy()
                            dis_thr2[train_sup2_dis_sample <= thr2] = 0
                            pred_thr2 = train_sup2_pred_sample.copy()
                            pred_thr2[dis_thr2 == 0] = 0

                            skl_thr1 = skeletonize(pred_thr1, method='lee')
                            skl_thr1_ = cv2.dilate(skl_thr1, structure_skl_kernel)
                            skl_thr2 = skeletonize(pred_thr2, method='lee')
                            skl_thr2_ = cv2.dilate(skl_thr2, structure_skl_kernel)

                            diff_thr1 = train_sup1_skl_sample.astype(np.float32) - skl_thr1.astype(np.float32)
                            diff_thr1[diff_thr1 == -1] = 1
                            diff_thr1[(train_sup1_skl_sample_ == 1) & (skl_thr1_ == 1)] = 0
                            diff_thr1 = diff_thr1 - diff_thr_before1
                            diff_thr_before1 = diff_thr1

                            diff_thr2 = train_sup2_skl_sample.astype(np.float32) - skl_thr2.astype(np.float32)
                            diff_thr2[diff_thr2 == -1] = 1
                            diff_thr2[(train_sup2_skl_sample_ == 1) & (skl_thr2_ == 1)] = 0
                            diff_thr2 = diff_thr2 - diff_thr_before2
                            diff_thr_before2 = diff_thr2

                            structure_map1 = np.zeros(train_sup1_pred_sample.shape)
                            x, y = np.nonzero(diff_thr1)
                            for k in range(len(x)):
                                weight_range = math.ceil(args.structure_range * train_sup1_dis_sample[x[k], y[k]])
                                structure_map1[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
                            structure_map_list1.append(structure_map1)

                            structure_map2 = np.zeros(train_sup2_pred_sample.shape)
                            x, y = np.nonzero(diff_thr2)
                            for k in range(len(x)):
                                weight_range = math.ceil(args.structure_range * train_sup2_dis_sample[x[k], y[k]])
                                structure_map2[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
                            structure_map_list2.append(structure_map2)

                        structure_map1_ = np.zeros(train_sup1_pred_sample.shape)
                        structure_map_list1.reverse()
                        for j in range(len(structure_map_list1)):
                            if j <= args.structure_thr:
                                structure_map1_[structure_map_list1[j] == 1] = 1
                            else:
                                structure_map1_[structure_map_list1[j] == 1] = j + 1 - args.structure_thr

                        structure_map2_ = np.zeros(train_sup2_pred_sample.shape)
                        structure_map_list2.reverse()
                        for j in range(len(structure_map_list2)):
                            if j <= args.structure_thr:
                                structure_map2_[structure_map_list2[j] == 1] = 1
                            else:
                                structure_map2_[structure_map_list2[j] == 1] = j + 1 - args.structure_thr

                        structure_map1_[structure_map1_ == 0] = 1
                        structure_map2_[structure_map2_ == 0] = 1
                        structure1_list_sup.append(structure_map1_)
                        structure2_list_sup.append(structure_map2_)

                        # semantic
                        train_sup1_skl_sample_ = cv2.dilate(train_sup1_skl_sample, semantic_skl_kernel)
                        train_sup2_skl_sample_ = cv2.dilate(train_sup2_skl_sample, semantic_skl_kernel)

                        diff_sample = train_sup1_skl_sample.astype(np.float32) - train_sup2_skl_sample.astype(np.float32)
                        diff_sample[diff_sample == -1] = 1
                        diff_sample[(train_sup1_skl_sample_ == 1) & (train_sup2_skl_sample_ == 1)] = 0

                        semantic_map = cv2.dilate(diff_sample.astype(np.uint8), semantic_range_kernel)
                        semantic_map[semantic_map == 1] = args.semantic_level
                        semantic_map[semantic_map == 0] = 1

                        semantic_list_sup.append(semantic_map)
                        # attention
                        attention_map = np.stack([semantic_map, structure_map1_, structure_map2_], axis=1)
                        attention_map = np.max(attention_map, 1)
                        # attention_map = cv2.GaussianBlur(attention_map, (9, 9), 0, 0)
                        attention_list_sup.append(attention_map)
                    torch.cuda.empty_cache()
                    # update method
                    if args.update_method == 'new':
                        structure1_train_unsup = torch.from_numpy(np.array(structure1_list_unsup)).float()
                        structure2_train_unsup = torch.from_numpy(np.array(structure2_list_unsup)).float()
                        semantic_train_unsup = torch.from_numpy(np.array(semantic_list_unsup)).float()
                        attention_train_unsup = torch.from_numpy(np.array(attention_list_unsup)).float()

                        structure1_train_sup = torch.from_numpy(np.array(structure1_list_sup)).float()
                        structure2_train_sup = torch.from_numpy(np.array(structure2_list_sup)).float()
                        semantic_train_sup = torch.from_numpy(np.array(semantic_list_sup)).float()
                        attention_train_sup = torch.from_numpy(np.array(attention_list_sup)).float()
                    elif args.update_method == 'both':
                        structure1_train_unsup = args.attention_weight * unsup_index['structure1'] + (1 - args.attention_weight) * torch.from_numpy(np.array(structure1_list_unsup)).float()
                        structure2_train_unsup = args.attention_weight * unsup_index['structure2'] + (1 - args.attention_weight) * torch.from_numpy(np.array(structure2_list_unsup)).float()
                        semantic_train_unsup = args.attention_weight * unsup_index['semantic'] + (1 - args.attention_weight) * torch.from_numpy(np.array(semantic_list_unsup)).float()
                        attention_train_unsup = args.attention_weight * unsup_index['attention'] + (1 - args.attention_weight) * torch.from_numpy(np.array(attention_list_unsup)).float()

                        structure1_train_sup = args.attention_weight * sup_index['structure1'] + (1-args.attention_weight) * torch.from_numpy(np.array(structure1_list_sup)).float()
                        structure2_train_sup = args.attention_weight * sup_index['structure2'] + (1-args.attention_weight) * torch.from_numpy(np.array(structure2_list_sup)).float()
                        semantic_train_sup = args.attention_weight * sup_index['semantic'] + (1-args.attention_weight) * torch.from_numpy(np.array(semantic_list_sup)).float()
                        attention_train_sup = args.attention_weight * sup_index['attention'] + (1-args.attention_weight) * torch.from_numpy(np.array(attention_list_sup)).float()
                    else:
                        structure1_train_unsup = unsup_index['structure1']
                        structure2_train_unsup = unsup_index['structure2']
                        semantic_train_unsup = unsup_index['semantic']
                        attention_train_unsup = unsup_index['attention']

                        structure1_train_sup = sup_index['structure1']
                        structure2_train_sup = sup_index['structure2']
                        semantic_train_sup = sup_index['semantic']
                        attention_train_sup = sup_index['attention']

                    structure1_train_unsup = Variable(structure1_train_unsup.cuda())
                    structure2_train_unsup = Variable(structure2_train_unsup.cuda())
                    semantic_train_unsup = Variable(semantic_train_unsup.cuda())
                    attention_train_unsup = Variable(attention_train_unsup.cuda())

                    structure1_train_sup = Variable(structure1_train_sup.cuda())
                    structure2_train_sup = Variable(structure2_train_sup.cuda())
                    semantic_train_sup = Variable(semantic_train_sup.cuda())
                    attention_train_sup = Variable(attention_train_sup.cuda())

            model1.train()
            model2.train()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            pred_train_unsup1 = model1(img_train_unsup, structure1_train_unsup.unsqueeze(1), structure2_train_unsup.unsqueeze(1), semantic_train_unsup.unsqueeze(1))
            pred_train_unsup2 = model2(img_train_unsup, structure1_train_unsup.unsqueeze(1), structure2_train_unsup.unsqueeze(1), semantic_train_unsup.unsqueeze(1))

            max_train1 = torch.max(pred_train_unsup1, dim=1)[1]
            max_train2 = torch.max(pred_train_unsup2, dim=1)[1]

            loss_train_unsup1 = args.loss_weight * criterion(pred_train_unsup1, max_train2.long()) + args.attention_loss_weight * criterion_attention(pred_train_unsup1, max_train2.long(), attention_train_unsup)
            loss_train_unsup2 = args.loss_weight * criterion(pred_train_unsup2, max_train1.long()) + args.attention_loss_weight * criterion_attention(pred_train_unsup2, max_train1.long(), attention_train_unsup)
            loss_train_unsup = (loss_train_unsup1 + loss_train_unsup2) * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            pred_train_sup1 = model1(img_train_sup, structure1_train_sup.unsqueeze(1), structure2_train_sup.unsqueeze(1), semantic_train_sup.unsqueeze(1))
            pred_train_sup2 = model2(img_train_sup, structure1_train_sup.unsqueeze(1), structure2_train_sup.unsqueeze(1), semantic_train_sup.unsqueeze(1))

            loss_train_sup1 = args.loss_weight * criterion(pred_train_sup1, mask_train_sup) + args.attention_loss_weight * criterion_attention(pred_train_sup1, mask_train_sup, attention_train_sup)
            loss_train_sup2 = args.loss_weight * criterion(pred_train_sup2, mask_train_sup) + args.attention_loss_weight * criterion_attention(pred_train_sup2, mask_train_sup, attention_train_sup)

            loss_train_sup = loss_train_sup1 + loss_train_sup2
            loss_train_sup.backward()

            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss += loss_train.item()

            if count_iter % args.display_iter == 0:
                pred_train1 = torch.max(pred_train_sup1, dim=1)[1]
                pred_train2 = torch.max(pred_train_sup2, dim=1)[1]
                if i == 0:
                    pred_list_train1 = pred_train1
                    pred_list_train2 = pred_train2
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / cfg['TRAIN_RATIO']:
                    pred_list_train1 = torch.cat((pred_list_train1, pred_train1), dim=0)
                    pred_list_train2 = torch.cat((pred_list_train2, pred_train2), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

        exp_lr_scheduler1.step()
        exp_lr_scheduler2.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            pred_gather_list_train1 = [torch.zeros_like(pred_list_train1) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(pred_gather_list_train1, pred_list_train1)
            pred_list_train1 = torch.cat(pred_gather_list_train1, dim=0)
            pred_list_train1 = pred_list_train1.data.cpu().detach().numpy()

            pred_gather_list_train2 = [torch.zeros_like(pred_list_train2) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(pred_gather_list_train2, pred_list_train2)
            pred_list_train2 = torch.cat(pred_gather_list_train2, dim=0)
            pred_list_train2 = pred_list_train2.data.cpu().detach().numpy()

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)
            mask_list_train = mask_list_train.data.cpu().numpy()

            if rank == args.rank_index:
                torch.cuda.empty_cache()
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_TA(train_loss_sup_1, train_loss_sup_2, train_loss_unsup, train_loss, num_batches, print_num, print_num_half)
                train_eval_list1, train_eval_list2 = print_train_eval(pred_list_train1, pred_list_train2, mask_list_train, print_num_half)
                torch.cuda.empty_cache()

            with torch.no_grad():
                model1.eval()
                model2.eval()

                for i, data in enumerate(dataloaders['val']):

                    if 0 <= i <= num_batches['val'] / cfg['VAL_RATIO']:
                        inputs_val = Variable(data['image'].cuda())
                        structure1_val = Variable(data['structure1'].cuda())
                        structure2_val = Variable(data['structure2'].cuda())
                        semantic_val = Variable(data['semantic'].cuda())
                        attention_val = Variable(data['attention'].cuda())
                        mask_val = Variable(data['mask'].cuda())
                        name_val = data['ID']

                        optimizer1.zero_grad()
                        optimizer2.zero_grad()

                        outputs_val1 = model1(inputs_val, structure1_val.unsqueeze(1), structure2_val.unsqueeze(1), semantic_val.unsqueeze(1))
                        outputs_val2 = model2(inputs_val, structure1_val.unsqueeze(1), structure2_val.unsqueeze(1), semantic_val.unsqueeze(1))
                        torch.cuda.empty_cache()

                        pred_val1 = torch.max(outputs_val1, dim=1)[1]
                        pred_val2 = torch.max(outputs_val2, dim=1)[1]

                        pred_val1_np = pred_val1.data.cpu().numpy().astype(np.uint8)
                        pred_val2_np = pred_val2.data.cpu().numpy().astype(np.uint8)

                        structure1_list_val = []
                        structure2_list_val = []
                        semantic_list_val = []
                        attention_list_val = []

                        for sample in range(pred_val1_np.shape[0]):
                            val1_pred_sample = pred_val1_np[sample, :, :]
                            val2_pred_sample = pred_val2_np[sample, :, :]

                            val1_skl_sample = skeletonize(val1_pred_sample, method='lee')
                            val2_skl_sample = skeletonize(val2_pred_sample, method='lee')

                            # structure
                            val1_skl_sample_ = cv2.dilate(val1_skl_sample, structure_skl_kernel)
                            val1_dis_sample = ndimage.distance_transform_edt(val1_pred_sample)
                            val2_skl_sample_ = cv2.dilate(val2_skl_sample, structure_skl_kernel)
                            val2_dis_sample = ndimage.distance_transform_edt(val2_pred_sample)

                            structure_map_list1 = []
                            structure_map_list2 = []
                            diff_thr_before1 = np.zeros(val1_pred_sample.shape)
                            diff_thr_before2 = np.zeros(val2_pred_sample.shape)
                            max_dis1 = val1_dis_sample.max()
                            max_dis2 = val2_dis_sample.max()
                            interval1 = (max_dis1 - 1) / (args.structure_level - 1)
                            interval2 = (max_dis2 - 1) / (args.structure_level - 1)

                            for level in range(args.structure_level):

                                thr1 = 1.0 + level * interval1
                                dis_thr1 = val1_dis_sample.copy()
                                dis_thr1[val1_dis_sample <= thr1] = 0
                                pred_thr1 = val1_pred_sample.copy()
                                pred_thr1[dis_thr1 == 0] = 0

                                thr2 = 1.0 + level * interval2
                                dis_thr2 = val2_dis_sample.copy()
                                dis_thr2[val2_dis_sample <= thr2] = 0
                                pred_thr2 = val2_pred_sample.copy()
                                pred_thr2[dis_thr2 == 0] = 0

                                skl_thr1 = skeletonize(pred_thr1, method='lee')
                                skl_thr1_ = cv2.dilate(skl_thr1, structure_skl_kernel)
                                skl_thr2 = skeletonize(pred_thr2, method='lee')
                                skl_thr2_ = cv2.dilate(skl_thr2, structure_skl_kernel)

                                diff_thr1 = val1_skl_sample.astype(np.float32) - skl_thr1.astype(np.float32)
                                diff_thr1[diff_thr1 == -1] = 1
                                diff_thr1[(val1_skl_sample_ == 1) & (skl_thr1_ == 1)] = 0
                                diff_thr1 = diff_thr1 - diff_thr_before1
                                diff_thr_before1 = diff_thr1

                                diff_thr2 = val2_skl_sample.astype(np.float32) - skl_thr2.astype(np.float32)
                                diff_thr2[diff_thr2 == -1] = 1
                                diff_thr2[(val2_skl_sample_ == 1) & (skl_thr2_ == 1)] = 0
                                diff_thr2 = diff_thr2 - diff_thr_before2
                                diff_thr_before2 = diff_thr2

                                structure_map1 = np.zeros(val1_pred_sample.shape)
                                x, y = np.nonzero(diff_thr1)
                                for k in range(len(x)):
                                    weight_range = math.ceil(args.structure_range * val1_dis_sample[x[k], y[k]])
                                    structure_map1[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
                                structure_map_list1.append(structure_map1)

                                structure_map2 = np.zeros(val2_pred_sample.shape)
                                x, y = np.nonzero(diff_thr2)
                                for k in range(len(x)):
                                    weight_range = math.ceil(args.structure_range * val2_dis_sample[x[k], y[k]])
                                    structure_map2[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
                                structure_map_list2.append(structure_map2)

                            structure_map1_ = np.zeros(val1_pred_sample.shape)
                            structure_map_list1.reverse()
                            for j in range(len(structure_map_list1)):
                                if j <= args.structure_thr:
                                    structure_map1_[structure_map_list1[j] == 1] = 1
                                else:
                                    structure_map1_[structure_map_list1[j] == 1] = j + 1 - args.structure_thr

                            structure_map2_ = np.zeros(val2_pred_sample.shape)
                            structure_map_list2.reverse()
                            for j in range(len(structure_map_list2)):
                                if j <= args.structure_thr:
                                    structure_map2_[structure_map_list2[j] == 1] = 1
                                else:
                                    structure_map2_[structure_map_list2[j] == 1] = j + 1 - args.structure_thr

                            structure_map1_[structure_map1_ == 0] = 1
                            structure_map2_[structure_map2_ == 0] = 1
                            structure1_list_val.append(structure_map1_)
                            structure2_list_val.append(structure_map2_)

                            # semantic
                            val1_skl_sample_ = cv2.dilate(val1_skl_sample, semantic_skl_kernel)
                            val2_skl_sample_ = cv2.dilate(val2_skl_sample, semantic_skl_kernel)

                            diff_sample = val1_skl_sample.astype(np.float32) - val2_skl_sample.astype(np.float32)
                            diff_sample[diff_sample == -1] = 1
                            diff_sample[(val1_skl_sample_ == 1) & (val2_skl_sample_ == 1)] = 0

                            semantic_map = cv2.dilate(diff_sample.astype(np.uint8), semantic_range_kernel)
                            semantic_map[semantic_map == 1] = args.semantic_level
                            semantic_map[semantic_map == 0] = 1

                            semantic_list_val.append(semantic_map)
                            # attention
                            attention_map = np.stack([semantic_map, structure_map1_, structure_map2_], axis=1)
                            attention_map = np.max(attention_map, 1)
                            # attention_map = cv2.GaussianBlur(attention_map, (9, 9), 0, 0)
                            attention_list_val.append(attention_map)
                        torch.cuda.empty_cache()

                        # update method
                        if args.update_method == 'new':
                            structure1_val = torch.from_numpy(np.array(structure1_list_val)).float()
                            structure2_val = torch.from_numpy(np.array(structure2_list_val)).float()
                            semantic_val = torch.from_numpy(np.array(semantic_list_val)).float()
                            attention_val = torch.from_numpy(np.array(attention_list_val)).float()
                        elif args.update_method == 'both':
                            structure1_val = args.attention_weight * data['structure1'] + (1-args.attention_weight) * torch.from_numpy(np.array(structure1_list_val)).float()
                            structure2_val = args.attention_weight * data['structure2'] + (1-args.attention_weight) * torch.from_numpy(np.array(structure2_list_val)).float()
                            semantic_val = args.attention_weight * data['semantic'] + (1-args.attention_weight) * torch.from_numpy(np.array(semantic_list_val)).float()
                            attention_val = args.attention_weight * data['attention'] + (1-args.attention_weight) * torch.from_numpy(np.array(attention_list_val)).float()
                        else:
                            structure1_val = data['structure1']
                            structure2_val = data['structure2']
                            semantic_val = data['semantic']
                            attention_val = data['attention']

                        structure1_val = Variable(structure1_val.cuda())
                        structure2_val = Variable(structure2_val.cuda())
                        semantic_val = Variable(semantic_val.cuda())
                        attention_val = Variable(attention_val.cuda())

                        optimizer1.zero_grad()
                        optimizer2.zero_grad()

                        outputs_val1 = model1(inputs_val, structure1_val.unsqueeze(1), structure2_val.unsqueeze(1), semantic_val.unsqueeze(1))
                        outputs_val2 = model2(inputs_val, structure1_val.unsqueeze(1), structure2_val.unsqueeze(1), semantic_val.unsqueeze(1))

                        loss_val_sup1 = args.loss_weight * criterion(outputs_val1, mask_val) + args.attention_loss_weight * criterion_attention(outputs_val1, mask_val, attention_val)
                        loss_val_sup2 = args.loss_weight * criterion(outputs_val2, mask_val) + args.attention_loss_weight * criterion_attention(outputs_val2, mask_val, attention_val)
                        val_loss_sup_1 += loss_val_sup1.item()
                        val_loss_sup_2 += loss_val_sup2.item()

                        pred_val1 = torch.max(outputs_val1, dim=1)[1]
                        pred_val2 = torch.max(outputs_val2, dim=1)[1]

                        if i == 0:
                            pred_list_val1 = pred_val1
                            pred_list_val2 = pred_val2
                            mask_list_val = mask_val
                            name_list_val = name_val
                        else:
                            pred_list_val1 = torch.cat((pred_list_val1, pred_val1), dim=0)
                            pred_list_val2 = torch.cat((pred_list_val2, pred_val2), dim=0)
                            mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                            name_list_val = np.append(name_list_val, name_val, axis=0)

                torch.cuda.empty_cache()
                pred_gather_list_val1 = [torch.zeros_like(pred_list_val1) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(pred_gather_list_val1, pred_list_val1)
                pred_list_val1 = torch.cat(pred_gather_list_val1, dim=0)
                pred_list_val1 = pred_list_val1.data.cpu().detach().numpy()

                pred_gather_list_val2 = [torch.zeros_like(pred_list_val2) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(pred_gather_list_val2, pred_list_val2)
                pred_list_val2 = torch.cat(pred_gather_list_val2, dim=0)
                pred_list_val2 = pred_list_val2.data.cpu().detach().numpy()

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)
                mask_list_val = mask_list_val.data.cpu().numpy()

                name_gather_list_val = [None for _ in range(ngpus_per_node)]
                torch.distributed.all_gather_object(name_gather_list_val, name_list_val)
                name_list_val = np.concatenate(name_gather_list_val, axis=0)

                if rank == args.rank_index:
                    val_epoch_loss_sup1, val_epoch_loss_sup2 = print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half)
                    val_eval_list1, val_eval_list2 = print_val_eval(pred_list_val1, pred_list_val2, mask_list_val, print_num_half)
                    best_val_eval_list, best_model, best_result = save_val_best_TA(best_model, best_val_eval_list, best_result, model1, model2, pred_list_val1, pred_list_val2, name_list_val, val_eval_list1, val_eval_list2, path_trained_models, path_seg_results, cfg['PALETTE'])
                    torch.cuda.empty_cache()

                    if args.vis:
                        draw_img = draw_pred_TA(mask_list_train, mask_list_val, pred_list_train1, pred_list_train2, pred_list_val1, pred_list_val2)
                        visualization_TA(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_unsup, train_eval_list1[0], train_eval_list2[0], train_eval_list1[2], train_eval_list2[2], val_epoch_loss_sup1, val_epoch_loss_sup2, val_eval_list1[0], val_eval_list2[0], val_eval_list1[2], val_eval_list2[2])
                        visual_image_TA(visdom, draw_img[0], draw_img[1], draw_img[2], draw_img[3], draw_img[4], draw_img[5])

                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)

        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        print_best(best_val_eval_list, best_model, best_result, path_trained_models, print_num_minus)
        print('=' * print_num)

