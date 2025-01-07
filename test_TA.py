from torchvision import transforms, datasets
import torch
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
from models.getnetwork import get_network
from dataload.dataset import imagefloder_TA
from config.train_test_config.train_test_config import print_test_eval, save_test
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
    parser.add_argument('--path_dataset', default='/mnt/data1/Semi-Topology-Attention/dataset/CBMI_ER')
    parser.add_argument('--path_model1', default='/mnt/data1/Semi-Topology-Attention/pretrained_model/CBMI_ER/best_Result1_BetaE_15.4500.pth')
    parser.add_argument('--path_model2', default='/mnt/data1/Semi-Topology-Attention/pretrained_model/CBMI_ER/best_Result1_BetaE_15.4500_other.pth')
    parser.add_argument('--path_seg_results', default='/mnt/data1/Semi-Topology-Attention/seg_pred/val')
    parser.add_argument('--dataset_name', default='CBMI_ER', help='CREMI, SNEMI3D, CBMI_ER')
    parser.add_argument('--if_mask', default=True)
    parser.add_argument('--result', default='model1', help='model1, model2')

    # attention
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

    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-n', '--network', default='unet_ta_v1')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Results Save
    if not os.path.exists(args.path_seg_results) and rank == args.rank_index:
        os.mkdir(args.path_seg_results)
    path_seg_results = args.path_seg_results + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    if args.result == 'model1':
        path_seg_results = path_seg_results + '/' + str(os.path.splitext(os.path.split(args.path_model1)[1])[0])
    elif args.result == 'model2':
        path_seg_results = path_seg_results + '/' + str(os.path.splitext(os.path.split(args.path_model2)[1])[0])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    # Dataset
    data_transforms = data_transform(cfg['SIZE'], cfg['MEAN'], cfg['STD'])

    dataset_val = imagefloder_TA(
        data_dir=args.path_dataset + '/' + os.path.split(args.path_seg_results)[1],
        data_transform=data_transforms['val'],
        sup=args.if_mask,
        num_images=None,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=val_sampler)
    num_batches = {'val': len(dataloaders['val'])}

    # Model
    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model2 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    model1 = model1.cuda()
    model2 = model2.cuda()

    model1 = DistributedDataParallel(model1, device_ids=[args.local_rank])
    state_dict1 = torch.load(args.path_model1)
    model1.load_state_dict(state_dict=state_dict1)
    model2 = DistributedDataParallel(model2, device_ids=[args.local_rank])
    state_dict2 = torch.load(args.path_model2)
    model2.load_state_dict(state_dict=state_dict2)
    dist.barrier()

    # Test
    since = time.time()
    structure_skl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.structure_skl_kernel_size, args.structure_skl_kernel_size))
    semantic_skl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.semantic_skl_kernel_size, args.semantic_skl_kernel_size))
    semantic_range_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.semantic_range, args.semantic_range))

    with torch.no_grad():
        model1.eval()
        model2.eval()

        for i, data in enumerate(dataloaders['val']):
            image_test = Variable(data['image'].cuda())
            structure1_test = Variable(data['structure1'].cuda())
            structure2_test = Variable(data['structure2'].cuda())
            semantic_test = Variable(data['semantic'].cuda())
            attention_test = Variable(data['attention'].cuda())
            name_test = data['ID']
            if args.if_mask:
                mask_test = Variable(data['mask'].cuda())

            outputs_test1 = model1(image_test, structure1_test.unsqueeze(1), structure2_test.unsqueeze(1), semantic_test.unsqueeze(1))
            outputs_test2 = model2(image_test, structure1_test.unsqueeze(1), structure2_test.unsqueeze(1), semantic_test.unsqueeze(1))

            pred_test1 = torch.max(outputs_test1, dim=1)[1]
            pred_test2 = torch.max(outputs_test2, dim=1)[1]

            pred_test1_np = pred_test1.data.cpu().numpy().astype(np.uint8)
            pred_test2_np = pred_test2.data.cpu().numpy().astype(np.uint8)

            structure1_list_test = []
            structure2_list_test = []
            semantic_list_test = []
            attention_list_test = []

            for sample in range(pred_test1_np.shape[0]):
                test1_pred_sample = pred_test1_np[sample, :, :]
                test2_pred_sample = pred_test2_np[sample, :, :]

                test1_skl_sample = skeletonize(test1_pred_sample, method='lee')
                test2_skl_sample = skeletonize(test2_pred_sample, method='lee')

                # structure
                test1_skl_sample_ = cv2.dilate(test1_skl_sample, structure_skl_kernel)
                test1_dis_sample = ndimage.distance_transform_edt(test1_pred_sample)
                test2_skl_sample_ = cv2.dilate(test2_skl_sample, structure_skl_kernel)
                test2_dis_sample = ndimage.distance_transform_edt(test2_pred_sample)

                structure_map_list1 = []
                structure_map_list2 = []
                diff_thr_before1 = np.zeros(test1_pred_sample.shape)
                diff_thr_before2 = np.zeros(test2_pred_sample.shape)
                max_dis1 = test1_dis_sample.max()
                max_dis2 = test2_dis_sample.max()
                interval1 = (max_dis1 - 1) / (args.structure_level - 1)
                interval2 = (max_dis2 - 1) / (args.structure_level - 1)

                for level in range(args.structure_level):

                    thr1 = 1.0 + level * interval1
                    dis_thr1 = test1_dis_sample.copy()
                    dis_thr1[test1_dis_sample <= thr1] = 0
                    pred_thr1 = test1_pred_sample.copy()
                    pred_thr1[dis_thr1 == 0] = 0

                    thr2 = 1.0 + level * interval2
                    dis_thr2 = test2_dis_sample.copy()
                    dis_thr2[test2_dis_sample <= thr2] = 0
                    pred_thr2 = test2_pred_sample.copy()
                    pred_thr2[dis_thr2 == 0] = 0

                    skl_thr1 = skeletonize(pred_thr1, method='lee')
                    skl_thr1_ = cv2.dilate(skl_thr1, structure_skl_kernel)
                    skl_thr2 = skeletonize(pred_thr2, method='lee')
                    skl_thr2_ = cv2.dilate(skl_thr2, structure_skl_kernel)

                    diff_thr1 = test1_skl_sample.astype(np.float32) - skl_thr1.astype(np.float32)
                    diff_thr1[diff_thr1 == -1] = 1
                    diff_thr1[(test1_skl_sample_ == 1) & (skl_thr1_ == 1)] = 0
                    diff_thr1 = diff_thr1 - diff_thr_before1
                    diff_thr_before1 = diff_thr1

                    diff_thr2 = test2_skl_sample.astype(np.float32) - skl_thr2.astype(np.float32)
                    diff_thr2[diff_thr2 == -1] = 1
                    diff_thr2[(test2_skl_sample_ == 1) & (skl_thr2_ == 1)] = 0
                    diff_thr2 = diff_thr2 - diff_thr_before2
                    diff_thr_before2 = diff_thr2

                    structure_map1 = np.zeros(test1_pred_sample.shape)
                    x, y = np.nonzero(diff_thr1)
                    for k in range(len(x)):
                        weight_range = math.ceil(args.structure_range * test1_dis_sample[x[k], y[k]])
                        structure_map1[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
                    structure_map_list1.append(structure_map1)

                    structure_map2 = np.zeros(test2_pred_sample.shape)
                    x, y = np.nonzero(diff_thr2)
                    for k in range(len(x)):
                        weight_range = math.ceil(args.structure_range * test2_dis_sample[x[k], y[k]])
                        structure_map2[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
                    structure_map_list2.append(structure_map2)

                structure_map1_ = np.zeros(test1_pred_sample.shape)
                structure_map_list1.reverse()
                for j in range(len(structure_map_list1)):
                    if j <= args.structure_thr:
                        structure_map1_[structure_map_list1[j] == 1] = 1
                    else:
                        structure_map1_[structure_map_list1[j] == 1] = j + 1 - args.structure_thr

                structure_map2_ = np.zeros(test2_pred_sample.shape)
                structure_map_list2.reverse()
                for j in range(len(structure_map_list2)):
                    if j <= args.structure_thr:
                        structure_map2_[structure_map_list2[j] == 1] = 1
                    else:
                        structure_map2_[structure_map_list2[j] == 1] = j + 1 - args.structure_thr

                structure_map1_[structure_map1_ == 0] = 1
                structure_map2_[structure_map2_ == 0] = 1
                structure1_list_test.append(structure_map1_)
                structure2_list_test.append(structure_map2_)

                # semantic
                test1_skl_sample_ = cv2.dilate(test1_skl_sample, semantic_skl_kernel)
                test2_skl_sample_ = cv2.dilate(test2_skl_sample, semantic_skl_kernel)

                diff_sample = test1_skl_sample.astype(np.float32) - test2_skl_sample.astype(np.float32)
                diff_sample[diff_sample == -1] = 1
                diff_sample[(test1_skl_sample_ == 1) & (test2_skl_sample_ == 1)] = 0

                semantic_map = cv2.dilate(diff_sample.astype(np.uint8), semantic_range_kernel)
                semantic_map[semantic_map == 1] = args.semantic_level
                semantic_map[semantic_map == 0] = 1

                semantic_list_test.append(semantic_map)
                # attention
                attention_map = np.stack([semantic_map, structure_map1_, structure_map2_], axis=1)
                attention_map = np.max(attention_map, 1)
                # attention_map = cv2.GaussianBlur(attention_map, (9, 9), 0, 0)
                attention_list_test.append(attention_map)
            torch.cuda.empty_cache()

            # update method
            if args.update_method == 'new':
                structure1_test = torch.from_numpy(np.array(structure1_list_test)).float()
                structure2_test = torch.from_numpy(np.array(structure2_list_test)).float()
                semantic_test = torch.from_numpy(np.array(semantic_list_test)).float()
                attention_test = torch.from_numpy(np.array(attention_list_test)).float()
            elif args.update_method == 'both':
                structure1_test = args.attention_weight * data['structure1'] + (1 - args.attention_weight) * torch.from_numpy(np.array(structure1_list_test)).float()
                structure2_test = args.attention_weight * data['structure2'] + (1 - args.attention_weight) * torch.from_numpy(np.array(structure2_list_test)).float()
                semantic_test = args.attention_weight * data['semantic'] + (1 - args.attention_weight) * torch.from_numpy(np.array(semantic_list_test)).float()
                attention_test = args.attention_weight * data['attention'] + (1 - args.attention_weight) * torch.from_numpy(np.array(attention_list_test)).float()
            else:
                structure1_test = data['structure1']
                structure2_test = data['structure2']
                semantic_test = data['semantic']
                attention_test = data['attention']

            structure1_test = Variable(structure1_test.cuda())
            structure2_test = Variable(structure2_test.cuda())
            semantic_test = Variable(semantic_test.cuda())
            attention_test = Variable(attention_test.cuda())

            if args.result == 'model1':
                outputs_test = model1(image_test, structure1_test.unsqueeze(1), structure2_test.unsqueeze(1), semantic_test.unsqueeze(1))
            elif args.result == 'model2':
                outputs_test = model2(image_test, structure1_test.unsqueeze(1), structure2_test.unsqueeze(1), semantic_test.unsqueeze(1))

            pred_test = torch.max(outputs_test, dim=1)[1]
            if args.if_mask:
                if i == 0:
                    pred_list_test = pred_test
                    name_list_test = name_test
                    mask_list_test = mask_test
                # else:
                elif 0 < i <= num_batches['val'] / cfg['VAL_RATIO']:
                    pred_list_test = torch.cat((pred_list_test, pred_test), dim=0)
                    name_list_test = np.append(name_list_test, name_test, axis=0)
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
                torch.cuda.empty_cache()
            else:
                save_test(pred_test.data.cpu().detach().numpy(), name_test, path_seg_results, cfg['PALETTE'])
                torch.cuda.empty_cache()

        if args.if_mask:
            pred_gather_list_test = [torch.zeros_like(pred_list_test) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(pred_gather_list_test, pred_list_test)
            pred_list_test = torch.cat(pred_gather_list_test, dim=0)
            pred_list_test = pred_list_test.data.cpu().detach().numpy()

            mask_gather_list_test = [torch.zeros_like(mask_list_test) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_test, mask_list_test)
            mask_list_test = torch.cat(mask_gather_list_test, dim=0)
            mask_list_test = mask_list_test.data.cpu().numpy()

            name_gather_list_test = [None for _ in range(ngpus_per_node)]
            torch.distributed.all_gather_object(name_gather_list_test, name_list_test)
            name_list_test = np.concatenate(name_gather_list_test, axis=0)

        if args.if_mask and rank == args.rank_index:
            print('=' * print_num)
            test_eval_list = print_test_eval(pred_list_test, mask_list_test, print_num_minus)
            save_test(pred_list_test, name_list_test, path_seg_results, cfg['PALETTE'])
            torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)