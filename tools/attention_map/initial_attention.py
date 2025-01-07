import numpy as np
import os
import argparse
from skimage.morphology import skeletonize
from scipy import ndimage
from PIL import Image
import cv2
import math
from config.dataset_config.dataset_cfg import dataset_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred1_path', default='D:/Desktop/attention map figure/pred1')
    parser.add_argument('--pred2_path', default='D:/Desktop/attention map figure/pred2')
    # structure
    parser.add_argument('--structure_skl_kernel_size', default=5, type=int, help='5')
    parser.add_argument('--structure_level', default=10, type=int, help='10')
    parser.add_argument('--structure_range', default=1.5, type=float, help='1.5, 1.1')
    parser.add_argument('--structure_thr', default=1, type=int, help='3')
    # semantic
    parser.add_argument('--semantic_skl_kernel_size', default=5, type=int, help='5')
    parser.add_argument('--semantic_level', default=4, type=int, help='4')
    parser.add_argument('--semantic_range', default=15, type=int, help='10')

    parser.add_argument('--save_path_attention', default='D:/Desktop/attention map figure/attention')
    parser.add_argument('--save_path_semantic', default='D:/Desktop/attention map figure/semantic')
    parser.add_argument('--save_path_structure1', default='D:/Desktop/attention map figure/structure1')
    parser.add_argument('--save_path_structure2', default='D:/Desktop/attention map figure/structure2')
    args = parser.parse_args()

    if not os.path.exists(args.save_path_attention):
        os.mkdir(args.save_path_attention)
    if not os.path.exists(args.save_path_semantic):
        os.mkdir(args.save_path_semantic)
    if not os.path.exists(args.save_path_structure1):
        os.mkdir(args.save_path_structure1)
    if not os.path.exists(args.save_path_structure2):
        os.mkdir(args.save_path_structure2)

    structure_skl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.structure_skl_kernel_size, args.structure_skl_kernel_size))
    semantic_skl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.semantic_skl_kernel_size, args.semantic_skl_kernel_size))
    semantic_range_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.semantic_range, args.semantic_range))

    for img in os.listdir(args.pred1_path):
        pred1_path = os.path.join(args.pred1_path, img)
        pred2_path = os.path.join(args.pred2_path, img)
        save_path_semantic = os.path.join(args.save_path_semantic, img)
        save_path_structure1 = os.path.join(args.save_path_structure1, img)
        save_path_structure2 = os.path.join(args.save_path_structure2, img)
        save_path_attention = os.path.join(args.save_path_attention, img)

        pred1 = Image.open(pred1_path)
        pred1 = np.array(pred1)
        pred2 = Image.open(pred2_path)
        pred2 = np.array(pred2)

        pred1_skl = skeletonize(pred1, method='lee')
        pred2_skl = skeletonize(pred2, method='lee')

        # structure
        pred1_skl_ = cv2.dilate(pred1_skl, structure_skl_kernel)
        pred1_dis = ndimage.distance_transform_edt(pred1)
        pred2_skl_ = cv2.dilate(pred2_skl, structure_skl_kernel)
        pred2_dis = ndimage.distance_transform_edt(pred2)

        structure_map_list1 = []
        structure_map_list2 = []
        diff_thr_before1 = np.zeros(pred1.shape)
        diff_thr_before2 = np.zeros(pred2.shape)
        max_dis1 = pred1_dis.max()
        max_dis2 = pred2_dis.max()
        interval1 = (max_dis1 - 1) / (args.structure_level - 1)
        interval2 = (max_dis2 - 1) / (args.structure_level - 1)

        for level in range(args.structure_level):

            thr1 = 1.0 + level * interval1
            dis_thr1 = pred1_dis.copy()
            dis_thr1[pred1_dis <= thr1] = 0
            pred_thr1 = pred1.copy()
            pred_thr1[dis_thr1 == 0] = 0

            thr2 = 1.0 + level * interval2
            dis_thr2 = pred2_dis.copy()
            dis_thr2[pred2_dis <= thr2] = 0
            pred_thr2 = pred2.copy()
            pred_thr2[dis_thr2 == 0] = 0

            skl_thr1 = skeletonize(pred_thr1, method='lee')
            skl_thr1_ = cv2.dilate(skl_thr1, structure_skl_kernel)
            skl_thr2 = skeletonize(pred_thr2, method='lee')
            skl_thr2_ = cv2.dilate(skl_thr2, structure_skl_kernel)

            diff_thr1 = pred1_skl.astype(np.float32) - skl_thr1.astype(np.float32)
            diff_thr1[diff_thr1 == -1] = 1
            diff_thr1[(pred1_skl_ == 1) & (skl_thr1_ == 1)] = 0
            diff_thr1 = diff_thr1 - diff_thr_before1
            diff_thr_before1 = diff_thr1

            diff_thr2 = pred2_skl.astype(np.float32) - skl_thr2.astype(np.float32)
            diff_thr2[diff_thr2 == -1] = 1
            diff_thr2[(pred2_skl_ == 1) & (skl_thr2_ == 1)] = 0
            diff_thr2 = diff_thr2 - diff_thr_before2
            diff_thr_before2 = diff_thr2

            structure_map1 = np.zeros(pred1.shape)
            x, y = np.nonzero(diff_thr1)
            for k in range(len(x)):
                weight_range = math.ceil(args.structure_range * pred1_dis[x[k], y[k]])
                structure_map1[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
            structure_map_list1.append(structure_map1)

            structure_map2 = np.zeros(pred2.shape)
            x, y = np.nonzero(diff_thr2)
            for k in range(len(x)):
                weight_range = math.ceil(args.structure_range * pred2_dis[x[k], y[k]])
                structure_map2[x[k] - weight_range:x[k] + weight_range, y[k] - weight_range:y[k] + weight_range] = 1
            structure_map_list2.append(structure_map2)

        structure_map1_ = np.zeros(pred1.shape)
        structure_map_list1.reverse()
        for i in range(len(structure_map_list1)):
            if i <= args.structure_thr:
                structure_map1_[structure_map_list1[i] == 1] = 1
            else:
                structure_map1_[structure_map_list1[i] == 1] = i + 1 - args.structure_thr

        structure_map2_ = np.zeros(pred2.shape)
        structure_map_list2.reverse()
        for i in range(len(structure_map_list2)):
            if i <= args.structure_thr:
                structure_map2_[structure_map_list2[i] == 1] = 1
            else:
                structure_map2_[structure_map_list2[i] == 1] = i + 1 - args.structure_thr

        structure_map1_[structure_map1_ == 0] = 1
        structure_map2_[structure_map2_ == 0] = 1

        # semantic
        pred1_skl_ = cv2.dilate(pred1_skl, semantic_skl_kernel)
        pred2_skl_ = cv2.dilate(pred2_skl, semantic_skl_kernel)

        diff_sample = pred1_skl.astype(np.float32) - pred2_skl.astype(np.float32)
        diff_sample[diff_sample == -1] = 1
        diff_sample[(pred1_skl_ == 1) & (pred2_skl_ == 1)] = 0

        semantic_map = cv2.dilate(diff_sample.astype(np.uint8), semantic_range_kernel)
        semantic_map[semantic_map == 1] = args.semantic_level
        semantic_map[semantic_map == 0] = 1

        # attention
        attention_map = np.stack([semantic_map, structure_map1_, structure_map2_], axis=1)
        attention_map = np.max(attention_map, 1)

        # save
        semantic_map = Image.fromarray(semantic_map.astype(np.uint8), mode='P')
        semantic_map.putpalette(list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]).flatten()))
        semantic_map.save(save_path_semantic)

        structure_map_save1 = Image.fromarray(structure_map1_.astype(np.uint8), mode='P')
        structure_map_save1.save(save_path_structure1)

        structure_map_save2 = Image.fromarray(structure_map2_.astype(np.uint8), mode='P')
        structure_map_save2.save(save_path_structure2)

        attention_map = Image.fromarray(attention_map.astype(np.uint8), mode='P')
        attention_map.save(save_path_attention)
