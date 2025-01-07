import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse
from skimage.morphology import skeletonize
import sys
sys.path.append('/data/ldap_shared/home/s_zyf/Semi-Topology-Attention/')
import tools.eval.sknw
from config.dataset_config.dataset_cfg import dataset_cfg
import cv2
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, accuracy_score
from ripser import lower_star_img
import albumentations as A

def betti_number(img_true, pred):
    diags_pred = lower_star_img(pred)[:-1]
    diags = lower_star_img(img_true)[:-1]
    return abs(len(diags_pred) - len(diags))

def eval_pixel(mask_list_flatten, pred_list_flatten, num_classes):

    jaccard = jaccard_score(mask_list_flatten, pred_list_flatten)
    dice = f1_score(mask_list_flatten, pred_list_flatten)
    acc = accuracy_score(mask_list_flatten, pred_list_flatten)

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2
    print('-' * print_num)
    print('|  Px-Jc: {:.4f}'.format(jaccard).ljust(print_num_minus, ' '), '|')
    print('|  Px-Dc: {:.4f}'.format(dice).ljust(print_num_minus, ' '), '|')
    print('| Px-Acc: {:.4f}'.format(acc).ljust(print_num_minus, ' '), '|')

def eval_skl(mask_list, pred_list, num_classes, skl_kernel_size, deletion_thr):

    betti_error = 0
    deletion_num = 0
    fracture_num = 0
    mask_total_num = 0
    hyperplasia_num = 0
    pred_total_num = 0

    for i in range(mask_list.shape[0]):

        betti_error += betti_number(mask_list[i, ...], pred_list[i, ...])

        mask_skl = skeletonize(mask_list[i, ...], method='lee')
        pred_skl = skeletonize(pred_list[i, ...], method='lee')

        mask_graph = tools.eval.sknw.build_sknw(mask_skl, multi=True)
        pred_graph = tools.eval.sknw.build_sknw(pred_skl, multi=True)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (skl_kernel_size, skl_kernel_size))
        mask_skl_ = cv2.dilate(mask_skl, kernel)
        pred_skl_ = cv2.dilate(pred_skl, kernel)

        diff = mask_skl.astype(np.float32) - pred_skl.astype(np.float32)
        diff[(mask_skl_ == 1) & (pred_skl_ == 1)] = 0

        # skl-deletion, skl-fracture
        mask_graph_edge = list(mask_graph.edges.data())
        mask_total_num += len(mask_graph_edge)

        for m in range(len(mask_graph_edge)):

            mask_single_edge = np.zeros(mask_list[i].shape)
            mask_single_edge_ = np.zeros(mask_list[i].shape)
            edge_point = mask_graph_edge[m][2]['pts']

            x = edge_point[:, 0]
            y = edge_point[:, 1]
            coordinate = (x, y)
            mask_single_edge[coordinate] = 1

            mask_single_edge_[(mask_single_edge == 1) & (diff == 1)] = 1

            deletion_edge_num = len(np.nonzero(mask_single_edge_)[0])
            single_edge_num = len(x)

            deletion_severity_rate = deletion_edge_num / single_edge_num

            if deletion_severity_rate >= deletion_thr:
                deletion_num += 1
            elif deletion_edge_num > 0:
                fracture_num += 1

        # skl-hyperplasia
        pred_graph_edge = list(pred_graph.edges.data())
        pred_total_num += len(pred_graph_edge)

        for n in range(len(pred_graph_edge)):

            pred_single_edge = np.zeros(mask_list[i].shape)
            pred_single_edge_ = np.zeros(mask_list[i].shape)
            edge_point = pred_graph_edge[n][2]['pts']

            x = edge_point[:, 0]
            y = edge_point[:, 1]
            coordinate = (x, y)
            pred_single_edge[coordinate] = 1

            pred_single_edge_[(pred_single_edge == 1) & (diff == -1)] = 1
            if (pred_single_edge_ == 1).any():
                hyperplasia_num += 1

    betti_error /= mask_list.shape[0]
    skl_deletion = deletion_num / mask_total_num
    skl_fracture = fracture_num / mask_total_num
    skl_hyperplasia = hyperplasia_num / pred_total_num

    print_num = 42 + (num_classes - 3) * 7
    print_num_minus = print_num - 2
    print('|  BetaE: {:.4f}'.format(betti_error).ljust(print_num_minus, ' '), '|')
    print('|  Skl-D: {:.4f}'.format(skl_deletion).ljust(print_num_minus, ' '), '|')
    print('|  Skl-F: {:.4f}'.format(skl_fracture).ljust(print_num_minus, ' '), '|')
    print('|  Skl-H: {:.4f}'.format(skl_hyperplasia).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', default='//10.0.5.233/shared_data/Semi-Topology-Attention/dataset/CBMI_ER/val/mask')
    parser.add_argument('--pred_path', default='//10.0.5.233/shared_data/Semi-Topology-Attention/seg_pred/val/CBMI_ER/best_Result1_BetaE_15.4500')
    parser.add_argument('--dataset_name', default='CBMI_ER', help='SNEMI3D, CREMI, DRIVE')
    parser.add_argument('--skl_kernel_size', default=5, help='15, 5')
    parser.add_argument('--deletion_thr', default=0.7)
    args = parser.parse_args()

    cfg = dataset_cfg(args.dataset_name)

    pred_list = []
    mask_list = []

    for img in os.listdir(args.pred_path):
        mask_path = os.path.join(args.mask_path, img)
        pred_path = os.path.join(args.pred_path, img)

        mask = Image.open(mask_path)
        mask = np.array(mask)
        pred = Image.open(pred_path)
        pred = np.array(pred)

        resize = A.Resize(cfg['SIZE'], cfg['SIZE'], p=1)(image=pred, mask=mask)
        mask = resize['mask']
        # pred = resize['image']

        mask_list.append(mask)
        pred_list.append(pred)


    mask_list = np.array(mask_list)
    pred_list = np.array(pred_list)
    assert mask_list.shape == pred_list.shape

    eval_pixel(mask_list.flatten(), pred_list.flatten(), cfg['NUM_CLASSES'])
    eval_skl(mask_list, pred_list, cfg['NUM_CLASSES'], args.skl_kernel_size, args.deletion_thr)
