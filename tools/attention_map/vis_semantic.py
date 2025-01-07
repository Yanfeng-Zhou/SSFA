import numpy as np
import os
import argparse
from PIL import Image
import cv2
from config.dataset_config.dataset_cfg import dataset_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='D:/Desktop/attention map figure/image')
    parser.add_argument('--semantic_path', default='D:/Desktop/attention map figure/semantic')
    parser.add_argument('--dataset_name', default='SNEMI3D')
    parser.add_argument('--kernel_size', default=41)
    parser.add_argument('--merge_weight', default=1.0)
    parser.add_argument('--save_heatmap_path', default='D:/Desktop/attention map figure/heatmap')
    parser.add_argument('--save_merge_path', default='D:/Desktop/attention map figure/merge')
    args = parser.parse_args()

    cfg = dataset_cfg(args.dataset_name)

    if not os.path.exists(args.save_heatmap_path):
        os.mkdir(args.save_heatmap_path)
    if not os.path.exists(args.save_merge_path):
        os.mkdir(args.save_merge_path)

    for i in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, i)
        semantic_path = os.path.join(args.semantic_path, i)
        save_heatmap_path = os.path.join(args.save_heatmap_path, i)
        save_merge_path = os.path.join(args.save_merge_path, i)

        semantic = Image.open(semantic_path)
        semantic = np.array(semantic)

        semantic = (semantic - semantic.min()) / (semantic.max() - semantic.min()) * 255
        semantic = cv2.GaussianBlur(semantic, (args.kernel_size, args.kernel_size), 0, 0)
        semantic = cv2.applyColorMap(semantic.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(save_heatmap_path, semantic)

        semantic = np.float32(semantic) / 255

        img = cv2.imread(image_path)
        img = cv2.resize(img, (cfg['SIZE'], cfg['SIZE']))
        img = np.float32((img - img.min()) / (img.max() - img.min()))

        merge_image = args.merge_weight * semantic + img
        merge_image = (merge_image - merge_image.min()) / (merge_image.max() - merge_image.min()) * 255
        merge_image = merge_image.astype(np.uint8)
        cv2.imwrite(save_merge_path, merge_image)