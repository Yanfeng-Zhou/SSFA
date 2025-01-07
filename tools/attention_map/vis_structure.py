import numpy as np
import os
import argparse
from PIL import Image
import cv2
from config.dataset_config.dataset_cfg import dataset_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='D:/Desktop/attention map figure/image')
    parser.add_argument('--structure_path', default='D:/Desktop/attention map figure/structure2')
    parser.add_argument('--dataset_name', default='SNEMI3D')
    parser.add_argument('--kernel_size', default=11, help='7, 9, 11')
    parser.add_argument('--merge_weight', default=1.25, help='1, 1.25, 1.5')
    parser.add_argument('--save_heatmap_path', default='D:/Desktop/attention map figure/heatmap2')
    parser.add_argument('--save_merge_path', default='D:/Desktop/attention map figure/merge2')
    args = parser.parse_args()

    cfg = dataset_cfg(args.dataset_name)

    if not os.path.exists(args.save_heatmap_path):
        os.mkdir(args.save_heatmap_path)
    if not os.path.exists(args.save_merge_path):
        os.mkdir(args.save_merge_path)

    for i in os.listdir(args.structure_path):

        structure_path = os.path.join(args.structure_path, i)
        image_path = os.path.join(args.image_path, i)
        save_heatmap_path = os.path.join(args.save_heatmap_path, i)
        save_merge_path = os.path.join(args.save_merge_path, i)

        structure = Image.open(structure_path)
        structure = np.array(structure)

        structure = (structure - structure.min()) / (structure.max() - structure.min()) * 255
        structure = cv2.GaussianBlur(structure, (args.kernel_size, args.kernel_size), 0, 0)
        structure = cv2.applyColorMap(structure.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(save_heatmap_path, structure)

        structure = np.float32(structure) / 255

        img = cv2.imread(image_path)
        img = cv2.resize(img, (cfg['SIZE'], cfg['SIZE']))
        img = np.float32((img - img.min()) / (img.max() - img.min()))

        merge_image = args.merge_weight * structure + img
        merge_image = (merge_image - merge_image.min()) / (merge_image.max() - merge_image.min()) * 255
        merge_image = merge_image.astype(np.uint8)
        cv2.imwrite(save_merge_path, merge_image)

