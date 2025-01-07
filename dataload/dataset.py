import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

class Dataset_(Dataset):
    def __init__(self, data_dir, augmentation, sup=True, num_images=None):
        super(Dataset_, self).__init__()

        img_paths = []
        mask_paths = []

        image_dir = data_dir + '/image'
        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir):

            image_path = os.path.join(image_dir, image)
            img_paths.append(image_path)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)
        if sup:
            assert len(img_paths) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths = img_paths[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths = img_paths * quotient
                img_paths += [img_paths[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.sup = sup

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = np.array(img)

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)

            augment = self.augmentation(image=img, mask=mask)
            img = augment['image']
            mask = augment['mask']

            sampel = {'image': img, 'mask': mask.long(), 'ID': os.path.split(mask_path)[1]}
        else:
            augment = self.augmentation(image=img)
            img = augment['image']

            sampel = {'image': img, 'ID': os.path.split(img_path)[1]}
        return sampel

    def __len__(self):
        return len(self.img_paths)


def imagefloder(data_dir, data_transform, sup=True, num_images=None):
    dataset = Dataset_(data_dir=data_dir, augmentation=data_transform, sup=sup, num_images=num_images)
    return dataset


class Dataset_TA(Dataset):
    def __init__(self, data_dir, augmentation, sup=True, num_images=None):
        super(Dataset_TA, self).__init__()

        img_paths = []
        structure1_paths = []
        structure2_paths = []
        semantic_paths = []
        attention_paths = []
        mask_paths = []

        image_dir = data_dir + '/image'
        attention_dir = data_dir + '/attention'
        structure1_dir = data_dir + '/structure1'
        structure2_dir = data_dir + '/structure2'
        semantic_dir = data_dir + '/semantic'

        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir):

            image_path = os.path.join(image_dir, image)
            img_paths.append(image_path)

            structure1_path = os.path.join(structure1_dir, image)
            structure1_paths.append(structure1_path)
            structure2_path = os.path.join(structure2_dir, image)
            structure2_paths.append(structure2_path)
            semantic_path = os.path.join(semantic_dir, image)
            semantic_paths.append(semantic_path)
            attention_path = os.path.join(attention_dir, image)
            attention_paths.append(attention_path)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        if num_images is not None:
            len_img_paths = len(img_paths)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths = img_paths[:num_images]
                structure1_paths = structure1_paths[:num_images]
                structure2_paths = structure2_paths[:num_images]
                semantic_paths = semantic_paths[:num_images]
                attention_paths = attention_paths[:num_images]

                if sup:
                    mask_paths = mask_paths[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths = img_paths * quotient
                img_paths += [img_paths[i] for i in new_indices]
                structure1_paths = structure1_paths * quotient
                structure1_paths += [structure1_paths[i] for i in new_indices]
                structure2_paths = structure2_paths * quotient
                structure2_paths += [structure2_paths[i] for i in new_indices]
                semantic_paths = semantic_paths * quotient
                semantic_paths += [semantic_paths[i] for i in new_indices]
                attention_paths = attention_paths * quotient
                attention_paths += [attention_paths[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths = img_paths
        self.structure1_paths = structure1_paths
        self.structure2_paths = structure2_paths
        self.semantic_paths = semantic_paths
        self.attention_paths = attention_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.sup = sup

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = np.array(img)

        structure1_path = self.structure1_paths[index]
        structure1 = Image.open(structure1_path)
        structure1 = np.array(structure1)

        structure2_path = self.structure2_paths[index]
        structure2 = Image.open(structure2_path)
        structure2 = np.array(structure2)

        semantic_path = self.semantic_paths[index]
        semantic = Image.open(semantic_path)
        semantic = np.array(semantic)

        attention_path = self.attention_paths[index]
        attention = Image.open(attention_path)
        attention = np.array(attention)

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)

            augment = self.augmentation(image=img, structure1=structure1, structure2=structure2, semantic=semantic, attention=attention, mask=mask)
            img = augment['image']
            structure1 = augment['structure1']
            structure2 = augment['structure2']
            semantic = augment['semantic']
            attention = augment['attention']
            mask = augment['mask']

            sampel = {'image': img, 'mask': mask.long(), 'structure1': structure1.float(), 'structure2': structure2.float(), 'semantic': semantic.float(), 'attention': attention.float(), 'ID': os.path.split(mask_path)[1]}
        else:

            augment = self.augmentation(image=img, structure1=structure1, structure2=structure2, semantic=semantic, attention=attention)
            img = augment['image']
            structure1 = augment['structure1']
            structure2 = augment['structure2']
            semantic = augment['semantic']
            attention = augment['attention']

            sampel = {'image': img, 'structure1': structure1.float(), 'structure2': structure2.float(), 'semantic': semantic.float(), 'attention': attention.float(), 'ID': os.path.split(img_path)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths)


def imagefloder_TA(data_dir, data_transform, sup=True, num_images=None):
    dataset = Dataset_TA(data_dir=data_dir, augmentation=data_transform, sup=sup, num_images=num_images)
    return dataset
