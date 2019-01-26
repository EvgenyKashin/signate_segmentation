import torch
import imageio
import os
from torch.utils.data import Dataset
import numpy as np

eval_names = ('car', 'pedestrian', 'lane', 'signal')
eval_colors = ((0, 0, 255), (255, 0, 0), (69, 47, 142), (255, 255, 0))

full_names = ('car', 'pedestrian', 'lane', 'bus', 'truck',
              'svehicle', 'motorbike', 'bicycle', 'signal',
              'signs', 'sky', 'building', 'natural', 'wall',
              'ground', 'sidewalk', 'roadshoulder', 'obstacle',
              'others', 'own')
full_colors = [[0, 0, 255], [255, 0, 0], [69, 47, 142], [193, 214, 0],
               [180, 0, 129], [255, 121, 166], [65, 166, 1],
               [208, 149, 1], [255, 255, 0], [255, 134, 0],
               [0, 152, 225], [0, 203, 151], [85, 255, 50],
               [92, 136, 125], [136, 45, 66], [0, 255, 255],
               [215, 0, 255], [180, 131, 135], [81, 99, 0], [86, 62, 67]]


class SignateSegDataset(Dataset):
    def __init__(self, base_path, file_names, transform, mode='train', labels_set='eval'):
        self.base_path = base_path
        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        self.labels_set = labels_set

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_path = self.file_names[item]
        image_path = self.base_path / 'seg_train_images' / (file_path + '.jpg')
        mask_path = self.base_path / 'seg_train_annotations' / (file_path + '.png')
        image = load_img(image_path)
        mask = load_img(mask_path)
        # TODO: after aug transform
        if self.labels_set == 'eval':
            mask = transform_label_mask(mask, eval_colors, delta=1)
        elif self.labels_set == 'full':
            mask = transform_label_mask(mask, full_colors, delta=0)
        else:
            raise ValueError('Wrong labels_set parameter')

        data = {'image': image, 'mask': mask}
        augmented = self.transform(**data)
        image, mask = augmented['image'], augmented['mask'][0]

        if self.mode == 'train':
            return image, mask.long()
        else:
            return image, str(file_path)


def load_img(path):
    return imageio.imread(path)


def transform_label_mask(mask, colors, delta=1):
    # delta for undefined zero class, if not all image pixel are annotated
    label_mask = np.zeros(mask.shape[:2])
    for i, col in enumerate(colors):
        label = (mask == col).sum(axis=2) == 3
        label_mask[label] = i + delta
    return np.expand_dims(label_mask, -1)
