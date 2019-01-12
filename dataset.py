import torch
import imageio
import os
from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor
import numpy as np

eval_names = ('car', 'pedestrian', 'lane', 'signal')
eval_colors = ((0, 0, 255), (255, 0, 0), (69, 47, 142), (255, 255, 0))


class SignateSegDataset(Dataset):
    def __init__(self, base_path, file_names, transform, mode='train'):
        self.base_path = base_path
        self.file_names = file_names
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_path = os.path.join(self.base_path, self.file_names[item])
        image_path = file_path + '.jpg'
        mask_path = file_path + '.png'
        image = load_img(image_path)
        mask = load_img(mask_path)
        mask = transform_label_mask(mask)

        data = {'image': image, 'mask': mask}
        augmented = self.transform(**data)
        image, mask = augmented['image'], augmented['mask']

        if self.mode == 'train':
            return img_to_tensor(image), torch.from_numpy(mask).float()
        else:
            return img_to_tensor(image), str(file_path)


def load_img(path):
    return imageio.imread(path)


def transform_label_mask(mask):
    label_mask = np.zeros(mask.shape[:2])
    for i, eval_color in enumerate(eval_colors):
        label = (mask == eval_color).sum(axis=2) == 3
        label_mask[label] = i
    return label_mask
