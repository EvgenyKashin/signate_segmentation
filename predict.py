from pathlib import Path
import argparse
import json
import imageio
import cv2
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from models import ResNetUnet, TernausNetV2
from albumentations.torch import ToTensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize
)

base_path = Path('/mnt/ssd/kashin/ai_edge/segmentation')
test_imgs = base_path / 'seg_test_images'  # seg_val_images
eval_colors = ((0, 0, 255), (255, 0, 0), (69, 47, 142), (255, 255, 0))
full_colors = [[0, 0, 255], [255, 0, 0], [69, 47, 142], [193, 214, 0],
               [180, 0, 129], [255, 121, 166], [65, 166, 1],
               [208, 149, 1], [255, 255, 0], [255, 134, 0],
               [0, 152, 225], [0, 203, 151], [85, 255, 50],
               [92, 136, 125], [136, 45, 66], [0, 255, 255],
               [215, 0, 255], [180, 131, 135], [81, 99, 0], [86, 62, 67]]
origin_height = 1216
origin_width = 1936
pad_to_full = 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/nor_resize_jaccard', type=str)
    parser.add_argument('--method', default='full', type=str,
                        choices=['crop', 'crop_stride', 'resize', 'full'])
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--channels', default=5, type=int)
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--predict_labels_set', choices=['eval', 'full'], default='eval')

    args = parser.parse_args()

    method = args.method
    root = base_path / args.root
    labels_set = args.predict_labels_set

    if labels_set == 'eval':
        num_classes = 5
    else:
        num_classes = 20

    train_args = json.loads(root.joinpath('params.json').read_text())
    resize_height = int(train_args['resize_height'])
    resize_width = int(train_args['resize_width'])
    crop_height = int(train_args['crop_height'])
    crop_width = int(train_args['crop_width'])
    fold = int(train_args['fold'])

    if args.epoch == 0:
        state = torch.load(root / f'model_{fold}_best.pth')
        submit_folder = root / f'submit_{method}_{labels_set}'  # val_
        submit_folder.mkdir()
    else:
        state = torch.load(root / f'model_{fold}_{args.epoch}_ep.pth')
        submit_folder = root / f'submit_{method}_{labels_set}_{args.epoch}_ep'
        submit_folder.mkdir()

    if train_args['backbone'] == 'wider':
        model = TernausNetV2(num_classes)  # TODO: add parameter
    else:
        model = ResNetUnet(num_classes, backbone=train_args['backbone'],
                           is_deconv=train_args['is_deconv'])

    model = nn.DataParallel(model, device_ids=[0]).cuda()  # TODO: add parameter
    model.load_state_dict(state['model'])
    model = model.eval()

    def transform(p=1):
        # fix bug
        base_trans = [
            Normalize(),
            ToTensor(num_classes=num_classes)
        ]

        if method == 'resize':
            base_trans.insert(0, Resize(resize_height, resize_width))
        elif method == 'full':
            base_trans.insert(0, PadIfNeeded(origin_height, origin_width + pad_to_full * 2))
        return Compose(base_trans)

    pr_transform = transform()

    # TODO: make predict dataset
    if method == 'crop':
        predict_crops(model, pr_transform, crop_height, crop_width, submit_folder,
                      args.batch_size, labels_set)
    elif method == 'crop_stride':
        predict_stride_crops(model, pr_transform, crop_height, crop_width, submit_folder,
                             args.channels, args.batch_size, labels_set)
    elif method == 'resize':
        predict_full(model, pr_transform, submit_folder, resize=True, labels_set=labels_set)
    elif method == 'full':
        predict_full(model, pr_transform, submit_folder, resize=False, labels_set=labels_set)
    else:
        raise ValueError('Wrong method value')

    root.joinpath('params_predict.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True)
    )


def transform_prediction(pr, labels_set):
    if labels_set == 'eval':
        colors = eval_colors
    elif labels_set == 'full':
        colors = full_colors
    else:
        raise ValueError('Wrong labels_set parameter')
    pr_mask = np.zeros(pr.shape + (3,))
    for i, col in enumerate(colors):
        label = (pr == i + 1)
        pr_mask[label] = col
    pr_mask = pr_mask.astype(np.uint8)
    return pr_mask


def pad_img(img, crop_height, crop_width):
    img = cv2.copyMakeBorder(img, crop_height, crop_height,
                             crop_width, crop_width,
                             cv2.BORDER_REPLICATE)
    return img


def predict_crops(model, tr, crop_height, crop_width, folder, bs=4, labels_set='eval'):
    height_start_inds = np.arange(0, origin_height, crop_height)
    width_start_inds = np.arange(0, origin_width, crop_width)

    for path in tqdm(list(test_imgs.iterdir())):
        img = imageio.imread(path)
        crops = []

        for h_s in height_start_inds:
            for w_s in width_start_inds:
                crop = img[h_s: h_s + crop_height, w_s: w_s + crop_width]
                crops.append(crop)

        crop_tr = [tr(image=cr)['image'].unsqueeze(0) for cr in crops]
        crop_tr = torch.cat(crop_tr, 0)

        crop_pr = []
        for j in range(0, crop_tr.size(0), bs):
            pr = model(crop_tr[j:j+bs])
            crop_pr.append(pr)
        crop_pr = torch.cat(crop_pr, 0)

        pr = []
        for i in range(crop_pr.shape[0]):
            pr.append(transform_prediction(crop_pr[i].data.cpu().numpy().argmax(0),
                                           labels_set=labels_set))

        pr_image = np.zeros((origin_height, origin_width, 3))
        i = 0
        for h_s in height_start_inds:
            for w_s in width_start_inds:
                h_end = min(img.shape[0] - h_s, crop_height)
                w_end = min(img.shape[1] - w_s, crop_width)
                pr_image[h_s: h_s + crop_height, w_s: w_s + crop_width] = pr[i][:h_end, :w_end]
                i += 1
        imageio.imsave(folder / (path.name.split('.')[0] + '.png'), pr_image)


def predict_stride_crops(model, tr, crop_height, crop_width, folder, channels=5, bs=4,
                         labels_set='eval'):
    height_start_stride_inds = np.arange(0, origin_height + 2 * crop_height,
                                         crop_height // channels)
    width_start_stride_inds = np.arange(0, origin_width + 2 * crop_width,
                                        crop_width // channels)
    for path in tqdm(list(test_imgs.iterdir())):
        img = imageio.imread(path)
        paded_img = pad_img(img, crop_height, crop_width)
        pr_image = np.zeros(paded_img.shape[:2] + (channels,))

        all_crops = []
        for i in range(channels):
            crops_stride = []

            for h_s in height_start_stride_inds[i::channels]:
                for w_s in width_start_stride_inds[i::channels]:
                    crop = paded_img[h_s: h_s + crop_height, w_s: w_s + crop_width]
                    crops_stride.append(crop)
            all_crops.append(crops_stride)

            crop_tr = [tr(image=cr)['image'].unsqueeze(0) for cr in crops_stride]
            crop_tr = torch.cat(crop_tr, 0)

            crop_pr = []
            for j in range(0, crop_tr.size(0), bs):
                pr = model(crop_tr[j:j + bs])
                crop_pr.append(pr)
            crop_pr = torch.cat(crop_pr, 0)

            pr = []
            for j in range(crop_pr.shape[0]):
                pr.append(crop_pr[j].data.cpu().numpy().argmax(0))  # transform_prediction after

            k = 0
            for h_s in height_start_stride_inds[i::channels]:
                for w_s in width_start_stride_inds[i::channels]:
                    h_end = min(paded_img.shape[0] - h_s, crop_height)
                    w_end = min(paded_img.shape[1] - w_s, crop_width)
                    pr_image[h_s: h_s + crop_height, w_s: w_s + crop_width, i] = \
                        pr[k][:h_end, :w_end]
                    k += 1
        pr_image_mode = torch.Tensor(pr_image).mode(2, keepdim=False)[0].numpy()
        pr_image_mode = pr_image_mode[crop_height:-crop_height, crop_width:-crop_width]
        pr_image_mode = transform_prediction(pr_image_mode, labels_set=labels_set)
        imageio.imsave(folder / (path.name.split('.')[0] + '.png'), pr_image_mode)


def predict_full(model, tr, folder, resize=True, labels_set='eval'):
    for test_path in tqdm(list(test_imgs.iterdir())):
        test_img = imageio.imread(test_path)
        tr_img = tr(image=test_img)['image']
        pr = model(tr_img.unsqueeze(0))[0]
        pr = pr.data.cpu().numpy().argmax(0)
        pr_full = transform_prediction(pr, labels_set=labels_set)
        if resize:
            pr_full = cv2.resize(pr_full, (origin_width, origin_height),
                                 interpolation=cv2.INTER_CUBIC)
        else:
            pr_full = pr_full[:origin_height, pad_to_full:origin_width+pad_to_full, :]
        imageio.imsave(folder / (test_path.name.split('.')[0] + '.png'), pr_full)


if __name__ == '__main__':
    main()
