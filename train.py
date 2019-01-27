import json
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from dataset import SignateSegDataset
from models import ResNetUnet, TernausNetV2
from losses import LossMulti, FocalLoss
import utils
from validation import validation

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet34')
    parser.add_argument('--is_deconv', default=False,
                        type=lambda x: str(x).lower() == 'true')
    parser.add_argument('--device_ids', default='0,1')
    parser.add_argument('--root', default='runs/full_labels')
    parser.add_argument('--crop_width', default=768, type=int)
    parser.add_argument('--crop_height', default=768, type=int)
    parser.add_argument('--resize_width', default=1408, type=int)
    parser.add_argument('--resize_height', default=896, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--scheduler_factor', default=0.3, type=float)
    parser.add_argument('--scheduler_patience', default=3, type=int)
    parser.add_argument('--early_stopping', default=10, type=int)
    parser.add_argument('--scheduler_metric', default='iou', choices=['iou', 'loss'])
    parser.add_argument('--without_batchnorm', nargs='?', const=True, default=False)
    parser.add_argument('--bn_sync', default=False, type=lambda x: str(x).lower() == 'true')
    parser.add_argument('--metric_threshold', default=1e-3, type=float)
    parser.add_argument('--jaccard_weight', default=0.0, type=float)
    parser.add_argument('--labels_set', default='eval', choices=['eval', 'full'])
    parser.add_argument('--loss', default='bce', choices=['bce', 'focal'])
    parser.add_argument('--method', default='full', choices=['crop', 'resize', 'full'])

    args = parser.parse_args()

    root = base_path / args.root
    root.mkdir(exist_ok=True, parents=True)

    if args.labels_set == 'eval':
        num_classes = 5
    else:
        num_classes = 20

    if args.backbone == 'wider':
        model = TernausNetV2(num_classes)
    else:
        model = ResNetUnet(num_classes, backbone=args.backbone, is_deconv=args.is_deconv)
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    if args.loss == 'bce':
        loss = LossMulti(num_classes, args.jaccard_weight).cuda()
    else:
        loss = FocalLoss(gamma=2).cuda()
    cudnn.benchmark = True

    def get_split(fold=0):
        with open('folds_split.json') as f:
            folds = json.load(f)

        test_names = folds[str(fold)]
        train_names = []
        for f in range(len(folds)):
            if f != fold:
                train_names += folds[str(f)]
        return train_names, test_names

    def train_transform(p=1):
        # TODO: more augs
        base_trans = [
            HorizontalFlip(p=0.5),
            Normalize(),
            ToTensor(num_classes=num_classes)
        ]
        if args.method == 'resize':
            base_trans.insert(0, Resize(args.resize_height, args.resize_width))
        elif args.method == 'crop':
            base_trans.insert(0, RandomCrop(args.crop_height, args.crop_width))
        else:
            base_trans.insert(0, PadIfNeeded(1216, 1984))

        return Compose(base_trans, p=1)

    def val_transform(p=1):
        base_trans = [
            Normalize(),
            ToTensor(num_classes=num_classes)
        ]
        if args.method == 'resize':
            base_trans.insert(0, Resize(args.resize_height, args.resize_width))
        elif args.method == 'crop':
            base_trans.insert(0, CenterCrop(args.crop_height, args.crop_width))
        else:
            base_trans.insert(0, PadIfNeeded(1216, 1984))

        return Compose(base_trans, p=1)

    def make_loader(file_names, shuffle=False, transform=None, mode='train', batch_size=1):
        return DataLoader(SignateSegDataset(base_path, file_names, transform, mode,
                                            args.labels_set),
                          shuffle=shuffle,
                          num_workers=args.num_workers,
                          batch_size=batch_size,
                          pin_memory=torch.cuda.is_available())

    train_file_names, val_file_names = get_split(args.fold)
    print(f'num train: {len(train_file_names)}, num val: {len(val_file_names)}')

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(),
                               batch_size=args.batch_size)
    val_loader = make_loader(val_file_names, transform=val_transform(),
                             batch_size=len(device_ids))

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True)
    )

    utils.train(
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        val_loader=val_loader,
        validation=validation,
        init_optimizer=lambda lr: Adam(model.parameters(), lr),
        root=root,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
