import json
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from dataset import SignateSegDataset
from models import ResNetUnet
from losses import CrossEntropyLoss2d
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

base_path = Path('/mnt/ssd0_1/kashin/ai_edge/segmentation')
num_classes = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet34', type=str)
    parser.add_argument('--is_deconv', default=False, type=lambda x: str(x).lower() == 'true')
    parser.add_argument('--device_ids', default='0,1', type=str)
    parser.add_argument('--root',
                        default='/mnt/ssd0_1/kashin/ai_edge/segmentation/runs/crop_512',
                        type=str)
    parser.add_argument('--train_crop_width', default=512, type=int)
    parser.add_argument('--train_crop_height', default=512, type=int)
    parser.add_argument('--train_resize_width', default=960, type=int)
    parser.add_argument('--train_resize_height', default=576, type=int)
    parser.add_argument('--val_crop_width', default=1024, type=int)
    parser.add_argument('--val_crop_height', default=1024, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--batch_size', default=9, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--n_epochs', default=200, type=int)

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    model = ResNetUnet(num_classes, backbone=args.backbone, is_deconv=args.is_deconv)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    loss = CrossEntropyLoss2d().cuda()
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
        return Compose([
#             Resize(args.train_resize_height, args.train_resize_width),
            PadIfNeeded(args.train_crop_height, args.train_crop_width),
            RandomCrop(args.train_crop_height, args.train_crop_width),
            Normalize(),
            ToTensor(num_classes=num_classes)
        ], p=1)

    def val_transform(p=1):
        return Compose([
#             Resize(args.train_resize_height, args.train_resize_width),
            PadIfNeeded(args.train_crop_height, args.train_crop_width),
            CenterCrop(args.train_crop_height, args.train_crop_width),
            Normalize(),
            ToTensor(num_classes=num_classes)
        ])

    def make_loader(file_names, shuffle=False, transform=None, mode='train', batch_size=1):
        return DataLoader(SignateSegDataset(base_path, file_names, transform, mode),
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
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
