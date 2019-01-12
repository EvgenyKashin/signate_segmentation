from pathlib import Path
import json
from itertools import product
from sklearn.model_selection import StratifiedKFold
import argparse

base_path = Path('/mnt/ssd0_1/kashin/ai_edge/segmentation/')
train_ann_path = base_path / 'seg_train_annotations'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', default=7, type=int)
    args = parser.parse_args()

    routes = []
    tods = []
    names = []

    for ann in train_ann_path.iterdir():
        if '.json' == ann.suffix:
            names.append(ann.name.split('.')[0])
            meta = json.loads(ann.read_text())
            routes.append(meta['attributes']['route'])
            tods.append(meta['attributes']['timeofday'])

    attr_prod = list(product(set(routes), set(tods)))
    attr_prod_ind = []

    for route, tod in zip(routes, tods):
        index = attr_prod.index((route, tod))
        attr_prod_ind.append(index)

    folds = {}
    skf = StratifiedKFold(n_splits=args.n_splits, random_state=42)
    splits = skf.split(attr_prod_ind, attr_prod_ind)

    for i, split in enumerate(splits):
        fold_names = []
        for j in split[1]:
            fold_names.append(names[j])
        folds[i] = fold_names

    with open('folds_split.json', 'w') as f:
        json.dump(folds, f)


if __name__ == '__main__':
    main()
