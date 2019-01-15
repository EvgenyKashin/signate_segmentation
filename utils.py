from pathlib import Path
import random
import json
from datetime import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model: nn.Module, criterion, train_loader, val_loader, validation,
          init_optimizer, num_classes=None):
    lr = args.lr
    n_epochs = args.n_epochs
    fold = args.fold
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / f'model_{fold}.pth'
    model_path_best = root / f'model_{fold}_best.pth'
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print(f'Restored model, epoch {epoch}, step {step}')
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step
    }, str(model_path))

    save_best = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step
    }, str(model_path_best))

    report_each = 10
    log = root.joinpath(f'train_{fold}.log').open('a')
    valid_losses = []
    min_loss = 1e6
    best_epoch = 0
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) * args.batch_size)
        tq.set_description(f'Epoch: {epoch}, lr: {lr}')
        losses = []

        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()

                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.5f}')
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)

            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, val_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

            # TODO: min_iou
            if valid_loss < min_loss:
                min_loss = valid_loss
                save_best(epoch)
                write_event(log, step, best_model=True)
                print('New best model')
                best_epoch = epoch

            if epoch - best_epoch > 10:
                write_event(log, step, early_stopping=True)
                print('Early stopping')
                return
            # TODO: add to tqdm valid loss
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

