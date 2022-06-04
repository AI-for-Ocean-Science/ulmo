from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        try:
            return [self.transform(x), self.transform(x)]
        except:
            import pdb; pdb.set_trace()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    """
    This function is used to adjust the learning rate during the 
    training process.
    
    Args:
        args: (Prams) option Prams for the training.
        optimizer: (torch.optim) optim used in training.
        epoch: (int) epoch of the training.
    """
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    """
    This is a function used for warm up the learning rate in the
    training.
    Args:
        args: (Params) hyper-parameters and flags for training.
        epoch: (int) epoch id for the training.
        batch_id: (int) batch id for the training.
        total_batches: (int) total_batches of the training.
        optimizer: (torch.optim) optimizer used for training.
    """
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    """
    Create the optim for the model
    Args:
        opt: (Params) opt for the training.
        model: (torch.nn.Module) training model.
        
    Returns:
        optimizer: (torch.optim) optimizer for the training.
    """
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    """
    This is a function used for saving model.

    Args:
        model: (torch.nn.Module) training model.
        optimizer: (torch.optim) optimizer for the model.
        opt: (Params) hyper-parameters and flags for the training.
        epoch: (int) epoch id of the training.
        save_file: (str) path for the saving model.
        
    """
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
