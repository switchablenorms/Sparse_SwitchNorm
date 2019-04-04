from __future__ import division, absolute_import

import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import models
import numpy as np

from utils.core import load_state_ckpt
from utils.imagenet_dataset import ImagenetDataset

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='configs/config_resnetv1sn50.yaml')
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--workers", type=int, default=None)
parser.add_argument('--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('--save_result', action='store_true',
                    help='Save logits output and labels into numpy.')

args = parser.parse_args()


def main():
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            if (k in args.__dict__) is False:
                try:
                    setattr(args, k, v)
                except:
                    pass  # keep this one False
            elif args.__dict__[k] is None:
                setattr(args, k, v)

    print('Enabled distributed training.')

    rank, world_size = 0, 1
    args.rank = rank
    args.world_size = world_size

    if rank == 0:
        print('################################')
        print('Parameters')
        print(args)
        print('################################')

    # create model
    print("=> creating model '{}'".format(args.model))
    if 'resnetv1sn' in args.model:
        model = models.__dict__[args.model](using_moving_average=args.using_moving_average, using_bn=args.using_bn,
                                            last_gamma=args.last_gamma)
    else:
        model = models.__dict__[args.model](using_moving_average=args.using_moving_average, using_bn=args.using_bn,
                                            last_gamma=args.last_gamma)

    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    load_state_ckpt(args.checkpoint_path, model)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = ImagenetDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size // args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    validate(val_loader, model, criterion)
    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    world_size = args.world_size
    rank = args.rank
    results = []
    results_label = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var) / world_size

            if args.save_result:
                logit_w = output.data.cpu().numpy()
                label_w = target_var.data.cpu().numpy()
                results.append(logit_w)
                results_label.append(label_w)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data.clone()

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                      loss=losses,
                                                                      top1=top1, top5=top5))

        print('Loss: {loss.avg:.4f}\t'
              'Prec@1: {top1.avg:.3f}\t'
              'Prec@5: {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))

        print('Done')

        if args.save_result:
            results = np.concatenate(results, axis=0)
            results_label = np.concatenate(results_label, axis=0)
            np.save('output_label.npy', results_label)
            np.save('output.npy', results)
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
    """Computes the precision@k for the specified values of k"""
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


if __name__ == '__main__':
    main()
