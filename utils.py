import os
import csv
import torch
import shutil
import numpy as np


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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


# def calculate_accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def count_tp_tn(predicted, target):
    return sum(k == l for (k, l) in zip(predicted, target.tolist()))


def count_tp_fp_fn_tn(predicted, target):
    tp = fp = fn = tn = 0
    for (k, l) in zip(predicted, target.tolist()):
        if k == l:
            if k == 0:
                tn += 1
            else:
                tp += 1
        else:
            if k == 1:
                fp += 1
            else:
                fn += 1
    return tp, fp, fn, tn


def save_checkpoint(state, result_path, is_best):
    file_path = os.path.join(result_path, str(state['epoch']) + '_checkpoint.pth')
    torch.save(state, file_path)
    if is_best:
        best_file_path = os.path.join(result_path, 'best_checkpoint.pth')
        shutil.copyfile(file_path, best_file_path)


def adjust_learning_rate(initial_lr, optimizer, epoch_num, steps):
    # Sets the learning rate to the initial LR decayed by 10 after number of epochs in the list below
    lr_new = initial_lr * (0.1 ** (sum(epoch_num >= np.array(steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
