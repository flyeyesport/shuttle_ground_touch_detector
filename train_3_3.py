import os
import time
import sys
import argparse
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from shuttle_clips_in_sequences_dataset import ShuttleClipsInSequencesDataset
from squeezenet_one_channel_two_heads_with_positions import SqueezeNet
from two_heads_loss_1_5 import TwoHeadsLoss
from utils import save_checkpoint, AverageMeter
import torch.distributed
import torch.multiprocessing


def write_summary(summary_writer, mode_str, epoch, losses, tp, fp,
                  fn, tn, detected_touches, not_detected_touches,
                  missed_detected_touches, missed_by_more_detected_touches):
    if summary_writer:
        summary_writer.add_scalar(mode_str + ' loss', losses.avg,
                                  global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 1 TP', tp,
                                  global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 1 FP', fp,
                                  global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 1 FN', fn,
                                  global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 1 TN', tn,
                                  global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 2 detected touches',
                                  detected_touches, global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 2 not detected touches',
                                  not_detected_touches, global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 2 detected but missed '
                                  'touches',
                                  missed_detected_touches,
                                  global_step=epoch)
        summary_writer.add_scalar(mode_str + 'stage 2 incorrect touches with '
                                  'bigger diff',
                                  missed_by_more_detected_touches,
                                  global_step=epoch)


def run_epoch(gpu, mode, epoch, data_loader, model, criterion, optimizer,
              frames_count, margin, convert_func, summary_writer):
    end_time = time.time()
    if mode == 'train':
        model.train()
        mode_str = 'Training'
    else:
        model.eval()
        mode_str = 'Validation'
    if gpu == 0:
        print('{} at epoch {}'.format(mode_str, epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    tp = fp = fn = tn = 0

    # result == target
    detected_touches = 0

    # target >= 0, but result == -1
    not_detected_touches = 0

    # target >= 0 and result >= 0, but target != result
    missed_detected_touches = 0

    # target >= 0 and result >= 0,
    # but target != result and abs(target - result) > 1
    missed_by_more_detected_touches = 0

    for i, (inputs, positions, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs = model(inputs.cuda(), positions.cuda())
        predicted = convert_func(outputs)

        a_tp = a_fp = a_fn = a_tn = 0
        for k in range(targets.size(0)):
            if targets[k] == -1:
                if predicted[k] == -1:
                    a_tn += 1
                else:
                    a_fp += 1
            elif targets[k] >= margin and targets[k] < frames_count - margin:
                # we don't care about results in margins
                if predicted[k] == -1:
                    a_fn += 1
                    not_detected_touches += 1
                else:
                    a_tp += 1
                    if targets[k] == predicted[k]:
                        detected_touches += 1
                    else:
                        missed_detected_touches += 1
                        if abs(targets[k] - predicted[k]) > 1:
                            missed_by_more_detected_touches += 1
        tp += a_tp
        fp += a_fp
        fn += a_fn
        tn += a_tn

        cuda_targets = targets.cuda()
        cuda_targets = cuda_targets.unsqueeze_(1).float()
        loss = criterion(outputs, cuda_targets)
        losses.update(loss.data, inputs.size(0))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 == 0 and gpu == 0:
            if mode == 'train':
                lr_str = ' lr: {lr:.5f}\t'.format(
                    lr=optimizer.param_groups[0]['lr'])
            else:
                lr_str = ''
            print('{0} Epoch: [{1}][{2}/{3}]\t{4}'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      mode_str,
                      epoch,
                      i,
                      len(data_loader),
                      lr_str,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
            print('Stage 1: TP {tp}\tFP {fp}\tFN {fn}\tTN {tn}'.format(
                tp=tp, fp=fp, fn=fn, tn=tn))
            print('Stage 2: detected touches:', detected_touches,
                  '  not detected touches:', not_detected_touches,
                  '  detected but missed touches:', missed_detected_touches,
                  '  incorrect touches with bigger diff:',
                  missed_by_more_detected_touches)
            sys.stdout.flush()
    if gpu == 0:
        write_summary(summary_writer, mode_str, epoch, losses, tp, fp,
                      fn, tn, detected_touches, not_detected_touches,
                      missed_detected_touches, missed_by_more_detected_touches)

    return losses.avg.item()


def main():
    parser = argparse.ArgumentParser(description='SqueezeNet training tool')
    parser.add_argument('-t', '--train_dir',
                        metavar='<directory with training data>',
                        required=False, default='',
                        help='path to directory with training data')
    parser.add_argument('-v', '--val_dir',
                        metavar='<directory with validation data>',
                        required=False, default='',
                        help='path to directory with validation data')
    parser.add_argument('-r', '--result_dir',
                        metavar='<directory with result data>', default='',
                        required=False, help='path to directory in which to '
                        'save results of training')
    parser.add_argument('-b', '--batch_size', metavar='<size of the batch>',
                        required=False, default=32, type=int,
                        help='batch size, default: 32')
    parser.add_argument('-s', '--start_from',
                        metavar='<path to saved weights and state>',
                        required=False, default='',
                        help='path to saved result of previous training '
                             'to be continued')
    parser.add_argument('-sp', '--start_partial', dest='start_partial',
                        action='store_true')
    parser.set_defaults(start_partial=False)
    parser.add_argument('-d', '--board_dir',
                        metavar='<directory with logs for tensorboard>',
                        required=False, default='',
                        help='path to directory with logs to be generated for '
                             'tensorboard')
    parser.add_argument('-f', '--frames_count', metavar='<number of frames>',
                        required=False, default=8, type=int,
                        help='number of frames processed at once, default: 8')
    parser.add_argument('-sf', '--shaken_frames_count',
                        metavar='<number of randomly shifted sets of frames>',
                        required=False, default=16, type=int,
                        help='number of randomly shifted sets of frames, '
                             'default: 16')
    parser.add_argument('-mf', '--margin_frames',
                        metavar='<number of frames to treat as margin>',
                        required=False, default=0, type=int,
                        help='the number of frames from each side of the '
                        'buffer of frames of length defined by --frames_count '
                        'parameter; in such border frames the model does not '
                        'predict if the shuttlecock hit the ground, but uses '
                        'them to predict the touch of the ground in the '
                        'frames between border frames, default: 0')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-ma', '--master_address',
                        metavar='<IP address of the master node>',
                        required=True)
    parser.add_argument('-mp', '--master_port',
                        metavar='<TCP port of the master node>',
                        required=True)

    args = parser.parse_args()
    if args.train_dir == '' and args.val_dir == '':
        parser.error("at least one of --train_dir and --val_dir required")
    if args.train_dir != '' and args.result_dir == '':
        parser.error("when --train_dir is defined then --result_dir required")
    if args.margin_frames * 2 >= args.frames_count:
        parser.error("value of --margin_frames must be less than half of "
                     "the --frames_count value")
    if args.board_dir != '':
        args.summary_writer = SummaryWriter(args.board_dir)
    else:
        args.summary_writer = None

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.multiprocessing.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(12345)
    torch.autograd.set_detect_anomaly(True)
    model = SqueezeNet(sample_size=128, sample_duration=args.frames_count)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    convert_func = model.convert_predicted_vals

    model = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],
                                                  find_unused_parameters=True)
    )
    if args.train_dir != '':
        training_data = ShuttleClipsInSequencesDataset(
            args.train_dir,
            randomize_shake=True,
            randomize_shake_shift_all=3,
            shaken_frames_count=args.shaken_frames_count,
            frames_count=args.frames_count,
            use_only_with_touch=True,
            only_first_touches=True,
            with_patch_positions=True)

        with_touch = len(training_data)

        training_data = ShuttleClipsInSequencesDataset(
            args.train_dir,
            randomize_shake=True,
            randomize_shake_shift_all=3,
            shaken_frames_count=args.shaken_frames_count,
            frames_count=args.frames_count,
            use_only_without_touch=True,
            only_first_touches=True,
            with_patch_positions=True)

        no_touch = len(training_data)

        training_data = ShuttleClipsInSequencesDataset(
            args.train_dir,
            frames_count=args.frames_count,
            randomize_shake=True,
            randomize_shake_shift_all=3,
            shaken_frames_count=args.shaken_frames_count,
            only_first_touches=True,
            with_patch_positions=True)

        all = len(training_data)
        # print('Data counts:', with_touch, no_touch, all)
        weights = torch.tensor([all / no_touch, all / with_touch])
        # print('weights', weights)

        training_sampler = torch.utils.data.distributed.DistributedSampler(
            training_data,
            num_replicas=args.world_size,
            rank=rank
        )
        training_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=training_sampler)

        optimizer = torch.optim.Adam(model.parameters())

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                   patience=3)

    if args.val_dir != '':
        if args.train_dir == '':
            validation_data = ShuttleClipsInSequencesDataset(
                args.val_dir,
                randomize_shake=True,
                randomize_shake_shift_all=3,
                shaken_frames_count=args.shaken_frames_count,
                frames_count=args.frames_count,
                use_only_with_touch=True,
                only_first_touches=True,
                with_patch_positions=True)

            with_touch = len(validation_data)

            validation_data = ShuttleClipsInSequencesDataset(
                args.val_dir,
                randomize_shake=True,
                randomize_shake_shift_all=3,
                shaken_frames_count=args.shaken_frames_count,
                frames_count=args.frames_count,
                use_only_without_touch=True,
                only_first_touches=True,
                with_patch_positions=True)

            no_touch = len(validation_data)

        validation_data = ShuttleClipsInSequencesDataset(
            args.val_dir,
            frames_count=args.frames_count,
            randomize_shake=True,
            randomize_shake_shift_all=3,
            shaken_frames_count=args.shaken_frames_count,
            only_first_touches=True,
            with_patch_positions=True)

        if args.train_dir == '':
            all = len(validation_data)
            # print('Data counts:', with_touch, no_touch, all)
            weights = torch.tensor([all / no_touch, all / with_touch])
            # print('weights', weights)

        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_data,
            num_replicas=args.world_size,
            rank=rank
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=validation_sampler)

    criterion = TwoHeadsLoss(alpha=1.0, weights=weights,
                             frames_count=args.frames_count,
                             margin=args.margin_frames).cuda()

    if args.start_from != '':
        checkpoint = torch.load(args.start_from,
                                map_location=torch.device('cuda'))
        if args.start_partial:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            first_epoch_num = 1
            best_loss = -1
            best_epoch = 1
        else:
            model.load_state_dict(checkpoint['state_dict'])
            if args.train_dir != '':
                optimizer.load_state_dict(checkpoint['optimizer'])
            first_epoch_num = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
            else:
                best_epoch = first_epoch_num
    else:
        first_epoch_num = 1
        best_loss = -1
        best_epoch = 1

    epochs = 250
    for epoch_num in range(first_epoch_num, epochs):
        if args.train_dir != '':
            loss = run_epoch(gpu, 'train', epoch_num, training_loader,
                             model, criterion, optimizer, args.frames_count,
                             args.margin_frames, convert_func,
                             args.summary_writer)
        if args.val_dir != '':
            validation_loss = run_epoch(gpu, 'validate', epoch_num,
                                        validation_loader, model, criterion,
                                        optimizer, args.frames_count,
                                        args.margin_frames, convert_func,
                                        args.summary_writer)

            is_best = (best_loss == -1 or validation_loss < best_loss)
            if is_best:
                best_loss = validation_loss
                best_epoch = epoch_num
        else:
            is_best = (best_loss == -1 or loss < best_loss)
            if is_best:
                best_loss = loss
                best_epoch = epoch_num
            validation_loss = loss

        if args.result_dir != '' and gpu == 0:
            state = {
                'model_from': 'squeezenet_one_channel_two_heads',
                'epoch': epoch_num,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'loss': loss,
                'validation_loss': validation_loss
            }
            save_checkpoint(state, args.result_dir, is_best)

        if args.train_dir != '':
            scheduler.step(validation_loss)


if __name__ == '__main__':
    main()
