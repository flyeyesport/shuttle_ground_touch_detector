import argparse
import torch
import sys
import os
import torch.distributed
import torch.multiprocessing

sys.path.append(os.getcwd() + '/../..')

from squeezenet_one_channel_two_heads_with_positions import SqueezeNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model convertion tool')
    parser.add_argument('-m', '--model_weights',
                        metavar='<file with pretrained model weights>',
                        required=True,
                        help='file with pretrained model weights')
    parser.add_argument('-o', '--output',
                        metavar='<name of the file to save converted model>',
                        required=True,
                        help='name of the file to save converted model')
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=1,
        rank=0
    )

    model = SqueezeNet(sample_size=128, sample_duration=8)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0],
                                                      find_unused_parameters=True)
    checkpoint = torch.load(args.model_weights,
                            map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    script_module = torch.jit.script(model.module)
    script_module.save(args.output)
