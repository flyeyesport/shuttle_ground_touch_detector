import argparse
import torch
import onnx
import sys
import os
sys.path.append(os.getcwd() + '/../..')

from squeezenet_one_channel_two_heads import SqueezeNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model convertion tool')
    parser.add_argument('-o', '--output',
                        metavar='<name of the file to save converted model>',
                        required=True,
                        help='name of the file to save converted model')
    args = parser.parse_args()

    model = SqueezeNet(sample_size=128, sample_duration=16)
    model = model.cuda()

    random_input = torch.randn(1, 1, 16, 128, 128).cuda()

    torch.onnx.export(model, random_input,
                      args.output, input_names=['input'],
                      output_names=['output'], export_params=True,
                      verbose=True, opset_version=12)
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

