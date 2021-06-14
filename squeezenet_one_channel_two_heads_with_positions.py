'''SqueezeNet in PyTorch.

See the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
and <0.5MB model size" for more details.
'''

import math
import torch
import torch.nn as nn
from typing import List
from decimal import Decimal, ROUND_HALF_UP


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU()
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)

        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
            out += x
        out = self.relu(out)

        return out


class SqueezeNet(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration):
        super(SqueezeNet, self).__init__()
        self.sample_duration = sample_duration

        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=(1, 2, 2),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256, use_bypass=True)
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv3d(512, 1024, kernel_size=1)
        self.middle = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1),
            nn.Flatten(),
            nn.BatchNorm1d(1024)
        )

        self.classifier_A = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024 + 2 * self.sample_duration, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2)
        )

        self.regressor_A = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024 + 2 * self.sample_duration, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, p):
        f = self.features(x)
        f = torch.cat((self.middle(f), torch.flatten(p, start_dim=1)), 1)
        x = self.classifier_A(f)
        y = torch.full((x.shape[0], 1), -100, dtype=x.dtype)
        if x.is_cuda:
            y = y.cuda()
        y = torch.cat((x, y), 1)
        indexes: List[int] = []
        i = 0
        # next line is only needed for the torch.jit.script() to process
        # this script without errors. Variable r will change value in the for
        # loop.
        r = torch.empty(0)
        for v in x:
            if v[1] >= v[0]:
                if len(indexes) == 0:
                    r = f[i].unsqueeze(0)
                else:
                    r = torch.cat((r, f[i].unsqueeze(0)), 0)
                indexes.append(i)
            i += 1
        if len(indexes) > 0:
            r = self.regressor_A(r)
            # print(r, r[0], r[0].item())
            i = 0
            for j in indexes:
                y[j][2] = r[i].item()
                i += 1
        return y

    def convert_predicted_vals(self, x):
        y = [-1 if (
            v[2] == -100
        ) else (
            int(Decimal(v[2].item()).quantize(0, rounding=ROUND_HALF_UP))
        ) for v in x]
        return torch.IntTensor(y)


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' "
                         "expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = SqueezeNet(**kwargs)
    return model


if __name__ == '__main__':
    model = SqueezeNet(sample_size=128, sample_duration=16)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    output = model(torch.randn(8, 1, 16, 128, 128))
    print(output.shape)

    from torchsummary import summary

    summary(model, (1, 16, 128, 128))
