import torch


class TwoHeadsLoss(torch.nn.Module):

    def __init__(self, weights=None, alpha=0.5, frames_count=16, margin=0):
        super(TwoHeadsLoss, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)
        self.smooth_loss = torch.nn.SmoothL1Loss()
        self.alpha = alpha
        self.frames_count = frames_count
        self.margin = margin

    def cuda(self):
        self.cross_entropy_loss.cuda()
        self.smooth_loss.cuda()
        return self

    def forward(self, outputs, targets):
        # print(targets, targets + 1)
        # return self.cross_entropy_loss(outputs, targets.long())
        ret = torch.empty(targets.shape[0])
        i = 0
        # print(targets)
        # print(outputs)
        for target in targets:
            if target == -1:
                output = outputs[i][:2].unsqueeze(0)
                # print('c', output, target.long() + 1)
                ret[i] = self.cross_entropy_loss(output, target.long() + 1)
                # print('C', ret[i], output, target.long() + 1)
            elif (
                target < self.margin or
                target >= self.frames_count - self.margin
            ):
                ret[i] = 0
            else:
                output = outputs[i][:2].unsqueeze(0)
                smooth_output = outputs[i][2].unsqueeze(0)
                positive = torch.tensor([1])
                if target.is_cuda:
                    positive = positive.cuda()
                # print('a', output, target, positive)
                ret[i] = (self.cross_entropy_loss(output, positive) *
                          self.alpha +
                          self.smooth_loss(smooth_output, target) *
                          (1 - self.alpha))
                # print('A', ret[i], output, target.unsqueeze(0))
            i += 1
        # it may happen, that if there are only 0 assigned to ret
        # (some lines above: ret[i] = 0), then it will not require grad
        if not ret.requires_grad:
            ret.requires_grad = True
        return ret.mean()
