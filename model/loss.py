# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss2d(nn.Module):
    # output : NxCxHxW float tensor
    # target :  NxHxW long tensor
    # weights : C float tensor
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs, dim=1)) ** self.gamma * F.log_softmax(inputs, dim=1), targets)

def check_focal_loss2d():
    num_c = 3
    weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
    out_x_np = np.random.randint(0, num_c, size=(16*64*64*num_c)).reshape((16, num_c, 64, 64))
    target_np = np.random.randint(0, num_c, size=(16*64*64*1)).reshape((16, 64, 64))
    logits = torch.Tensor(out_x_np)
    target = torch.LongTensor(target_np)
    loss_val = weighted_loss(logits, target, weight=weights)
    print("Focalloss2d: ", loss_val.item())
    
def focal_loss2d(output, target, weights=None):
    return FocalLoss2d(weight=weights)(output, target)

if __name__ == '__main__':
    check_focal_loss2d()
