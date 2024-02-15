# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score


def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return precision_score(target.view(-1).cpu(), pred.view(-1).cpu())


def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return recall_score(target.view(-1).cpu(), pred.view(-1).cpu())


def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1(target.view(-1).cpu(), pred.view(-1).cpu())


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / (target.shape[1]*target.shape[2]*len(target)) 
