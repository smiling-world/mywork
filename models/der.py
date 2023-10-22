# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle

import torch
from torch.nn import functional as F
from torch.optim import SGD

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


def get(file):
    with open(file, "rb") as f:
        var = pickle.load(f)
    return var


class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = torch.nn.Sequential(
                *list(model.children())[:-1]
            )
        # print(self.resnet_layer)
        # self.resnet_layer = self.resnet_layer.to(self.device)
        for arg in self.resnet_layer.parameters():
            arg.requires_grad = False
        self.resnet_layer.eval()
        self.classifier = torch.nn.Linear(512, 10)
        # print(self.classifier)

    def forward(self, x, returnt='out'):
        features = self.resnet_layer(x).squeeze()
        if returnt == 'features':
            return features
        # print(features.shape)
        out = self.classifier(features)
        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, features)


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.net = Net(get("resnet18/model")).to(self.device)
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
