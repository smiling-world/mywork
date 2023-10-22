# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import pickle

import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torchvision import transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.status import progress_bar
from cifar10_models.resnet import resnet18


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--layer', type=str, default='resnet18')
    return parser


def get(file):
    with open(file, "rb") as f:
        var = pickle.load(f)
    return var


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0
        self.device = "cuda:" + str(self.args.cuda)
        # self.device = "cpu"
        resnet = get(self.args.layer + "/model")
        print(resnet)
        self.resnet_layer = torch.nn.Sequential(
            *list(resnet.children())[:-1]
        )
        self.resnet_layer = self.resnet_layer.to(self.device)
        # self.status = self.resnet_layer.training
        for arg in self.resnet_layer.parameters():
            arg.requires_grad = False
        self.resnet_layer.eval()
        # self.net = torch.nn.Linear(2048, 10).to(self.device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 10)
        )
        # self.model = torch.load("state_dicts/resnet18.pt")
        # self.model = resnet18(pretrained=True).to(self.device)
        # self.model.eval()
        # print(self.model)


    def end_task(self, dataset):
        # return
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            # reinit network
            # self.net = torch.nn.Linear(dataset.SIZE, self.current_task * dataset.N_CLASSES_PER_TASK)
            # print(self.net)
            # self.net.to(self.device)
            self.net.train()
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            print(all_data.shape)
            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            # self.resnet_layer.train(self.status)
            # status = self.resnet_layer.training
            # self.resnet_layer.eval()
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # print(inputs.shape)
                    inputs = self.resnet_layer(inputs).squeeze()
                    # print(inputs.shape)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    # outputs = torch.softmax(outputs, dim=1)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()

                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
            # self.status = self.resnet_layer.training
            # self.resnet_layer.eval()
            # self.resnet_layer.train(status)
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            scheduler = dataset.get_scheduler(self, self.args)

            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i + 1) * bs]
                    labels = all_labels[order][i * bs: (i + 1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs):
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet_layer(x.to(self.device)).squeeze()
        output = self.net(x)
        # output = self.model(x)
        # output = torch.softmax(output, dim=1)
        return output

