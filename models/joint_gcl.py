# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pickle
import re
import time

import clip
import numpy as np
import openai
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


def save(var, file):
    print(file)
    with open(file, "wb") as f:
        pickle.dump(var, f)


def get(file):
    with open(file, "rb") as f:
        var = pickle.load(f)
    return var


def query_chatGPT(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=
        [
            {"role": "system", "content": "You are a helpful assistant to tell me the visual characteristics of any objects."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    text = response.choices[0].message.content.lower()
    return text


def get_feature_from_chatGPT(c, dataset):
    file_path = os.path.join('features_'+dataset.NAME, c)
    print(file_path)
    if os.path.exists(file_path):
        features = get(file_path)
        print(c, features)
        return features
    # query_prompt = "Can you use {} words to describe a {}.".format(num_class_feature, object)
    text = []
    query_prompt = "Please list some simple words separated by commas to describe the visual features of a handwritten \"{}\".".format(c)
    # query_prompt = "Please list some nouns to tell me the visual part of handwritten \"{}\" in visual.".format(c)
    text = query_chatGPT(query_prompt)
    print(text)
    p = re.compile('(\.|, etc|\n)')
    features = p.sub('', text).split(", ")
    save(features, file_path)
    time.sleep(20)
    return features


class MLP(torch.nn.Module):
    def __init__(self, ebd, size):
        super(MLP, self).__init__()
        self.ebd = ebd.float()
        self.layer1 = torch.nn.Sequential(
                            torch.nn.Linear(size, 1024),
                            torch.nn.ReLU()
                        )
        self.layer2 = torch.nn.Linear(1024, 512)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.nn.functional.normalize(x)
        return torch.matmul(x, self.ebd)

class JointGCL(ContinualModel):
    NAME = 'jointGCL'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(JointGCL, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0
        self.device = "cuda:" + str(self.args.cuda)
        # self.device = "cpu"

        resnet = get(self.args.layer + "/model")
        self.resnet_layer = torch.nn.Sequential(
            *list(resnet.children())[:-1]
        )
        self.resnet_layer = self.resnet_layer.to(self.device)
        for arg in self.resnet_layer.parameters():
            arg.requires_grad = False
        self.resnet_layer.eval()

        # self.net = torch.nn.Linear(2048, 10).to(self.device)
        self.text_encoder, _ = clip.load('ViT-B/32', self.device)

        self.labels = []
        self.MLPs = []
        self.optims = []
        self.MLP2c = []
        self.classes = []
        self.features = []
        self.c2MLP = None

        # self.model = torch.load("state_dicts/resnet18.pt")
        # self.model = resnet18(pretrained=True).to(self.device)
        # self.model.eval()
        # print(self.model)

    def add_class(self, dataset, c):
        self.classes.append(c)
        features = get_feature_from_chatGPT(c, dataset)
        for feature in features:
            if feature in self.features:
                index = self.features.index(feature)
                self.MLP2c[index].append(len(self.classes) - 1)
                self.c2MLP[index, len(self.classes)-1] = 1
            else:
                self.features.append(feature)
                self.MLP2c.append([len(self.classes) - 1])
                ebd = self.text_encoder.encode_text(clip.tokenize(f"{feature}").to(self.device)).detach()
                ebd = ebd / ebd.norm(dim=-1, keepdim=True)

                self.MLPs.append(MLP(ebd.squeeze(), dataset.SIZE).to(self.device))
                self.optims.append(torch.optim.Adam(self.MLPs[-1].parameters(), lr=self.args.lr))

                self.c2MLP = torch.cat([self.c2MLP.cpu(), torch.zeros((1, self.c2MLP.shape[1]))], dim=0).to(self.device)
                self.c2MLP[-1, len(self.classes)-1] = 1
                self.c2MLP[self.c2MLP == 0] = -1
                self.c2MLP_pos = self.c2MLP.clone()
                self.c2MLP_neg = self.c2MLP.clone()
                self.c2MLP_pos[self.c2MLP_pos < 0] = 0
                self.c2MLP_pos /= torch.sum(self.c2MLP_pos, dim=1, keepdim=True)
                self.c2MLP_neg[self.c2MLP_neg > 0] = 0
                self.c2MLP_neg /= torch.sum(self.c2MLP_neg < 0, dim=1, keepdim=True)

    def begin_task(self, dataset):
        # if self.memsize is None:
        #     self.memsize = self.buffer.buffer_size // dataset.N_TASKS
        if self.c2MLP is None:
            self.c2MLP = torch.zeros((0, dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)).to(self.device)
        for i in range(dataset.i - dataset.N_CLASSES_PER_TASK, dataset.i):
            self.add_class(dataset, dataset.classes[i])

    def end_task(self, dataset):
        # return
        # print("hihihi")
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1
            # print("!!!!", len(dataset.test_loaders), dataset.N_TASKS)
            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            # reinit network
            # self.net = torch.nn.Linear(dataset.SIZE, self.current_task * dataset.N_CLASSES_PER_TASK)
            # print(self.net)
            # self.net.to(self.device)
            # self.net.train()
            # self.opt = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

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
                    # self.opt.zero_grad()
                    for j in range(len(self.MLPs)):
                        self.optims[j].zero_grad()

                    inputs = self.resnet_layer(inputs).squeeze()
                    # print(inputs.shape)
                    similarity = torch.cat([MLP(inputs).unsqueeze(1) for MLP in self.MLPs], dim=1).to(self.device)
                    # mat = torch.where(self.c2MLP > 0, self.c2MLP_pos, self.c2MLP_neg).to(self.device)
                    outputs = torch.matmul(similarity, self.c2MLP_pos)
                    # outputs = self.net(inputs)
                    # outputs = torch.softmax(outputs, dim=1)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()

                    for j in range(len(self.MLPs)):
                        self.optims[j].step()
                    # self.opt.step()
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
        similarity = torch.cat([MLP(x).unsqueeze(1) for MLP in self.MLPs], dim=1).to(self.device)
        # mat = torch.where(self.c2MLP > 0, self.c2MLP_pos, self.c2MLP_neg).to(self.device)
        output = torch.matmul(similarity, self.c2MLP_pos)
        # output = self.net(x)
        # output = self.model(x)
        # output = torch.softmax(output, dim=1)
        return output

