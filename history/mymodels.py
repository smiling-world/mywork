import os
import pickle
import re
import time

import clip
import numpy
import openai
import torch
import torch.nn.functional as F
from PIL import Image

from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from datasets import get_dataset
import torchvision.transforms as transforms



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # parser.add_argument('--memsize', type=int, default=20)
    parser.add_argument('--mix_epoch', type=int, default=0)
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
    def __init__(self, ebd):
        super(MLP, self).__init__()
        self.ebd = ebd.float()
        self.layer1 = torch.nn.Sequential(
                            torch.nn.Linear(28 * 28, 256),
                            torch.nn.ReLU()
                        )
        self.layer2 = torch.nn.Linear(256, 512)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(x)
        return torch.matmul(x, self.ebd)


class mymodel(ContinualModel):
    NAME = 'mymodel'
    COMPATIBILITY = ['task-il', 'class-il']

    def __init__(self, backbone, loss, args, transform):
        super(mymodel, self).__init__(backbone, loss, args, transform)
        self.task_number = 0
        self.backbone = backbone
        self.lr = args.lr
        self.eps = 1e-6
        self.loss = loss
        self.memsize = None
        self.augment = args.augment
        if self.augment == 'none':
            self.transform = None
        elif self.augment == "H-flip":
            self.transform = transforms.RandomHorizontalFlip(0.5)
        elif self.augment == "crop":
            self.transform =  transforms.RandomResizedCrop(28, scale=(0.64, 1.0))
        elif self.augment == "H-flip+crop":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.64, 1.0)),
                transforms.RandomHorizontalFlip(1)
            ])
        self.buffer = Buffer(self.args.buffer_size, self.device, get_dataset(args).N_TASKS, mode='ring')
        self.device = "cuda:0"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.labels = []
        self.MLPs = []
        self.MLP2c = []
        self.features = []
        self.classes = []
        self.c2MLP = None

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
                ebd = self.model.encode_text(clip.tokenize(f"an object with {feature}").to(self.device)).detach()
                ebd = ebd / ebd.norm(dim=-1, keepdim=True)
                self.MLPs.append(MLP(ebd.squeeze()).to(self.device))
                # self.c2MLP = numpy.concatenate((self.c2MLP, numpy.zeros((1, self.c2MLP.shape[1]))), axis=0)
                self.c2MLP = torch.cat([self.c2MLP.cpu(), torch.zeros((1, self.c2MLP.shape[1]))], dim=0).to(self.device)
                self.c2MLP[-1, len(self.classes)-1] = 1

    def begin_task(self, dataset):
        # dataset.i
        if self.memsize is None:
            self.memsize = self.buffer.buffer_size // dataset.N_TASKS
        if self.c2MLP is None:
            # self.c2MLP = numpy.zeros((0, dataset.N_CLASSES_PER_TASK * dataset.N_TASKS))
            self.c2MLP = torch.zeros((0, dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)).to(self.device)
        for i in range(dataset.i - dataset.N_CLASSES_PER_TASK, dataset.i):
            self.add_class(dataset, str(i))
        pass

    def end_task(self, dataset):
        self.task_number += 1
        pass

    def self_train(self, MLP, inputs, labels):
        optimizer = torch.optim.Adam(MLP.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()
        optimizer.zero_grad()
        outputs = MLP(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        real_batch_size = inputs.shape[0] + not_aug_inputs.shape[0]
        if not hasattr(self, 'input_shape'):
            self.input_shape = inputs.shape[1:]

        for MLP, i in zip(self.MLPs, range(len(self.MLPs))):
            if not self.buffer.is_empty():
                # buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
                train_inputs, train_labels = inputs, labels
                train_labels = torch.cat([train_labels, labels], dim=0)
                train_inputs = torch.cat([train_inputs, not_aug_inputs], dim=0)
                while train_labels.shape[0] < real_batch_size * 2:
                    remain_size = real_batch_size * 2 - train_labels.shape[0]
                    buf_inputs, buf_labels = self.buffer.get_data(min(self.args.minibatch_size, remain_size), transform=self.transform)
                    train_labels = torch.cat([train_labels, buf_labels], dim=0)
                    train_inputs = torch.cat([train_inputs, buf_inputs], dim=0)
                # train_inputs = numpy.array(torch.cat((inputs, buf_inputs)))
                # train_labels = numpy.array(torch.cat((labels, buf_labels)))
            else:
                train_inputs, train_labels = inputs, labels
            mask = torch.zeros_like(train_labels)
            for c in self.MLP2c[i]:
                mask = mask | (train_labels == c)
            train_labels = mask.float()
            train_inputs = train_inputs.float().to(self.device)
            train_labels = train_labels.float().to(self.device)
            # print(train_inputs.shape, train_labels.shape)
            self.self_train(MLP, train_inputs, train_labels)

        similarity = torch.cat([MLP(inputs).unsqueeze(1) for MLP in self.MLPs], dim=1)
        # print(numpy.matmul(similarity, self.c2MLP))
        # print(numpy.sum(self.c2MLP, axis=0))
        # similarity = numpy.matmul(similarity, self.c2MLP) / (numpy.sum(self.c2MLP, axis=0) + self.eps)
        similarity = torch.matmul(similarity, self.c2MLP) / (torch.sum(self.c2MLP, dim=0) + self.eps)
        loss = self.loss(similarity, labels)
        buffer_remain = (self.task_number + 1) * self.memsize - len(self.buffer)
        if buffer_remain > 0:
            self.buffer.add_data(examples=not_aug_inputs[:min(not_aug_inputs.shape[0], buffer_remain)],
                                 labels=labels[:min(not_aug_inputs.shape[0], buffer_remain)])
        # print(labels.shape, not_aug_inputs.shape)
        return float(loss)
        # save(self.MLPs, "MNIST/MLPs-2(2)")
        # pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        similarity = torch.cat([MLP(x).unsqueeze(1) for MLP in self.MLPs], dim=1)
        similarity = torch.matmul(similarity, self.c2MLP) / (torch.sum(self.c2MLP, dim=0) + self.eps)
        # similarity = torch.cat([MLP(x).unsqueeze(1) for MLP in self.MLPs], dim=1).detach().numpy()
        # similarity = numpy.matmul(similarity, self.c2MLP) / (numpy.sum(self.c2MLP, axis=0) + self.eps)
        return similarity
