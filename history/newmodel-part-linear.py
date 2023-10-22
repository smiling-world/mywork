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
from resnet import Resnet18
import torchvision.transforms as transforms
difference_id = 0
judge_id = 0
eps = 1e-6
answer_difference = [("automobile", "airplane", "wings"),
                     ("airplane", "bird", "feathers"),
                     ("automobile", "cat", "fur"),
                     ("cat", "deer", "antlers"),
                     ("cat", "dog", "floppy ears"),
                     ("automobile", "frog", "webbed feet"),
                     ("cat", "horse", "mane"),
                     ("automobile", "ship", "hull"),
                     ("automobile", "truck", "sedan body")]
answer_judge = ["yes", "no", "no", "yes", "no", "yes", "no", "no", "no", "no", "yes", "no", "no", "no", "no", "no", "no", "no", "no", "no"]
openai.api_key = "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"

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--layer', type=str, default='resnet18')
    parser.add_argument('--func', type=str, default='sigmoid')
    return parser


class MLP(torch.nn.Module):
    def __init__(self, ebd, size):
        super(MLP, self).__init__()
        self.ebd = ebd.float()
        self.layer1 = torch.nn.Sequential(
                            torch.nn.Linear(size, 1024),
                            torch.nn.ReLU(),
                        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )
        self.layer3 = torch.nn.Linear(1024, 512)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = self.layer3(x)
        x = F.normalize(x)
        # print(x.shape)
        return torch.matmul(x, self.ebd)


def save(var, file):
    print(file)
    with open(file, "wb") as f:
        pickle.dump(var, f)


def get(file):
    with open(file, "rb") as f:
        var = pickle.load(f)
    return var


def query_chatGPT(c, feature):
    global judge_id
    global answer_judge
    # print(judge_id)
    result = answer_judge[judge_id]
    # print(c, feature, result)
    judge_id = judge_id + 1
    if result == "yes":
        return True
    else:
        return False


def query_difference_chatGPT(c):
    global difference_id
    global answer_difference
    result = answer_difference[difference_id]
    difference_id = difference_id + 1
    return result


def func(x, type):
    if type == 'linear':
        return x
    elif type == 'sigmoid':
        y = 2 * x - 1
        # return torch.exp(x) / (torch.exp(x) + 1)
        return torch.exp(x * 3) / (torch.exp(x * 3) + 1)
    # return torch.exp(x) / (torch.exp(x) + torch.exp(1 - x))

class newmodel(ContinualModel):
    NAME = 'newmodel'
    COMPATIBILITY = ['task-il', 'class-il']

    def __init__(self, backbone, loss, args, transform):
        super(newmodel, self).__init__(backbone, loss, args, transform)
        self.device = "cuda:" + str(self.args.cuda)
        # self.device = "cpu"
        self.buffer = Buffer(self.args.buffer_size, self.device, get_dataset(args).N_TASKS, mode='ring')
        self.model, _ = clip.load('ViT-B/32', "cpu")
        self.task_number = 0
        self.memsize = args.buffer_size // get_dataset(args).N_TASKS
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.depth = [1e-6] * self.num_classes
        self.features = []
        self.rules = dict()
        self.MLPs = [None]
        self.M = []
        self.optims = [None]
        self.son = [[-1, -1]]
        self.father = [-1]
        self.node = [[None, []]]
        self.classes = []
        self.path = []
        self.mat_mul = torch.zeros([0, self.num_classes]).to(self.device)
        self.mat_add = torch.zeros([0, self.num_classes]).to(self.device)
        resnet = get(self.args.layer + "/model")
        self.resnet_layer = torch.nn.Sequential(
            *list(resnet.children())[:-1]
        )
        print(self.resnet_layer)
        self.resnet_layer = self.resnet_layer.to(self.device)
        for arg in self.resnet_layer.parameters():
            arg.requires_grad = False
        self.resnet_layer.eval()

    def get_embedding(self, feature):
        ebd = self.model.encode_text(clip.tokenize(f"{feature}")).to(self.device).detach()
        ebd = ebd / ebd.norm(dim=-1, keepdim=True)
        return ebd.squeeze()

    def add_class(self, dataset, c):
        index = len(self.classes)
        self.classes.append(c)
        self.path.append([])
        node_id = 0
        depth = 0
        while not (self.node[node_id][0] is None):
            depth = depth + 1
            self.node[node_id][1].append(index)
            result = query_chatGPT(c, self.node[node_id][0])
            if result == True:
                self.mat_mul[self.M.index(node_id)][index] = 1
                node_id = self.son[node_id][1]
            else:
                self.mat_mul[self.M.index(node_id)][index] = -1
                node_id = self.son[node_id][0]
        self.node[node_id][1].append(index)
        self.depth[self.classes.index(c)] = depth
        if len(self.node[node_id][1]) == 2:
            size = len(self.node)
            self.son[node_id] = [size, size+1]
            self.father.extend([node_id, node_id])
            self.son.extend([[-1, -1], [-1, -1]])
            self.MLPs.extend([None, None])
            self.optims.extend([None, None])
            c1, c2, feature = query_difference_chatGPT([self.classes[i] for i in self.node[node_id][1]])
            self.MLPs[node_id] = MLP(self.get_embedding(feature), dataset.SIZE).to(self.device)
            self.optims[node_id] = torch.optim.SGD(self.MLPs[node_id].parameters(), lr=self.args.lr)
            self.node[node_id][0] = feature
            self.node.extend([[None, [self.classes.index(c1)]], [None, [self.classes.index(c2)]]])
            self.mat_mul = torch.cat([self.mat_mul, torch.zeros(1, self.num_classes).to(self.device)], dim=0)
            self.mat_mul[-1][self.classes.index(c1)] = -1
            self.mat_mul[-1][self.classes.index(c2)] = 1
            self.mat_add = torch.cat([self.mat_add, torch.ones(1, self.num_classes).to(self.device)], dim=0)
            # self.mat_add[-1][self.classes.index(c1)] = 0
            self.M.append(node_id)
            self.depth[self.classes.index(c1)] = self.depth[self.classes.index(c2)] = depth + 1
        print(self.node)
        print(self.mat_mul)
        print(self.mat_add)

    def MLP_train(self, id, inputs, labels):
        labels = labels.float() * 2 - 1
        optimizer = self.optims[id]
        MLP = self.MLPs[id]
        loss_func = torch.nn.MSELoss()
        optimizer.zero_grad()
        outputs = MLP(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss

    def begin_task(self, dataset):
        self.mem_label = self.mem_data = None
        for i in range(dataset.i - dataset.N_CLASSES_PER_TASK, dataset.i):
            self.add_class(dataset, dataset.classes[i])
        pass

    def end_task(self, dataset):
        self.task_number += 1
        self.buffer.add_data(examples=self.mem_data, labels=self.mem_label)
        pass

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        real_batch_size = inputs.shape[0]
        if not hasattr(self, 'input_shape'):
            self.input_shape = inputs.shape[1:]
        for id in self.M:
            self.optims[id].zero_grad()
        if not self.buffer.is_empty():
            size = 2 * real_batch_size
            buf_inputs, buf_labels = self.buffer.get_data(size=size, transform=self.transform)
            while buf_inputs.shape[0] < size:
                input, label = self.buffer.get_data(size=size - buf_inputs.shape[0], transform=self.transform)
                buf_inputs = torch.cat([buf_inputs, input], dim=0)
                buf_labels = torch.cat([buf_labels, label], dim=0)
            inputs = torch.cat([inputs, buf_inputs], dim=0)
            labels = torch.cat([labels, buf_labels], dim=0)

        # print(inputs.shape)
        x = self.resnet_layer(inputs.to(self.device))
        similarity = torch.cat([self.MLPs[id](x).unsqueeze(1) for id in self.M], dim=1)
        # target = torch.cat([(torch.matmul(torch.ones((1, similarity.shape[1])).to(self.device), torch.diag(self.mat_mul.T[labels[i]]))
        #                      + torch.matmul(similarity[i].unsqueeze(0), torch.diag((self.mat_mul.T[labels[i]] == 0).float()))) for i in range(similarity.shape[0])], dim=0).to(torch.float)
        print("!!!", x.shape, self.mat_mul.shape)
        similarity = torch.matmul(similarity, self.mat_mul)
        similarity = similarity / (torch.sum(torch.abs(self.mat_mul), dim=0) + eps)
        # for i in range(similarity.shape[0]):
        #     print(torch.ones_like(similarity[i]).shape, self.mat_mul.T[labels[i]].shape, torch.diag(self.mat_mul.T[labels[i]]).shape)
        #     print((torch.matmul(torch.ones_like(similarity[i]), torch.diag(self.mat_mul.T[labels[i]]))
        #                      + torch.matmul(similarity[i], torch.diag((self.mat_mul.T[labels[i]] == 0).float()))))
        # print(similarity.shape)
        # print(target.shape)
        # print(type(similarity), type(target))
        # if self.args.func == "linear":
        #
        print(similarity.shape, labels.shape)
        loss = torch.nn.functional.cross_entropy(similarity, labels)
        # loss = torch.nn.functional.mse_loss(similarity, target)
        loss.backward()

        for id in self.M:
            self.optims[id].step()

        if self.mem_data is None:
            buffer_remain = self.memsize
        else:
            buffer_remain = self.memsize - self.mem_data.shape[0]
        if buffer_remain > 0:
            if self.mem_data is None:
                self.mem_data = not_aug_inputs[:min(not_aug_inputs.shape[0], buffer_remain)]
                self.mem_label = labels[:min(not_aug_inputs.shape[0], buffer_remain)]
            else:
                self.mem_data = torch.cat([self.mem_data, not_aug_inputs[:min(not_aug_inputs.shape[0], buffer_remain)]], dim=0)
                self.mem_label = torch.cat([self.mem_label, labels[:min(not_aug_inputs.shape[0], buffer_remain)]], dim=0)
        print(loss)
        return float(loss)

    # def predict(self, node_id, x, mask):
    #     label = torch.zeros_like(mask).float()
    #     if self.MLPs[node_id] is None:
    #         index = self.node[node_id][1][0]
    #         label[mask] = index
    #         return label
    #     result = self.MLPs[node_id](x).cpu()
    #     pos = result >= 0.5
    #     neg = result < 0.5
    #     # pos = result >= 0.5
    #     # neg = result < 0.5
    #     label = self.predict(self.son[node_id][0], x, mask & neg) + self.predict(self.son[node_id][1], x, mask & pos)
    #     # if pos:
    #     #     self.predict(self.son[node_id][1], x, mask)
    #     # else:
    #     #     self.predict(self.son[node_id][0], x, mask)
    #     return label

    def predict(self, node_id, x, similarity):
        if self.MLPs[node_id] is None:
            return similarity
        output = self.MLPs[node_id](x).unsqueeze(1)
        similarity_pos = similarity.clone()
        similarity_pos[self.node[self.son[node_id][0]][1]] = 0
        similarity_neg = similarity - similarity_pos
        similarity_pos = (similarity_pos.T * func(output, self.args.func).cpu()).T
        similarity_neg = (similarity_neg.T * func(1-output, self.args.func).cpu()).T
        return self.predict(self.son[node_id][0], x, similarity_neg) + self.predict(self.son[node_id][1], x, similarity_pos)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.resnet_layer(x.to(self.device))
        # mask = torch.ones(x.shape[0]).bool()
        # predict = self.predict(0, x, mask)
        # print(predict)

        x = self.resnet_layer(x.to(self.device))
        similarity = torch.cat([self.MLPs[id](x).unsqueeze(1) for id in self.M], dim=1)
        if self.args.func == "linear":
            similarity = torch.matmul(similarity, self.mat_mul)
            similarity = similarity / (torch.sum(torch.abs(self.mat_mul), dim=0) + eps)
        else:
            # print(similarity * self.mat_mul + self.mat_add)
            similarity = -torch.log(similarity * self.mat_mul + self.mat_add)
            # print(similarity)
            similarity = (similarity / (torch.sum(torch.abs(self.mat_mul), dim=0) + eps))
        print(similarity)
        # similarity = torch.ones((self.num_classes, x.shape[0]))
        # empty = list(set([i for i in range(self.num_classes)]) - set(self.node[0][1]))
        # similarity[empty] = 0
        # result = torch.log(self.predict(0, x, similarity))
        # print(self.depth)
        # 10 * 2000
        # print(result)
        # result = result / torch.Tensor(self.depth).unsqueeze(1)
        # print(result.T)
        
        return similarity

        # x = self.resnet_layer(x.to(self.device))
        # predict = torch.zeros((x.shape[0], self.num_classes)).to(self.device)
        # cnt = torch.zeros((1, self.num_classes))
        # for i in range(len(self.node)):
        #     if not (self.MLPs[i] is None):
        #         out = self.MLPs[i](x).unsqueeze(1)
        #         MLP_c = torch.zeros(self.num_classes)
        #         MLP_c[torch.tensor(self.node[i][1])] = 1
        #         MLP_c = MLP_c.unsqueeze(0)
        #         print(MLP_c)
        #         cnt = cnt + MLP_c
        #         print(out.shape)
        #         print(MLP_c.shape)
        #         predict = predict + torch.exp(torch.abs(torch.matmul(out, MLP_c.to(self.device)) - 0.5) * 2)
        # predict = predict / (cnt.to(self.device) + 1e-6)
        # print(predict)
        # return predict
