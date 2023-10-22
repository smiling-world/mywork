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
    parser.add_argument('--alpha', type=float, default=0)
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
        x = self.layer2(x)
        x = self.layer3(x)
        return torch.matmul(x, self.ebd)


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


def query_judge_chatGPT(c, feature, dataset):
    global judge_id
    global answer_judge
    # print(judge_id)
    file = dataset.NAME + f"/judge-{c}-{feature}"
    print(file)
    if os.path.exists(file):
        result = get(file)
    else:
        prompt = f"Answer yes or no. Does {c} have {feature} or Is {c} {feature}."
        result = query_chatGPT(prompt)
        save(result, file)
    print(c, feature, result)
    if result.find("yes") >= 0:
        return True
    else:
        return False


def query_difference_chatGPT(c, dataset):
    global difference_id
    global answer_difference
    str = ""
    file = dataset.NAME + f"/diff-{c[0]}{c[1]}"
    print(file)
    if os.path.exists(file):
        feature = get(file)
    else:
        while True:
            prompt = f"Please only list one visual feature which {c[1]} has but {c[0]} not"+str+"."
            feature = query_chatGPT(prompt)
            feature = feature.split(",")[0].split(".")[0]
            if query_judge_chatGPT(c[1], feature, dataset) == True and query_judge_chatGPT(c[0], feature, dataset) == False:
                save(feature, file)
            else:
                if str == "":
                    str = ", except " + feature
                else:
                    str = str + ", " + feature
    print(c[0], c[1], feature)
    # result = answer_difference[difference_id]
    # difference_id = difference_id + 1
    return (c[0], c[1], feature)


def func(x, type):
    if type == 'linear':
        return (x + 1) / 2
    elif type == 'sigmoid':
        return torch.exp(x) / (torch.exp(x) + 1)

class newmodelpt(ContinualModel):
    NAME = 'newmodelpt'
    COMPATIBILITY = ['task-il', 'class-il']

    def __init__(self, backbone, loss, args, transform):
        super(newmodelpt, self).__init__(backbone, loss, args, transform)
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
        ebd = self.model.encode_text(clip.tokenize(f"a photo of object with feature of {feature}")).to(self.device).detach()
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
            result = query_judge_chatGPT(c, self.node[node_id][0], dataset)
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
            c1, c2, feature = query_difference_chatGPT([self.classes[i] for i in self.node[node_id][1]], dataset)
            self.MLPs[node_id] = MLP(self.get_embedding(feature), dataset.SIZE).to(self.device)
            self.optims[node_id] = torch.optim.SGD(self.MLPs[node_id].parameters(), lr=self.args.lr)
            self.node[node_id][0] = feature
            self.node.extend([[None, [self.classes.index(c1)]], [None, [self.classes.index(c2)]]])
            self.mat_mul = torch.cat([self.mat_mul, torch.zeros(1, self.num_classes).to(self.device)], dim=0)
            self.mat_mul[-1][self.classes.index(c1)] = -1
            self.mat_mul[-1][self.classes.index(c2)] = 1
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
            self.add_class(dataset, dataset.classes[i].replace("_", " "))
        pass

    def end_task(self, dataset):
        self.task_number += 1
        size = self.args.buffer_size // (dataset.N_CLASSES_PER_TASK * self.task_number)
        if not self.buffer.is_empty():
            buffer_data, buffer_label = self.buffer.get_all_data()
            self.buffer.empty()
            for label in buffer_label.unique():
                mask = (buffer_label == label)
                data, label = buffer_data[mask], buffer_label[mask]
                self.buffer.add_data(examples=data[:size], labels=label[:size])

        for label in self.mem_label.unique():
            mask = (self.mem_label == label)
            data, label = self.mem_data[mask].to(self.device), self.mem_label[mask].to(self.device)
            self.buffer.add_data(examples=data[:size], labels=label[:size])
        pass

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        real_batch_size = inputs.shape[0]
        if self.mem_data is None:
            self.mem_data, self.mem_label = not_aug_inputs.cpu(), labels.cpu()
        else:
            self.mem_data = torch.cat([self.mem_data, not_aug_inputs.cpu()], dim=0)
            self.mem_label = torch.cat([self.mem_label, labels.cpu()], dim=0)

        if not hasattr(self, 'input_shape'):
            self.input_shape = inputs.shape[1:]
        for id in self.M:
            self.optims[id].zero_grad()
        if not self.buffer.is_empty():
            # size = real_batch_size
            size = get_dataset(self.args).get_minibatch_size()
            buf_inputs, buf_labels = self.buffer.get_data(size=size, transform=self.transform)
            while buf_inputs.shape[0] < size:
                input, label = self.buffer.get_data(size=size - buf_inputs.shape[0], transform=self.transform)
                buf_inputs = torch.cat([buf_inputs, input], dim=0)
                buf_labels = torch.cat([buf_labels, label], dim=0)
            inputs = torch.cat([inputs, buf_inputs], dim=0)
            labels = torch.cat([labels, buf_labels], dim=0)

        x = self.resnet_layer(inputs.to(self.device))
        similarity = torch.zeros(((self.task_number+1) * get_dataset(self.args).N_CLASSES_PER_TASK, x.shape[0])).to(self.device)
        empty = list(set([i for i in range((self.task_number+1) * get_dataset(self.args).N_CLASSES_PER_TASK)]) - set(self.node[0][1]))
        similarity[empty] = 0
        result = self.predict(0, x, similarity)
        # result = torch.log(result)
        print(result.T[0])
        # result = result / torch.Tensor(self.depth[:(self.task_number+1) * get_dataset(self.args).N_CLASSES_PER_TASK]).unsqueeze(1).to(self.device)

        similarity = torch.cat([self.MLPs[id](x).unsqueeze(1) for id in self.M], dim=1)
        mat_mul = torch.abs(self.mat_mul[:, :(self.task_number + 1) * get_dataset(self.args).N_CLASSES_PER_TASK])
        similarity = torch.matmul(similarity, mat_mul)

        loss_func = torch.nn.NLLLoss()
        loss_func2 = torch.nn.CrossEntropyLoss()
        loss = loss_func(result.T, labels)
        loss.backward()



        # print(inputs.shape)

        # x = self.resnet_layer(inputs.to(self.device))
        # similarity = torch.cat([self.MLPs[id](x).unsqueeze(1) for id in self.M], dim=1)
        # # 2000 * 9
        # # self.mat_mul 9 * 10
        # mat = torch.cat([self.mat_mul.T[labels[i]].unsqueeze(0) for i in range(labels.shape[0])], dim=0)
        # similarity = similarity * torch.abs(mat)
        # loss_func = torch.nn.MSELoss()
        # # print(similarity, mat)
        # loss = loss_func(similarity, mat)
        # loss.backward()
        # loss = torch.nn.functional.mse_loss(similarity, target)
        #loss.backward()


        for id in self.M:
            self.optims[id].step()
        print(loss)
        return float(loss)

    # def predict(self, node_id, x, mask):
    #     label = torch.zeros_like(mask).float()
    #     if self.MLPs[node_id] is None:
    #         index = self.node[node_id][1][0]
    #         label[mask] = index
    #         return label
    #     result = self.MLPs[node_id](x).cpu()
    #     pos = result >= 0
    #     neg = result < 0
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
        # print(output)
        mask_pos = torch.tensor([1 if i in self.node[self.son[node_id][1]][1] else 0 for i in range(similarity.shape[0])]).unsqueeze(1).to(self.device)
        mask_neg = torch.tensor([1 if i in self.node[self.son[node_id][0]][1] else 0 for i in range(similarity.shape[0])]).unsqueeze(1).to(self.device)
        return self.predict(self.son[node_id][0], x, (similarity.T + torch.log(func(-output, self.args.func))).T * mask_neg ) + \
               self.predict(self.son[node_id][1], x, (similarity.T + torch.log(func(output, self.args.func))).T * mask_pos)
        # similarity_pos = similarity.clone()
        # similarity_pos[self.node[self.son[node_id][0]][1]] = 0
        # similarity_neg = similarity - similarity_pos
        # similarity_pos = (similarity_pos.T + torch.log(func(output, self.args.func))).T
        # similarity_neg = (similarity_neg.T + torch.log(func(-output, self.args.func))).T
        # return self.predict(self.son[node_id][0], x, similarity_neg) + self.predict(self.son[node_id][1], x, similarity_pos)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.resnet_layer(x.to(self.device))
        # mask = torch.ones(x.shape[0]).bool()
        # predict = self.predict(0, x, mask)
        # print(predict)
        # return predict.to(self.device)

        # x = self.resnet_layer(x.to(self.device))
        # similarity = torch.cat([self.MLPs[id](x).unsqueeze(1) for id in self.M], dim=1)
        # if self.args.func == "linear":
        #     similarity = torch.matmul(similarity, self.mat_mul)
        #     similarity = similarity / (torch.sum(torch.abs(self.mat_mul), dim=0) + eps)
        # else:
        #     # print(similarity * self.mat_mul + self.mat_add)
        #     similarity = -torch.log(similarity * self.mat_mul + self.mat_add)
        #     # print(similarity)
        #     similarity = (similarity / (torch.sum(torch.abs(self.mat_mul), dim=0) + eps))
        # print(similarity)

        x = self.resnet_layer(x.to(self.device))
        similarity = torch.zeros((self.task_number * get_dataset(self.args).N_CLASSES_PER_TASK, x.shape[0])).to(self.device)
        empty = list(set([i for i in range(self.task_number * get_dataset(self.args).N_CLASSES_PER_TASK)]) - set(self.node[0][1]))
        similarity[empty] = 0
        result = self.predict(0, x, similarity)
        # result = torch.log(result)
        # print(self.depth)
        # 10 * 2000
        # print(result)
        # result = result / torch.Tensor(self.depth[:self.task_number * get_dataset(self.args).N_CLASSES_PER_TASK]).unsqueeze(1).to(self.device)
        result = result.T
        # print(result.T)
        print(result[0])
        return result.to(self.device)

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
