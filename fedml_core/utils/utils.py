import numpy as np
import pandas as pd
import torch
import ruamel.yaml as yaml
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class ModelTrainer(ABC):
    """联邦学习训练器的抽象基类
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model=[], args=None):
        # 注意：这里传入的model的list 是一个列表
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        pass

    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass


def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

class DFDataset(Dataset):
    def __init__(self, x, y=False):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        x = self.x.iloc[[item]].values.reshape(-1)
        if type(self.y)!=type(False): # 有x有y
            y = self.y.iloc[[item]].values.reshape(-1)
            return np.float32(x), np.float32(y)
        return np.float32(x) # 只有x

    def __len__(self):
        return len(self.x)


class zhongyuan_dataset(Dataset):
    def __init__(self, data):
        # data = data
        self.Xa_train, self.Xb_train, self.y_train = data

    def __len__(self):
        return len(self.Xa_train)

    def __getitem__(self, index):
        return [torch.from_numpy(self.Xa_train[index]),torch.from_numpy(self.Xb_train[index])], torch.from_numpy(self.y_train[index])


class criteo_dataset(Dataset):
    def __init__(self, data):
        # data = data
        self.Xa_train, self.Xb_train, self.y_train = data

    def __len__(self):
        return len(self.Xa_train)

    def __getitem__(self, index):
        return [torch.from_numpy(self.Xa_train[index]),torch.from_numpy(self.Xb_train[index])], torch.from_numpy(self.y_train[index])



def Similarity(x, y):
    # 计算余弦相似度
    # if isinstance(x, torch.Tensor):
    x, y = x.clone().detach(), y.clone().detach()
    # 余弦相似度类
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    if len(x.shape) == 1:
        return cos(x, y).item()
    elif len(x.shape) == 2:
        # 对一个批次来计算相似度
        all_similarity = []

        for i in range(x.shape[0]):
            similarity = cos(x[i], y[i]).item()
            all_similarity.append(similarity)
            # 打印每一个的输出
            # print(f'\t{i}: Similarity: {similarity}')

        return sum(all_similarity)/(len(all_similarity))

def tensor2df(tensorDate):
    if not isinstance(tensorDate, np.ndarray):
        tensorDate = tensorDate.cpu().detach().numpy()

    dfData = pd.DataFrame(data=tensorDate, columns=None)

    return dfData



# =================================为表格设计的损失函数=================================
def bool_loss(input, boolList=None):
    # 针对bool列设计的损失函数
    if boolList:
        boolColumn = input[:, boolList]
    else:
        boolColumn = input
    return torch.sum(torch.abs( torch.abs( boolColumn - 0.5 ) - 0.5))
def int_loss(input, intList=None):
    # 针对整数列设计的损失函数
    if intList:
        intColumn = input[:, intList]
    else:
        intColumn = input
    # TODO： 发现一个严重的问题，torch.frac求导的输入和输出是一致的，相当常数
    return torch.sum(torch.abs(torch.frac(intColumn)))

def neg_loss(input, posList=None):
    # 针对非负列设计的损失函数
    if posList:
        input = input[:, posList]
    neg_values = input[input<0]
    if neg_values.numel() > 0:  # 如果张量中存在负数
        loss = torch.sum(torch.square(neg_values))  # 计算负数的平方和
    else:
        loss = torch.tensor(0.)  # 张量中没有负数，损失为0
    return loss


def normalize_loss(input):
    # 针对规范化后的数据设计的损失函数
    # 计算高斯分布的概率密度函数
    # f = torch.exp(-(x_scaled - self.mu) ** 2 / (2 * self.sigma ** 2)) / (torch.sqrt(2 * torch.tensor([3.1415926])) * self.sigma)
    f = torch.exp(-input ** 2 / 2 ) / (2 * 3.1415926)**0.5
    # 计算损失函数
    loss = -torch.sum(torch.log(f + 1e-10))

    return loss