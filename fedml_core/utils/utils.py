import numpy as np
import pandas as pd
import torch
import ruamel.yaml as yaml
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch.nn.functional as F


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ModelTrainer(ABC):
    """联邦学习训练器的抽象基类
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, active_model, passive_model=[], active_optimizer=[], passive_optimizer_list=[], args=None):
        # 注意：这里传入的model的list 是一个列表
        self.active_model = active_model
        self.passive_model_list = passive_model
        self.active_optimizer = active_optimizer
        self.passive_optimizer_list = passive_optimizer_list
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
    def __init__(self, data):
        # data = data
        self.X_a, self.X_b, self.y = data
    def __getitem__(self, item):
        X_a = self.X_a.iloc[[item]].values.reshape(-1)
        X_b = self.X_b.iloc[[item]].values.reshape(-1)
        y = self.y.iloc[[item]].values.reshape(-1)

        return [np.float32(X_a), np.float32(X_b)], np.float32(y)

    def __len__(self):
        return len(self.X_a)

class adult_dataset(Dataset):
    def __init__(self, data):
        # data = data
        self.Xa, self.Xb, self.y = data
    def __getitem__(self, item):
        Xa = self.Xa[item]
        Xb = self.Xb[item]
        y = self.y[item]

        return [np.float32(Xa), np.float32(Xb)], np.float32(y)

    def __len__(self):
        return len(self.Xa)



class bank_dataset(Dataset):
    def __init__(self, data):
        # data = data
        self.Xa, self.Xb, self.y = data
    def __getitem__(self, item):
        Xa = self.Xa[item]
        Xb = self.Xb[item]
        y = self.y[item]

        return [np.float32(Xa), np.float32(Xb)], np.float32(y)

class credit_dataset(Dataset):
    def __init__(self, data):
        # data = data
        self.Xa, self.Xb, self.y = data

    def __getitem__(self, item):
        Xa = self.Xa[item]
        Xb = self.Xb[item]
        y = self.y[item]

        return [np.float32(Xa), np.float32(Xb)], np.float32(y)

    def __len__(self):
        return len(self.Xa)

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


def tabRebuildAcc(originData, rebuildData, tab, sigma=0.2):
    originData = originData.detach().clone()
    rebuildData = rebuildData.detach().clone()

    # 1. 计算onehot列的准确率
    onehot_index = tab['onehot']

    onehotsuccessnum = 0
    numsuccessnum = 0

    for item in onehot_index:
        origin = torch.argmax(originData[:, onehot_index[item]], dim=1)
        rebuild = torch.argmax(rebuildData[:, onehot_index[item]], dim=1)

        onehotsuccessnum += torch.eq(origin, rebuild).sum().item()

    onehot_acc = onehotsuccessnum/(len(onehot_index) * rebuildData.shape[0])

    # 2. 计算num列的准确率
    for i in tab['numList']:
        origin = originData[:, i]
        rebuild = rebuildData[:, i]
        diff = torch.abs(origin - rebuild)
        # print("diff:", diff)
        # if diff < sigma:
        numsuccessnum += (diff < sigma).sum().item()

        # print("numsuccessnum:", numsuccessnum)
    num_acc = numsuccessnum/(len(tab['numList']) * rebuildData.shape[0])

    return (numsuccessnum+onehotsuccessnum)/((len(tab['numList']) + len(onehot_index)) * rebuildData.shape[0]), onehot_acc, num_acc

def keep_predict_loss(y_true, y_pred):
    return torch.sum(y_true * y_pred)

def test_rebuild_acc(train_queue, net, decoder, tab, device, args):
# for i in range(args.Ndata):
    #     (trn_X, trn_y) = next(train_queue)
    acc_list = []
    onehot_acc_list = []
    num_acc_list = []
    similarity_list = []
    euclidean_dist_list = []

    #  最后测试重建准确率需要在训练集上进行
    for trn_X, trn_y in train_queue:
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone().detach()

        xGen_before = decoder(protocolData)


        onehot_index = tab['onehot']
        # originData = onehot_softmax(originData, onehot_index)



        xGen = onehot_softmax(xGen_before, onehot_index)

        acc, onehot_acc, num_acc = tabRebuildAcc(originData, xGen, tab)
        similarity = Similarity(xGen, originData)
        euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(xGen, originData)).item()

        acc_list.append(acc)
        onehot_acc_list.append(onehot_acc)
        num_acc_list.append(num_acc)
        similarity_list.append(similarity)
        euclidean_dist_list.append(euclidean_dist)

    acc = np.mean(acc_list)
    onehot_acc = np.mean(onehot_acc_list)
    num_acc = np.mean(num_acc_list)
    similarity = np.mean(similarity_list)
    euclidean_dist = np.mean(euclidean_dist_list)
    # print("acc:", acc)
    # print("onehot_acc:", onehot_acc)
    # print("num_acc:", num_acc)
    # print("similarity", similarity)
    # print("euclidean_dist", euclidean_dist)

    return acc, onehot_acc, num_acc, similarity, euclidean_dist

# =================================为表格设计的损失函数=================================
def bool_loss(input, boolList=None):
    # 针对bool列设计的损失函数
    if boolList:
        boolColumn = input[:, boolList]
    else:
        boolColumn = input
    return torch.sum(torch.abs( torch.abs( boolColumn - 0.5 ) - 0.5))

def onehot_bool_loss(input, onehot_index=None, boolList=None):
    input = onehot_softmax(input, onehot_index)
    # 针对bool列设计的损失函数
    if boolList:
        boolColumn = input[:, boolList]
    else:
        boolColumn = input
    return torch.sum(torch.abs( torch.abs( boolColumn - 0.5 ) - 0.5))

def onehot_bool_loss_v2(input, onehot_index=None, boolList=None):
    input = onehot_softmax(input, onehot_index)
    # 针对bool列设计的损失函数
    if boolList:
        boolColumn = input[:, boolList]
    else:
        boolColumn = input

    lower=0
    upper=1

    lower_value = torch.sum(torch.abs(boolColumn[boolColumn < lower] - lower))
    upper_value = torch.sum(torch.abs(boolColumn[boolColumn > upper] - upper))

    return lower_value + upper_value

def onehot_softmax(data, onehot_index):
    # onehot 编码的预处理步骤

    result = data.clone().requires_grad_(True)
    for item in onehot_index:
        result[:, onehot_index[item]] = F.softmax(data[:, onehot_index[item]], dim=1)
    return result

def int_loss(input, intList=None):
    # 针对整数列设计的损失函数
    if intList:
        intColumn = input[:, intList]
    else:
        intColumn = input
    # TODO： 发现一个严重的问题，torch.frac求导的输入和输出是一致的，相当常数
    return torch.sum(torch.abs(torch.frac(intColumn)))

def num_loss(input, numList=None, lower=0, upper=1):
    # 针对整数列设计的损失函数
    if numList:
        numColumn = input[:, numList]
    else:
        numColumn = input

    lower_value = torch.sum(torch.abs(numColumn[numColumn < lower] - lower))
    upper_value = torch.sum(torch.abs(numColumn[numColumn > upper] - upper))

    return lower_value+upper_value

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