import numpy as np
import pandas as pd
import torch
import ruamel.yaml as yaml
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

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

    def __len__(self):
        return len(self.Xa)


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

def reconstruct_input(model, target_layer: int):
    """
    Reconstruct input from gradients at the specified target layer.
    
    Args:
        model (LeakageEnabledModel): The model with captured gradients.
        target_layer (int): The layer index where leakage occurs.
    
    Returns:
        Reconstructed input tensor.
    """
    # Assuming target_layer corresponds to a Linear layer
    # and only one neuron is activated by a single input
    fc_layer = model.local_model[target_layer]
    
    # Iterate over parameters to get gradients
    weight_grad = None
    bias_grad = None
    for name, grad in model.param_grads.items():
        if f"{target_layer}.weight" in name:
            weight_grad = grad  # Shape: [out_features, in_features]
        if f"{target_layer}.bias" in name:
            bias_grad = grad  # Shape: [out_features]
    
    if weight_grad is None or bias_grad is None:
        raise ValueError("Gradients for the target layer not found.")
    
    # Find the activated neuron
    # 这里大概是梯度有数值，我们就认为这个神经元被激活了？没有被relu给设置为0
    activated_neurons = torch.nonzero(bias_grad < 0, as_tuple=False).squeeze()
    if activated_neurons.ndimension() == 0:
        activated_neurons = activated_neurons.unsqueeze(0)
    
    # print(f"Activated neurons: {activated_neurons.numel()}")
    # For simplicity, assume only one neuron is activated
    # if activated_neurons.numel() != 1:
    #     print("Multiple neurons activated; exact reconstruction not possible.")
    #     return None
    # print("Activated neurons: ", activated_neurons)
    i = activated_neurons[0].item()
    
    # Compute the input using Equation (2): xi = (dL/dWi) / (dL/dBi)
    delta_Wi = weight_grad[i]
    delta_Bi = bias_grad[i]
    
    reconstructed_input = delta_Wi / delta_Bi
    return reconstructed_input

def reconstruct_input_count(model, target_layer: int):
    """
    Reconstruct input from gradients at the specified target layer.
    
    Args:
        model (LeakageEnabledModel): The model with captured gradients.
        target_layer (int): The layer index where leakage occurs.
    
    Returns:
        Reconstructed input tensor.
    """
    # Assuming target_layer corresponds to a Linear layer
    # and only one neuron is activated by a single input
    fc_layer = model.local_model[target_layer]
    
    
    
    # Iterate over parameters to get gradients
    weight_grad = None
    bias_grad = None
    for name, grad in model.param_grads.items():
        if f"{target_layer}.weight" in name:
            weight_grad = grad  # Shape: [out_features, in_features]
        if f"{target_layer}.bias" in name:
            bias_grad = grad  # Shape: [out_features]
    
    if weight_grad is None or bias_grad is None:
        raise ValueError("Gradients for the target layer not found.")
    
    # Find the activated neuron
    # 这里大概是梯度有数值，我们就认为这个神经元被激活了？没有被relu给设置为0
    activated_neurons = torch.nonzero(bias_grad != 0, as_tuple=False).squeeze()
    if activated_neurons.ndimension() == 0:
        activated_neurons = activated_neurons.unsqueeze(0)

    reconstructed_num = 0
    # i = activated_neurons[0].item()
    for index in activated_neurons:
        i = index.item()
        # Compute the input using Equation (2): xi = (dL/dWi) / (dL/dBi)
        delta_Wi = weight_grad[i]
        delta_Bi = bias_grad[i]
        reconstructed_input = delta_Wi / delta_Bi
        
        # 如果 reconstructed_input 中有 0和1，就认为是重建成果
        if torch.any(reconstructed_input == 0) and torch.any(reconstructed_input == 1):
            reconstructed_num += 1
        
    return reconstructed_num


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

def gaussian_noise_masking(g, ratio):
    device = g.device
    g_norm = torch.norm(g, p=2, dim=1)
    max_norm = torch.max(g_norm)
    gaussian_std = ratio * max_norm/torch.sqrt(torch.tensor(g.shape[1], dtype=torch.float).to(device))

    gaussian_noise = torch.normal(mean=0.0, std=gaussian_std.item(), size=g.shape).to(device)
    # res = [g+gaussian_noise]
    return g+gaussian_noise

def VFLDefender(delta_o, tmax, tmin):
    """
    Apply VFLDefender gradient perturbation to delta_o.

    Args:
        delta_o (torch.Tensor): Original gradients of the output layer (δo).
        tmax (float): Maximum clipping threshold.
        tmin (float): Minimum clipping threshold.

    Returns:
        torch.Tensor: Perturbed gradients (δ̂o).
    """
    # print("delta_o", delta_o)
    # sys.exit(0)
    # 1. Clipping
    delta_o_clipped = torch.clamp(delta_o, min=tmin, max=tmax)

    # 2. L2 Normalization
    norm = torch.norm(delta_o_clipped, p=2)
    if norm > 0:
        delta_o_normalized = delta_o_clipped / norm
    else:
        delta_o_normalized = delta_o_clipped

    # 3. Randomize the norm while keeping direction
    delta_hat_o = torch.empty_like(delta_o_normalized)
    positive_mask = delta_o_normalized >= 0
    negative_mask = delta_o_normalized < 0

    # Generate random values within (0, tmax) for positive gradients
    delta_hat_o[positive_mask] = torch.empty_like(delta_hat_o[positive_mask]).uniform_(0, tmax)

    # Generate random values within (tmin, 0) for negative gradients
    delta_hat_o[negative_mask] = torch.empty_like(delta_hat_o[negative_mask]).uniform_(tmin, 0)
    # print("delta_hat_o", delta_hat_o)
    # sys.exit(0)
    return delta_hat_o

def PA_iMFL(grad, epsilon, gamma, tmax, tmin):
    """
    综合应用局部差分隐私 (LDP)、隐私增强子采样 (PAS)、梯度符号重置 (GSR) 的方法
    :param grad: 原始梯度
    :param epsilon: 差分隐私参数 epsilon
    :param sensitivity: 拉普拉斯噪声的灵敏度
    :param gamma: 子采样比例
    :param tmax: 梯度符号重置的最大阈值
    :param tmin: 梯度符号重置的最小阈值
    :return: 应用隐私增强后的梯度
    """
    # sensitivity为拉普拉斯噪声的灵敏度，
    sensitivity = tmax
    
    # Step 1: 差分隐私 (LDP)
    noise = torch.distributions.Laplace(0, sensitivity / epsilon).sample(grad.shape).to(grad.device)
    grad = grad + noise

    # Step 2: 隐私增强子采样 (PAS)
    num_to_keep = int(gamma * grad.numel())
    scores = torch.abs(grad.view(-1))
    top_indices = torch.topk(scores, num_to_keep).indices
    mask = torch.zeros_like(grad.view(-1))
    mask[top_indices] = 1
    grad = (grad.view(-1) * mask).view_as(grad)

    # Step 3: 梯度符号重置 (GSR)
    grad = grad.clone()
    mask_pos = grad > 0
    mask_neg = grad < 0
    num_pos = mask_pos.sum().item()
    num_neg = mask_neg.sum().item()

    p_pos = num_pos / (num_pos + num_neg)
    rand_signs = torch.rand_like(grad)
    rand_signs[mask_pos] = torch.where(rand_signs[mask_pos] < p_pos, 1.0, -1.0)
    rand_signs[mask_neg] = torch.where(rand_signs[mask_neg] < (1 - p_pos), -1.0, 1.0)

    grad = rand_signs * torch.clamp(grad, tmin, tmax)
    return grad


def gaussian_noise_masking_v2(g, ratio):
    device = g.device
    std = ratio
    noise = torch.randn(g.size()).to(device) * std
    # 将噪声添加到原始张量
    return g + noise

def smashed_data_masking(g):
    # add scalar noise to align with the maximum norm in the batch
    # (expectation norm alignment)
    # yjr:这里的输入不明确但可以确定的是g[0][0]的shape是[batch_size, dim]
    device = g.device
    g_norm = torch.norm(g, p=2, dim=1)
    max_norm = torch.max(g_norm)
    stds = torch.sqrt(torch.maximum(max_norm ** 2 /
                              (g_norm ** 2 + 1e-32) - 1.0, torch.tensor(0.0)))
    standard_gaussian_noise = torch.normal(mean=0.0, std=1.0, size=(g_norm.shape[0], 1)).to(device)
    gaussian_noise = standard_gaussian_noise * stds.view(-1, 1)
    # res = [g[0][0] * (1 + gaussian_noise)]
    return g * (1 + gaussian_noise)
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

def test_rebuild_acc_v2(train_queue, net, decoder, tab, device, args):
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

        xGen_before = decoder(torch.cat((protocolData, trn_X[0]), dim=1))


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

# mse函数
def mse(image1, image2):
    # 不计算batch维度
    return torch.mean((image1 - image2) ** 2, [1, 2, 3])

# psnr函数
def PSNR(image1, image2):
    mse_values = mse(image1, image2)
    # if mse_values == 0:
    #     return float('inf')
    PIXEL_MAX = 1.0 if image1.max() <= 1 else 255.0  # 根据图像数据范围调整
    psnr_values = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse_values))
    return torch.mean(psnr_values).item()

def test_rebuild_psnr(train_queue, net, decoder, device, args):
# for i in range(args.Ndata):
    #     (trn_X, trn_y) = next(train_queue)
    psnr_list = []
    # similarity_list = []
    euclidean_dist_list = []

    #  最后测试重建准确率需要在训练集上进行
    for trn_X, trn_y in train_queue:
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone().detach()

        xGen = decoder(protocolData)

        average_psnr = PSNR(originData, xGen)
        
        # average_psnr = torch.mean(psnr_values)


        # similarity = Similarity(xGen, originData)
        euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(xGen, originData)).item()

        psnr_list.append(average_psnr)
        # similarity_list.append(similarity)
        euclidean_dist_list.append(euclidean_dist)


    psnr = np.mean(psnr_list)
    # similarity = np.mean(similarity_list)
    euclidean_dist = np.mean(euclidean_dist_list)
    # print("acc:", acc)
    # print("onehot_acc:", onehot_acc)
    # print("num_acc:", num_acc)
    # print("similarity", similarity)
    # print("euclidean_dist", euclidean_dist)

    return psnr, euclidean_dist


def save_tensor_as_image(tensor, filename):
    # 确保在CPU上
    tensor = tensor.cpu()
        
    # 转换为numpy数组
    numpy_image = tensor.detach().numpy().transpose(1, 2, 0)  # 从[C, H, W]转换为[H, W, C]
    
    # 如果是浮点类型，需要缩放到[0, 255]并转换为uint8
    if numpy_image.dtype == np.float32 or numpy_image.dtype == np.float64:
        numpy_image = (numpy_image * 255).astype(np.uint8)
    
    # 转换为PIL图像
    mode = 'RGB' if tensor.shape[0] == 3 else 'L'
    pil_image = Image.fromarray(numpy_image, mode=mode)
    
    # 保存图像
    pil_image.save(filename)

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