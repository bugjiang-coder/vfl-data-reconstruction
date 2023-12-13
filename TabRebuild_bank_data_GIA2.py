import logging
import os
import random
import sys
import pandas as pd

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
# 加入模块的搜索路径

from fedml_core.preprocess.bank.preprocess_bank import preprocess
from fedml_core.model.bankModels import TopModel, BottomModel, BottomModelDecoder
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import adult_dataset, over_write_args_from_file, Similarity, onehot_softmax, tabRebuildAcc, onehot_bool_loss_v2, onehot_bool_loss, num_loss, keep_predict_loss

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import wandb
import shutil
from tqdm import tqdm

import torch.nn.init as init

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def train_shadow_model(train_queue, device, args):
    # TODO：这里需要考虑2种情况，一种是完全重建一个client的模型（要求结构一致），
    #  一种是尽可能利用我们自己拥有的信息，仿照GRN模型，但是数据相关性的角度来看应该提升不大

    Xa_train = train_queue.dataset.Xa
    Xb_train = train_queue.dataset.Xb

    if os.path.isfile(args.shadow_model):
        print("=> loading decoder mode '{}'".format(args.shadow_model))
        shadow_model = torch.load(args.shadow_model, map_location=device)
        return shadow_model

    # 这里现实现一个完全一致的
    print("################################ load Federated Models ############################")
    # 加载VFL框架
    top_model = TopModel(input_dim=200, output_dim=1)
    bottom_model_list = [BottomModel(input_dim=Xa_train.shape[1], output_dim=100),
                         BottomModel(input_dim=Xb_train.shape[1], output_dim=100)]

    vfltrainer = VFLTrainer(top_model, bottom_model_list, args)

    checkpoint = torch.load(args.base_mode, map_location=device)
    args.start_epoch = checkpoint['epoch']
    vfltrainer.load_model(args.base_mode, device)
    print("=> loaded model '{}' (epoch: {} auc: {})"
          .format(args.base_mode, checkpoint['epoch'], checkpoint['auc']))

    # 重新初始化需要重建的网络
    vfltrainer.bottom_model_list[1].apply(weights_init)

    model_list = bottom_model_list + [top_model]

    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model
        in model_list
    ]

    optimizer_list[0] = None
    optimizer_list[2] = None

    # decoder = train_decoder(net, device, args)

    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    bottom_criterion = keep_predict_loss

    for epoch in range(0, 120):
        train_loss = vfltrainer.train_single_model(train_queue, criterion, bottom_criterion, optimizer_list, 1,False, device, args)
        # train_loss = vfltrainer.train_passive_mode(train_queue, criterion, device, args)
        test_loss, acc, auc, precision, recall, f1 = vfltrainer.test(train_queue, criterion, device)

        print(
            "---shadow_moder--- epoch: {0}, train_loss: {1},test_loss: {2}, test_acc: {3}, test_precison: {4}, test_recall: {5}, test_f1: {6}, test_auc: {7}"
            .format(epoch, train_loss, test_loss, acc, precision, recall, f1, auc))

    shadow_model = vfltrainer.bottom_model_list[1]

    # 检查args.decoder_mode目录是否存在
    save_dir = os.path.dirname(args.shadow_model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(shadow_model, args.shadow_model)
    print("model saved")

    return shadow_model


def train_decoder(net, train_queue, device, args):
    # 注意这个decoder 需要使用测试集进行训练

    Xb_train = train_queue.dataset.Xb

    print("################################ Set Federated Models, optimizer, loss ############################")

    net_output = net(torch.zeros_like(next(iter(train_queue))[0][1]).to(device))
    # print(net_output.shape)
    decoder = BottomModelDecoder(input_dim=net_output.shape[1], output_dim=Xb_train.shape[1]).to(device)


    optimizer = torch.optim.SGD(decoder.parameters(), args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay)

    # loss function
    criterion = nn.MSELoss().to(device)

    # 加载模型

    if os.path.isfile(args.decoder_mode):
        print("=> loading decoder mode '{}'".format(args.decoder_mode))
        decoder = torch.load(args.decoder_mode, map_location=device)
        return decoder

    print("################################ Train Decoder Models ############################")


    for epoch in range(0, 120):
        # train and update
        epoch_loss = []
        for step, (trn_X, trn_y) in enumerate(train_queue):
            trn_X = [x.float().to(device) for x in trn_X]
            batch_loss = []

            optimizer.zero_grad()

            out = decoder(net(trn_X[1]))

            # numloss = num_loss(out, tab['numList'])
            # bloss2 = onehot_bool_loss(out, tab['onehot'], tab['boolList'])
            # bloss2_v2 = onehot_bool_loss_v2(out, tab['onehot'], tab['boolList'])
            #
            # loss = criterion(out, trn_X[1]) + args.numloss * numloss + args.bloss2_v2*bloss2_v2 + args.bloss2*bloss2

            loss = criterion(out, trn_X[1])
            loss.backward()

            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))


        print(
            "--- epoch: {0}, train_loss: {1}"
            .format(epoch, epoch_loss))

    # 检查args.decoder_mode目录是否存在
    save_dir = os.path.dirname(args.decoder_mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(decoder, args.decoder_mode)
    print("model saved")
    return decoder


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




def rebuild(train_data, test_data, tab, device, args):
    print("################################ load Federated Models ############################")

    Xa_train, Xb_train, y_train = train_data
    train_dataset = adult_dataset(train_data)
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    test_dataset = adult_dataset(test_data)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, drop_last=False)
    train_queue = iter(train_queue)

    # 加载VFL框架
    top_model = TopModel(input_dim=200, output_dim=1)
    bottom_model_list = [BottomModel(input_dim=Xa_train.shape[1], output_dim=100),
                         BottomModel(input_dim=Xb_train.shape[1], output_dim=100)]

    vfltrainer = VFLTrainer(top_model, bottom_model_list, args)


    checkpoint = torch.load(args.base_mode, map_location=device)
    args.start_epoch = checkpoint['epoch']
    vfltrainer.load_model(args.base_mode, device)
    print("=> loaded model '{}' (epoch: {} auc: {})"
          .format(args.base_mode, checkpoint['epoch'], checkpoint['auc']))

    net = train_shadow_model(test_queue, device, args)  # 需要恢复数据的网络

    # decoder = train_decoder(net, test_queue, device, args)

    print("################################ recovery data ############################")

    acc_list = []
    onehot_acc_list = []
    num_acc_list = []
    similarity_list = []
    euclidean_dist_list = []

    for trn_X, trn_y in tqdm(train_queue):
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone().detach()
        # xGen_before = decoder(protocolData)

        # onehot_index = tab['onehot']
        # originData = onehot_softmax(originData, onehot_index)

        # xGen = onehot_softmax(xGen_before, onehot_index)

        xGen = torch.zeros(originData.size()).to(device)
        xGen.requires_grad = True

        optimizer = torch.optim.Adam(params=[xGen], lr=args.lr * 100, eps=args.eps, amsgrad=True)

        for j in range(100):  # 迭代优化
            optimizer.zero_grad()

            xProtocolData = net.forward(xGen)

            featureLoss = ((xProtocolData - protocolData) ** 2).mean()  # 欧几里得距离 损失函数

            featureLoss.backward(retain_graph=True)
            optimizer.step()

        acc, onehot_acc, num_acc = tabRebuildAcc(originData, xGen, tab)
        similarity = Similarity(xGen, originData)
        euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(xGen, originData)).item()
        # print("acc:", acc)
        # print("onehot_acc:", onehot_acc)
        # print("num_acc:", num_acc)
        # print(f"Similarity: {similarity}")
        # print(f"euclidean_dist: {euclidean_dist}")
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
    print("acc:", acc)
    print("onehot_acc:", onehot_acc)
    print("num_acc:", num_acc)
    print("similarity", similarity)
    print("euclidean_dist", euclidean_dist)
    print("################################ save data ############################")
    if args.save != '':
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        record_experiment(args, acc, onehot_acc, num_acc, similarity, euclidean_dist)         # # 保存元素数据
    return acc, onehot_acc, num_acc, similarity, euclidean_dist

# 现在需要进行实验记录
def record_experiment(args, acc, onehot_acc, num_acc, similarity, euclidean_dist):
    # 使用pandas记录实验数据
    df = pd.DataFrame()
    # print(acc, onehot_acc, num_acc, similarity, euclidean_dist)
    df['acc'] = [acc]
    df['onehot_acc'] = [onehot_acc]
    df['num_acc'] = [num_acc]
    df['similarity'] = [similarity]
    df['euclidean_dist'] = [euclidean_dist]
    for key in args.__dict__:
        df[key] = [args.__dict__[key]]
        # print(f"{key}: {args.__dict__[key]}")  # 添加打印语句，检查属性值
    # print(df)
    print(df.values)
    df.to_csv(args.save + "record_experiment_shadow.csv", mode='a', header=True, index=False)





def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    print("################################ prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("TabRebuild")
    parser.add_argument('--multiple', action='store_true', help='Whether to conduct multiple experiments')
    parser.add_argument('--name', type=str, default='decoder-rebuild', help='experiment name')
    parser.add_argument('--data_dir', default='/home/yangjirui/VFL/feature-infer-workspace/dataset/adult/adult.data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save',
                        default="/home/yangjirui/paper-code/data/adult/2layer-noloss/base-random/origin_data.csv",
                        help='location of the data corpus')
    parser.add_argument('--decoder_mode',
                        default="/home/yangjirui/paper-code/data/adult/2layer-noloss/base-random/origin_data.csv",
                        help='location of the decoder mode')
    # parser.add_argument('--inverse_data_output', default="/home/yangjirui/paper-code/data/adult/2layer-noloss/base-random/inverse_data.csv",
    #                     help='location of the data corpus')
    parser.add_argument('--base_mode', default='', type=str, metavar='PATH',
                        help='path to latest base mode (default: none)')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    # ==========下面是几个重要的超参数==========
    parser.add_argument('--NIters', type=int, default=5000, help="Number of times to optimize")
    parser.add_argument('--Ndata', type=int, default=100, help="Recovery data quantity")
    parser.add_argument('--iloss', type=float, default=0, help="Recovery data int loss intensity")
    parser.add_argument('--bloss', type=float, default=0, help="Recovery data boolean loss intensity")
    parser.add_argument('--bloss2', type=float, default=0, help="Recovery data boolean loss intensity")
    parser.add_argument('--bloss2_v2', type=float, default=0, help="Recovery data boolean loss intensity")

    parser.add_argument('--nloss', type=float, default=0, help="Recovery data negative number loss intensity")
    parser.add_argument('--norloss', type=float, default=0, help="Recovery data negative number loss intensity")
    parser.add_argument('--numloss', type=float, default=0.01, help="Recovery data negative number loss intensity")

    # config file
    parser.add_argument('--c', type=str, default='./configs/attack/bank/data.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)


    # 指定rebuild的表格特征
    tab = {
        'boolList': [i for i in range(0, 22)],
        'onehot': {
            'marital': [0, 1, 2, 3], 'default': [4, 5, 6],
            'loan': [7, 8, 9], 'month': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'poutcome': [20, 21, 22]
        },
        'numList': [i for i in range(23, 28)]
    }

    train, test = preprocess(args.data_dir)

    if args.multiple:
        # 进行多次实验
        acc_all, onehot_acc_all, num_acc_all, similarity_all, euclidean_dist_all = [], [], [], [], []
        decoder_mode = args.decoder_mode
        shadow_model = args.shadow_model

        for i in range(5):
            # 设置随机种子
            freeze_rand(args.seed + i)
            # 是否要规范化
            args.decoder_mode = decoder_mode + str(i)
            args.shadow_model = shadow_model + str(i)

            # 训练并生成
            # 白盒攻击本身并不需要训练数据
            acc, onehot_acc, num_acc, similarity, euclidean_dist = rebuild(train_data=train, test_data=test, tab=tab,
                                                                           device=device, args=args)
            acc_all.append(acc)
            onehot_acc_all.append(onehot_acc)
            num_acc_all.append(num_acc)
            similarity_all.append(similarity)
            euclidean_dist_all.append(euclidean_dist)
        # 计算均值和方差
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        onehot_acc_mean = np.mean(onehot_acc_all)
        onehot_acc_std = np.std(onehot_acc_all)
        num_acc_mean = np.mean(num_acc_all)
        num_acc_std = np.std(num_acc_all)
        similarity_mean = np.mean(similarity_all)
        similarity_std = np.std(similarity_all)
        euclidean_dist_mean = np.mean(euclidean_dist_all)
        euclidean_dist_std = np.std(euclidean_dist_all)

        # 打印结果
        print(f"Accuracy: Mean = {acc_mean}, Std = {acc_std}")
        print(f"One-hot Accuracy: Mean = {onehot_acc_mean}, Std = {onehot_acc_std}")
        print(f"Numeric Accuracy: Mean = {num_acc_mean}, Std = {num_acc_std}")
        print(f"Similarity: Mean = {similarity_mean}, Std = {similarity_std}")
        print(f"Euclidean Distance: Mean = {euclidean_dist_mean}, Std = {euclidean_dist_std}")


    else:
        # 设置随机种子
        freeze_rand(args.seed)
        # 是否要规范化


        # 训练并生成
        # 白盒攻击本身并不需要训练数据
        rebuild(train_data=train, test_data=test, tab=tab, device=device, args=args)