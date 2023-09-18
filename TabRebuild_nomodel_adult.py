import logging
import os
import random
import sys
import pandas as pd

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# 加入模块的搜索路径

from fedml_core.preprocess.adult.preprocess_adult import preprocess
from fedml_core.model.net import active_model, passive_model, passive_decoder_model
from fedml_core.utils.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import adult_dataset, over_write_args_from_file, Similarity, onehot_softmax, tabRebuildAcc, onehot_bool_loss_v2, onehot_bool_loss, num_loss

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import wandb
import shutil

import torch.nn.init as init

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def train_shadow_model(train_data, device, args):
    # TODO：这里需要考虑2种情况，一种是完全重建一个client的模型（要求结构一致），
    #  一种是尽可能利用我们自己拥有的信息，仿照GRN模型，但是数据相关性的角度来看应该提升不大

    if os.path.isfile(args.shadow_model):
        print("=> loading decoder mode '{}'".format(args.shadow_model))
        shadow_model = torch.load(args.shadow_model, map_location=device)
        return shadow_model

    # 这里现实现一个完全一致的
    print("################################ load Federated Models ############################")

    # 加载原始训练数据，用于对比恢复效果
    Xa_train, Xb_train, y_train = train_data
    train_dataset = adult_dataset(train_data)
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    # train_queue = iter(train_queue)

    # 加载VFL框架
    active_party = active_model(input_dim=Xa_train.shape[1], intern_dim=20, num_classes=1, k=args.k)

    passive_model_list = [passive_model(input_dim=Xb_train.shape[1], intern_dim=20, output_dim=20) for _ in
                          range(args.k - 1)]
    active_party.to(device)
    for model in passive_model_list:
        model.to(device)

    active_optimizer = torch.optim.SGD(active_party.parameters(), args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay)

    passive_optimizer_list = [
        torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model
        in passive_model_list
    ]

    vfltrainer = VFLTrainer(active_party, passive_model_list, active_optimizer, passive_optimizer_list, args)

    checkpoint = torch.load(args.base_mode, map_location=device)
    args.start_epoch = checkpoint['epoch']
    vfltrainer.load_model(args.base_mode, device)
    print("=> loaded model '{}' (epoch: {} auc: {})"
          .format(args.base_mode, checkpoint['epoch'], checkpoint['auc']))

    # 重新初始化需要重建的网络
    vfltrainer.passive_model_list[0].apply(weights_init)

    # decoder = train_decoder(net, device, args)

    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    for epoch in range(0, 120):
        train_loss = vfltrainer.train_passive_mode(train_queue, criterion, device, args)
        acc, auc, test_loss, precision, recall, f1 = vfltrainer.test(train_queue, criterion, device)

        print(
            "---shadow_moder--- epoch: {0}, train_loss: {1},test_loss: {2}, test_acc: {3}, test_precison: {4}, test_recall: {5}, test_f1: {6}, test_auc: {7}"
            .format(epoch, train_loss[0], test_loss, acc, precision, recall, f1, auc))

    shadow_model = vfltrainer.passive_model_list[0]
    torch.save(shadow_model, args.shadow_model)
    print("model saved")

    return shadow_model


def train_decoder(net, device, args):

    print("################################ Load Data ############################")
    train_data, test_data = preprocess(args.data_dir)

    Xa_train, Xb_train, y_train = train_data
    # Xa_test, Xb_test, y_test = test_data

    # dataloader
    train_dataset = adult_dataset(train_data)
    test_dataset = adult_dataset(test_data)

    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, drop_last=False)

    print("################################ Set Federated Models, optimizer, loss ############################")

    decoder = passive_decoder_model(input_dim=20, intern_dim=20, output_dim=Xb_train.shape[1]).to(device)


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

            loss = criterion(out, trn_X[1])
            loss.backward()

            optimizer.step()

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))


        print(
            "--- epoch: {0}, train_loss: {1}"
            .format(epoch, epoch_loss))

    torch.save(decoder, args.decoder_mode)
    print("model saved")
    return decoder

    # vfltrainer.save_model('/data/yangjirui/vfl-tab-reconstruction/model/adult/', 'final.pth.tar')


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




def rebuild(train_data, tab, device, args):
    print("hyper-parameters:")
    print("iteration optimization times: {0}".format(args.NIters))
    print("iloss rate: {0}".format(args.iloss))
    print("bloss rate: {0}".format(args.bloss))
    print("nloss rate: {0}".format(args.nloss))
    print("norloss rate: {0}".format(args.norloss))

    print("################################ load Federated Models ############################")

    # 加载原始训练数据，用于对比恢复效果
    Xa_train, Xb_train, y_train = train_data
    train_dataset = adult_dataset(train_data)
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    train_queue = iter(train_queue)

    # 加载VFL框架
    active_party = active_model(input_dim=Xa_train.shape[1], intern_dim=20, num_classes=1, k=args.k)

    passive_model_list = [passive_model(input_dim=Xb_train.shape[1], intern_dim=20, output_dim=20) for _ in
                          range(args.k - 1)]
    active_party.to(device)
    for model in passive_model_list:
        model.to(device)

    active_optimizer = torch.optim.SGD(active_party.parameters(), args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay)

    passive_optimizer_list = [
        torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model
        in passive_model_list
    ]

    vfltrainer = VFLTrainer(active_party, passive_model_list, active_optimizer, passive_optimizer_list, args)

    checkpoint = torch.load(args.base_mode, map_location=device)
    args.start_epoch = checkpoint['epoch']
    vfltrainer.load_model(args.base_mode, device)
    print("=> loaded model '{}' (epoch: {} auc: {})"
          .format(args.base_mode, checkpoint['epoch'], checkpoint['auc']))

    net = train_shadow_model(train_data,device,args)  # 需要恢复数据的网络

    decoder = train_decoder(net, device, args)

    print("################################ recovery data ############################")

    for i in range(args.Ndata):
        (trn_X, trn_y) = next(train_queue)
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone().detach()

        xGen_before = decoder(protocolData)


        onehot_index = tab['onehot']
        # originData = onehot_softmax(originData, onehot_index)



        xGen = onehot_softmax(xGen_before, onehot_index)

        acc, onehot_acc, num_acc = tabRebuildAcc(originData, xGen, tab)
        print("acc:", acc)
        print("onehot_acc:", onehot_acc)
        print("num_acc:", num_acc)

        similarity = Similarity(xGen, originData)
        euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(xGen, originData)).item()
        print(f"Similarity: {similarity}")
        print(f"euclidean_dist: {euclidean_dist}")

        print("################################ save data ############################")
        if args.save != '':
            if not os.path.exists(args.save):
                os.makedirs(args.save)

            record_experiment(args, acc, onehot_acc, num_acc, similarity, euclidean_dist)

            # # 保存元素数据
            # origin_data = tensor2df(originData.detach())
            # # 判断路径是否存在
            # origin_data.to_csv(args.save + "origin.csv", mode='a', header=False, index=False)
            #
            # inverse_data = tensor2df(xGen.detach())
            # inverse_data.to_csv(args.save + "inverse.csv", mode='a', header=False, index=False)


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
    parser.add_argument('--c', type=str, default='./configs/attack/nomodel/adult_base.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    freeze_rand(args.seed)

    # # 这个是一个类似tensorboard的东西,可视化实验过程
    # wandb.init(project="VFL-TabRebuild-v3", entity="yang-test",
    #            name="VFL-{}".format(args.name),
    #            config=args)

    # 是否要规范化
    train, test = preprocess(args.data_dir)

    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    # Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    # Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]

    # 指定rebuild的表格特征
    tab = {
        'boolList': [i for i in range(0, 76)],
        'onehot': {
            'education': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'occupation': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            'race': [30, 31, 32, 33, 34],
            'native-country': [
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
        },
        'numList': [76]
    }

    # 训练并生成
    # 白盒攻击本身并不需要训练数据
    rebuild(train_data=test, tab=tab, device=device, args=args)