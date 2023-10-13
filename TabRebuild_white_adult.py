import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fedml_core.preprocess.adult.preprocess_adult import preprocess
from fedml_core.utils.vfl_trainer import VFLTrainer
from fedml_core.model.net import active_model, passive_model
from fedml_core.utils.utils import tensor2df, Similarity, bool_loss, int_loss, neg_loss, normalize_loss, tabRebuildAcc
from fedml_core.utils.utils import adult_dataset, over_write_args_from_file, num_loss, onehot_softmax, onehot_bool_loss, onehot_bool_loss_v2

import torch
import pandas as pd

import argparse




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

    net = vfltrainer.passive_model_list[0].to(device)  # 需要恢复数据的网络

    print("################################ recovery data ############################")

    for i in range(args.Ndata):
        (trn_X, trn_y) = next(train_queue)
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone().detach()

        ture_logits = vfltrainer.active_model(trn_X[0], [protocolData]).clone().detach()

        # print("originData:", originData)
        # print("originData.size:", originData.size())

        onehot_index = tab['onehot']
        # originData = onehot_softmax(originData, onehot_index)


        # xGen = torch.zeros(originData.size()).to(device)
        xGen_before = torch.randn(originData.size()).to(device)
        xGen_before.requires_grad = True



        optimizer = torch.optim.Adam(params=[xGen_before], lr=args.lr, eps=args.eps, amsgrad=True)

        for j in range(args.NIters):  # 迭代优化
            optimizer.zero_grad()

            xGen = onehot_softmax(xGen_before, onehot_index)
            xProtocolData = net.forward(xGen)
            featureLoss = ((xProtocolData - protocolData) ** 2).mean()  # 欧几里得距离 损失函数

            # 计算model loss  这里借用iloss
            logits = vfltrainer.active_model(trn_X[0], [xProtocolData])
            modelLoss = ((ture_logits - logits) ** 2).mean()

            # 计算loss
            # bloss = bool_loss(xGen, tab['boolList'])
            numloss = num_loss(xGen, tab['numList'])
            bloss2 = onehot_bool_loss(xGen_before, tab['onehot'], tab['boolList'])
            bloss2_v2 = onehot_bool_loss_v2(xGen_before, tab['onehot'], tab['boolList'])
            # iloss = int_loss(xGen)
            # nloss = neg_loss(xGen)
            # norloss = normalize_loss(xGen)

            totalLoss = (featureLoss + args.numloss * numloss + args.bloss2_v2*bloss2_v2+ args.bloss2*bloss2 + args.iloss * modelLoss)
                         # + args.bloss * bloss + args.nloss * nloss + args.norloss * norloss)
            totalLoss.backward(retain_graph=True)
            optimizer.step()

            # similarity = Similarity(xGen, originData)  # 余弦相似度
            # euclidean_dist = torch.nn.functional.pairwise_distance(xGen, originData)  # 现在换用欧几里得距离

            # wandb.log({"totalLoss_" + str(i): totalLoss})
            # wandb.log({"featureLoss_" + str(i): featureLoss})
            # wandb.log({"iloss_" + str(i): iloss})
            # wandb.log({"bloss_" + str(i): bloss})
            # wandb.log({"nloss_" + str(i): nloss})
            # wandb.log({"norloss_" + str(i): norloss})
            # wandb.log({"euclidean_dist_" + str(i): euclidean_dist})
            # wandb.log({"similarity_" + str(i): similarity})

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
    df.to_csv(args.save + "record_experiment.csv", mode='a', header=False, index=False)





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
    parser.add_argument('--name', type=str, default='vfl_TabRebuild+norloss3', help='experiment name')
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
    # parser.add_argument('--inverse_data_output', default="/home/yangjirui/paper-code/data/adult/2layer-noloss/base-random/inverse_data.csv",
    #                     help='location of the data corpus')
    parser.add_argument('--base_mode', default='', type=str, metavar='PATH',
                        help='path to latest base mode (default: none)')
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
    parser.add_argument('--c', type=str, default='./configs/attack/whitebox/adult_base.yml', help='config file')

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
    rebuild(train_data=train, tab=tab, device=device, args=args)

# 超参数的最佳记录
# --zhognyuan 规范化
# 不使用norlosss 最佳
