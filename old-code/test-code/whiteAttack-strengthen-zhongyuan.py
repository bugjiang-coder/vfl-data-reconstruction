import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from preprocess.zhongyuan.preprocess_zhongyuan import preprocess_zhongyuan
from net import active_model, passive_model
from vfl_trainer import VFLTrainer
from utils import zhongyuan_dataset, Similarity

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import time
import glob
import wandb
import shutil



# 自己定义损失函数
def custom_loss(input):
    # 取 output 的小数部分

    output_frac = torch.abs(torch.frac(input))
    # print("output_frac",output_frac)

    loss = torch.sum(output_frac)
    return loss

def bool_loss(input):

    boolList = [1, 5, 6, 7]

    boolColumn = input[:, boolList]

    return torch.sum(torch.abs(boolColumn - 0.5))
def int_loss(input):
    intList = [1, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20]

    intColumn = input[:, intList]
    # print(intColumn)
    return torch.sum(torch.abs(torch.frac(intColumn)))

def zhongyuan_loss(input):
    # 第1列：
    # column = input[:, 0] # 注意这样取出来是横着的input[:,0:1]取出来的才是竖着的

    allLoss=0

    boolList = [1,5,6,7]

    boolColumn = input[:,boolList]
    allLoss += torch.sum(torch.abs(boolColumn-0.5))

    intList = [1,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20]

    intColumn = input[:,intList]
    allLoss += torch.sum(torch.abs(torch.frac(intColumn)))

    # print(allLoss)
    return allLoss

def inverse(train_data, test_data, device, args):
    print("hyper-parameters:")
    print("iteration optimization times: {0}".format(args.NIters))
    print("learning rate: {0}".format(args.lr))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    print("################################ Wire Federated Models ############################")

    # dataloader
    train_dataset = zhongyuan_dataset(train_data)

    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)

    vfltrainer = VFLTrainer()

    # 规范化模型
    # models = ['/home/yangjirui/paper-code/model/zhongyuan/0_.pth', '/home/yangjirui/paper-code/model/zhongyuan/1_.pth']
    # 非规范化模型
    models = ['/home/yangjirui/paper-code/model/zhongyuan/not-normalization_0.pth', '/home/yangjirui/paper-code/model/zhongyuan/not-normalization_1.pth']
    vfltrainer.load_model(models)

    print("################################ recovery data ############################")
    net = vfltrainer.model[1].to(device)

    train_queue = iter(train_queue)

    for i in range(3):
        (trn_X, trn_y) = train_queue.next()
        trn_X = [x.float().to(device) for x in trn_X]

        targetEntry = trn_X[1]
        refFeature = net.forward(targetEntry).clone()

        print("targetEntry:",targetEntry)


        xGen = torch.zeros(targetEntry.size()).to(device)
        xGen.requires_grad = True

        optimizer = torch.optim.Adam(params=[xGen], lr=args.lr, eps=args.eps, amsgrad=True)

        for j in range(args.NIters):  # 迭代优化
            optimizer.zero_grad()

            xFeature = net.forward(xGen)
            # 欧几里得距离 损失函数
            featureLoss = ((xFeature - refFeature) ** 2).mean()
            # 余弦相似度,这个和欧几里得距离高度相关，有一个就可以
            # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            # cosLoss = (1 - cos(xFeature, refFeature))

            # floss = zhongyuan_loss(xGen)
            floss = custom_loss(xGen)
            # floss = int_loss(xGen)
            bloss = bool_loss(xGen)
            # int_loss

            totalLoss = featureLoss  + 0.1*floss + 0.01*bloss# + cosLoss
            wandb.log({"featureLoss_" + str(i): featureLoss})
            wandb.log({"floss_" + str(i): floss})


            totalLoss.backward(retain_graph=True)
            optimizer.step()

            similarity = Similarity(xGen, targetEntry)
            # 现在换用欧几里得距离
            euclidean_dist = torch.nn.functional.pairwise_distance(xGen, targetEntry)
            wandb.log({"euclidean_dist_" + str(i): euclidean_dist})
            wandb.log({"similarity_" + str(i): similarity})

        similarity = Similarity(xGen, targetEntry)
        euclidean_dist = torch.nn.functional.pairwise_distance(xGen, targetEntry)

        print("xGen:", xGen)

        print(f"Similarity: {similarity}")
        print(f"euclidean_dist: {euclidean_dist}")


if __name__ == '__main__':
    print("################################ Prepare Data ############################")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("vflmodelnet")
    parser.add_argument('--data_dir', default="/home/yangjirui/feature-infer-workspace/dataset/zhongyuan/",
                        help='location of the data corpus')
    parser.add_argument('--name', type=str, default='vfl_whiteAttack_strengthen', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--layers', type=int, default=18, help='total number of layers')
    parser.add_argument('--layer', type=str, default='linear4', help='Target layer for recovery')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--NIters', type=int, default=1000, help="Number of times to optimize")
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
    parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--record', type=int, default=0, help='num of client')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--encoder', default='./model/VIME/checkpoint_0095.pth.tar', type=str, metavar='PATH',
                        help='path to trained encoder checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    logger.info(device)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # # 这个是一个类似tensorboard的东西,可视化实验过程
    wandb.init(project="VFL-project", entity="yang-test",
               name="VFL-{}".format(args.name),
               config=args)

    # 保存生成数据的位置
    save_inverted_data_dir = "/home/yangjirui/feature-infer-workspace/inverse_data/zhongyuan/whitebox/"

    # edge网络存放位置
    model_dir = "/home/yangjirui/feature-infer-workspace/train_model/edge_network/"
    model_name = "splitnn-avazu.pth"

    train, test = preprocess_zhongyuan(args.data_dir,normalize=False)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]

    # 训练并生成
    inverse(train_data=train, test_data=test, device=device, args=args)
    # run_experiment(train_data=train, test_data=test, device=device, args=args)

    # reference training result:
    # --- epoch: 99, batch: 1547, loss: 0.11550658332804839, acc: 0.9359105089400196, auc: 0.8736984159409958
    # --- (0.9270889578726378, 0.5111934752243287, 0.5054099033579607, None)

    # --- epoch: 99, batch: 200, loss: 0.09191526211798191, acc: 0.9636565918783608, auc: 0.9552342451916291
    # --- (0.9754657898538487, 0.7605652456769234, 0.8317858679682943, None)
