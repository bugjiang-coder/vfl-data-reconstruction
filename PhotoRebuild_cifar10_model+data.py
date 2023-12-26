import logging
import os
import random
import sys
import pandas as pd

import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
# 加入模块的搜索路径
sys.path.append("/data/yangjirui/vfl/vfl-tab-reconstruction")

from fedml_core.preprocess.cifar10.preprocess_cifar10 import IndexedCIFAR10
from fedml_core.model.cifar10Models import BottomModelForCifar10, TopModelForCifar10, CIFAR10CNNDecoder
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import over_write_args_from_file, Similarity, tabRebuildAcc, test_rebuild_psnr

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import wandb
import shutil


def train_decoder(net, train_queue, test_queue, device, args):
    # 注意这个decoder 需要使用测试集进行训练

    # Xb_shape = train_queue.dataset.Xb.shape

    print("################################ Set Federated Models, optimizer, loss ############################")

    # net_output = net(torch.zeros_like(next(iter(train_queue))[0][1]).to(device))
    # print(net_output.shape)
    decoder = CIFAR10CNNDecoder().to(device)

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


    # for epoch in range(0, 360):
    epoch = 0
    bestPsnr = 0
    consecutive_decreases = 0
    while True:
        # train and update
        epoch_loss = []
        for step, (trn_X, trn_y) in enumerate(test_queue):
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

        epoch += 1
        print("--- epoch: {0}, train_loss: {1}".format(epoch, epoch_loss))

        if epoch % 10 == 0:
            # acc, onehot_acc, num_acc, similarity, euclidean_dist = test_rebuild_acc(train_queue, net, decoder, tab,
            #                                                                         device, args)
            
            psnr, euclidean_dist = test_rebuild_psnr(train_queue, net, decoder, device, args)
            print(
                f"psnr: {psnr}, euclidean_dist: {euclidean_dist}")

            if psnr >= bestPsnr:
                if abs(psnr-bestPsnr) < 0.0001:
                    # 检查args.decoder_mode目录是否存在
                    save_dir = os.path.dirname(args.decoder_mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    torch.save(decoder, args.decoder_mode)
                    break

                consecutive_decreases = 0
                bestPsnr = psnr
                # 检查args.decoder_mode目录是否存在
                save_dir = os.path.dirname(args.decoder_mode)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(decoder, args.decoder_mode)


            else:
                consecutive_decreases += 1

            if consecutive_decreases >= 2:
                break



    print("model saved")
    return decoder


    # vfltrainer.save_model('/data/yangjirui/vfl-tab-reconstruction/model/adult/', 'final.pth.tar')


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




def rebuild(train_data, test_data, device, args):
    print("################################ load Federated Models ############################")

    # 加载原始训练数据，用于对比恢复效果
    # Xa_train, Xb_train, y_train = train_data
    # train_dataset = adult_dataset(train_data)
    # train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                           num_workers=args.workers, drop_last=False)
    # test_dataset = adult_dataset(test_data)
    # test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                             num_workers=args.workers, drop_last=False)
    
    train_queue = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers
        )
    test_queue = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers
        )

    # 加载VFL框架
    top_model = TopModelForCifar10()
    bottom_model_list = [BottomModelForCifar10(),
                         BottomModelForCifar10()]

    vfltrainer = VFLTrainer(top_model, bottom_model_list, args)


    checkpoint = torch.load(args.base_mode, map_location=device)
    args.start_epoch = checkpoint['epoch']
    vfltrainer.load_model(args.base_mode, device)
    print("=> loaded model '{}' (epoch: {} test_top1_acc: {})"
          .format(args.base_mode, checkpoint['epoch'], checkpoint['auc']))

    net = vfltrainer.bottom_model_list[1].to(device)  # 需要恢复数据的网络

    decoder = train_decoder(net, train_queue, test_queue, device, args)

    print("################################ recovery data ############################")

    # acc, onehot_acc, num_acc, similarity, euclidean_dist = test_rebuild_acc(train_queue, net, decoder, tab, device, args)
    # for i in range(args.Ndata):
    #     (trn_X, trn_y) = next(train_queue)
    acc_list = []
    onehot_acc_list = []
    num_acc_list = []
    similarity_list = []
    euclidean_dist_list = []

    #  最后测试重建准确率需要在训练集上进行
    for trn_X, trn_y in tqdm(train_queue):
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone().detach()

        xGen = decoder(protocolData)


        # onehot_index = tab['onehot']
        # originData = onehot_softmax(originData, onehot_index)

        # xGen = onehot_softmax(xGen_before, onehot_index)

        # # 生成随机数 作为一个基准测试
        # xGen = torch.rand_like(xGen)
        # # 将随机数张量进行线性变换，映射到 -1 到 1 的范围
        # xGen = 2 * xGen - 1

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

        record_experiment(args, acc, onehot_acc, num_acc, similarity, euclidean_dist)

            # # 保存元素数据
            # origin_data = tensor2df(originData.detach())
            # # 判断路径是否存在
            # origin_data.to_csv(args.save + "origin.csv", mode='a', header=False, index=False)
            #
            # inverse_data = tensor2df(xGen.detach())
            # inverse_data.to_csv(args.save + "inverse.csv", mode='a', header=False, index=False)

    return acc, onehot_acc, num_acc, similarity, euclidean_dist





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
    parser.add_argument('--name', type=str, default='decoder-rebuild-2layer-all-data', help='experiment name')
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
    parser.add_argument('--c', type=str, default='./configs/attack/cifar10/model+data.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    
    train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2471, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # Load CIFAR-10 dataset
    trainset = IndexedCIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    testset = IndexedCIFAR10(root=args.data_dir, train=False, download=True, transform=train_transform)





    freeze_rand(args.seed)
        # 是否要规范化

        # 训练并生成
        # 白盒攻击本身并不需要训练数据
    rebuild(train_data=trainset, test_data=testset, device=device, args=args)
