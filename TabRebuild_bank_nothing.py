import logging
import os
import random
import sys
import pandas as pd

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
# 加入模块的搜索路径

from fedml_core.preprocess.bank.preprocess_bank import preprocess
from fedml_core.model.bankModels import TopModel, BottomModel, BottomModelDecoder
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import adult_dataset, over_write_args_from_file, Similarity, onehot_softmax, tabRebuildAcc, onehot_bool_loss_v2, onehot_bool_loss, num_loss, test_rebuild_acc

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import wandb
import shutil


def train_decoder(net, train_queue, test_queue, device, args):
    Xb_train = train_queue.dataset.Xb
    print("################################ Set Federated Models, optimizer, loss ############################")

    net_output = net(torch.zeros_like(next(iter(train_queue))[0][1]).to(device))
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
    epoch = 0
    bestAcc = 0
    consecutive_decreases = 0
    while True:
        # train and update
        epoch_loss = []
        for step, (trn_X, trn_y) in enumerate(train_queue):
            if step == 1:
                trn_X = [x.float().to(device) for x in trn_X]

                optimizer.zero_grad()
                out = decoder(net(trn_X[1]))
                loss = criterion(out, trn_X[1])
                loss.backward()

                optimizer.step()
                print("loss:", loss.item())

        epoch += 1

        if epoch % 10 == 0:
            acc, onehot_acc, num_acc, similarity, euclidean_dist = test_rebuild_acc(train_queue, net, decoder, tab,
                                                                                    device, args)
            print(
                f"acc: {acc}, onehot_acc: {onehot_acc}, num_acc: {num_acc}, similarity: {similarity}, euclidean_dist: {euclidean_dist}")

            if acc >= bestAcc:
                if abs(acc-bestAcc) < 0.0001:
                    # 检查args.decoder_mode目录是否存在
                    save_dir = os.path.dirname(args.decoder_mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    torch.save(decoder, args.decoder_mode)
                    break
                consecutive_decreases = 0
                bestAcc = acc
                # 检查args.decoder_mode目录是否存在
                save_dir = os.path.dirname(args.decoder_mode)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(decoder, args.decoder_mode)
            else:
                consecutive_decreases += 1

            if consecutive_decreases >= 2:
                break

    return decoder


    # vfltrainer.save_model('/data/yangjirui/vfl-tab-reconstruction/model/adult/', 'final.pth.tar')


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




def rebuild(train_data, test_data, tab, device, args):
    print("hyper-parameters:")
    print("iteration optimization times: {0}".format(args.NIters))
    print("bloss2 rate: {0}".format(args.bloss2))
    print("bloss2_v2 rate: {0}".format(args.bloss2_v2))
    print("numloss rate: {0}".format(args.numloss))
    print("norloss rate: {0}".format(args.norloss))

    print("################################ load Federated Models ############################")

    # 加载原始训练数据，用于对比恢复效果
    Xa_train, Xb_train, y_train = train_data
    train_dataset = adult_dataset(train_data)
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, drop_last=False)
    test_dataset = adult_dataset(test_data)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, drop_last=False)

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

    net = vfltrainer.bottom_model_list[1].to(device)  # 需要恢复数据的网络

    decoder = train_decoder(net, train_queue, test_queue, device, args)

    print("################################ recovery data ############################")

    # for i in range(args.Ndata):
    acc_list = []
    onehot_acc_list = []
    num_acc_list = []
    similarity_list = []
    euclidean_dist_list = []

    #  最后测试重建准确率需要在训练集上进行
    for trn_X, trn_y in tqdm(train_queue):
        # (trn_X, trn_y) = next(train_queue)
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
    df.to_csv(args.save + "record_experiment_decoder_weak.csv", mode='a', header=False, index=False)





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
    parser.add_argument('--c', type=str, default='./configs/attack/bank/nothing.yml', help='config file')

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
        # shadow_model = args.shadow_model

        for i in range(5):
            # 设置随机种子
            freeze_rand(args.seed + i)
            # 是否要规范化
            args.decoder_mode = decoder_mode + str(i)
            # args.shadow_model = shadow_model + str(i)

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
        # 测试已知数据大小
        # args.batch_size = 8

        # 训练并生成
        # 白盒攻击本身并不需要训练数据
        rebuild(train_data=train, test_data=test, tab=tab, device=device, args=args)