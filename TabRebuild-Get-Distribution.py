import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fedml_core.preprocess.zhongyuan.preprocess_zhongyuan import preprocess_zhongyuan
from fedml_core.utils.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import zhongyuan_dataset, tensor2df, Similarity
from fedml_core.model.net import generator_model

import torch
import argparse
from tqdm import tqdm


def buildDGM(train_data, net, device, args):
    if args.DGM_model:
        return torch.load(args.DGM_model)

    # 构建生成模型
    train_data_example = iter(train_data)
    (trn_X, trn_y) = train_data_example.next()
    trn_X = [x.float().to(device) for x in trn_X]

    localData = trn_X[0]
    originData = trn_X[1]
    protocolData = net.forward(originData).clone()

    inputEntry = torch.cat((localData, protocolData), dim=1)

    DMG = generator_model(input_dim=inputEntry.shape[-1], intern_dim=inputEntry.shape[-1],
                          output_dim=originData.shape[-1]).to(device)
    optimizer = torch.optim.Adam(params=DMG.parameters(), lr=args.lr, eps=args.eps, amsgrad=True)

    for epoch in range(args.epochs):
        DMG.train()
        epochLoss = 0
        for trn_X, trn_y in train_data:
            # 准备数据
            trn_X = [x.float().to(device) for x in trn_X]
            localData = trn_X[0]
            originData = trn_X[1]
            protocolData = net.forward(originData).clone()
            inputEntry = torch.cat((localData, protocolData), dim=1)


            optimizer.zero_grad()

            # 生成数据
            xGen = DMG.forward(inputEntry)
            xProtocolData = net.forward(xGen)

            # 计算损失
            loss = ((xProtocolData - protocolData) ** 2).mean() # 欧几里得距离 损失函数

            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            # print("epoch: {0} | loss: {1}".format(epoch, loss.item()))

        print("epoch: {0} | loss: {1}".format(epoch, epochLoss/len(train_data)))

    # # 保存模型
    # if args.DGM_model is None:
    #     # DMG.load_state_dict(torch.load(args.DGM_model))
    torch.save(DMG, "/home/yangjirui/paper-code/model/zhongyuan/DMG.pth")
    print("save DMG model")

    return DMG

def rebuild(train_data, models, tab, device, args):
    print("hyper-parameters:")
    print("iteration optimization times: {0}".format(args.NIters))
    print("iloss rate: {0}".format(args.iloss))
    print("bloss rate: {0}".format(args.bloss))
    print("nloss rate: {0}".format(args.nloss))
    print("norloss rate: {0}".format(args.norloss))


    print("################################ Wire Federated Models ############################")

    # 加载原始训练数据，用于对比恢复效果
    train_dataset = zhongyuan_dataset(train_data)
    DGM_train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    # 恢复数据时使用的数据队列
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    # train_queue = iter(train_queue)

    # 加载VFL框架
    vfltrainer = VFLTrainer()
    vfltrainer.load_model(models)
    net = vfltrainer.model[1].to(device)   # 需要恢复数据的网络

    print("################################ recovery data ############################")

    DMG = buildDGM(DGM_train_queue, net, device, args)
    DMG.eval()
    similarity = 0
    euclidean_dist = 0
    for (trn_X, trn_y) in tqdm(train_queue):
        trn_X = [x.float().to(device) for x in trn_X]
        localData = trn_X[0]
        originData = trn_X[1]
        protocolData = net.forward(originData).clone()
        inputEntry = torch.cat((localData, protocolData), dim=1)
        xGen = DMG(inputEntry)

        similarity = Similarity(xGen, originData)
        # euclidean_dist = torch.nn.functional.pairwise_distance(xGen, originData)

        # print("xGen:", xGen)
        # 计算smiliarity的平均值

        similarity += similarity
        # euclidean_dist = euclidean_dist.item()

        # print(f"euclidean_dist: {euclidean_dist}")

        # print("################################ save data ############################")
        if args.origin_data_output:
            # 保存元素数据
            origin_data = tensor2df(originData.detach())
            origin_data.to_csv(args.origin_data_output, mode='a', header=False, index=False)
        if args.inverse_data_output:
            inverse_data = tensor2df(xGen.detach())
            inverse_data.to_csv(args.inverse_data_output, mode='a', header=False, index=False)
    print("################################ save data ############################")
    print(f"Similarity: {similarity/len(train_queue)}")



if __name__ == '__main__':
    print("################################ prepare Data ############################")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("TabRebuild")
    parser.add_argument('--name', type=str, default='DGM', help='experiment name')
    parser.add_argument('--data_dir', default="/home/yangjirui/feature-infer-workspace/dataset/zhongyuan/",
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--origin_data_output', default="/home/yangjirui/paper-code/data/zhongyuan/norm-2layer/DGM/origin_data-g3.csv",
                        help='location of the data corpus')
    parser.add_argument('--inverse_data_output', default="/home/yangjirui/paper-code/data/zhongyuan/norm-2layer/DGM/inverse_data-g3.csv",
                        help='location of the data corpus')
    parser.add_argument('--DGM_model', default="/home/yangjirui/paper-code/model/zhongyuan/DMG.pth",help='location of the DGM model')
    # parser.add_argument('--DGM_model', default="", help='location of the DGM model')
    # parser.add_argument('--origin_data_output',default="", help='location of the data corpus')
    # parser.add_argument('--inverse_data_output', default="", help='location of the data corpus')
    # ==========下面是几个重要的超参数==========
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    parser.add_argument('--NIters', type=int, default=5000, help="Number of times to optimize")
    parser.add_argument('--Ndata', type=int, default=100, help="Recovery data quantity")
    parser.add_argument('--iloss', type=float, default=0, help="Recovery data int loss intensity")
    parser.add_argument('--bloss', type=float, default=0, help="Recovery data boolean loss intensity")
    parser.add_argument('--nloss', type=float, default=0, help="Recovery data negative number loss intensity")
    parser.add_argument('--norloss', type=float, default=0.0001, help="Recovery data negative number loss intensity")


    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    logger.info(device)

    # 通过设置这些种子，我们可以在不同的计算机上运行同一份代码，
    # 并得到完全相同的随机数序列，这有助于确保结果的可重复性，方便我们进行模型调试和结果验证。
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # # 这个是一个类似tensorboard的东西,可视化实验过程
    # wandb.init(project="VFL-TabRebuild-v4", entity="yang-test",
    #            name="VFL-{}".format(args.name),
    #            config=args)

    # 是否要规范化
    # train, test = preprocess_zhongyuan(args.data_dir,normalize=False)
    train, test = preprocess_zhongyuan(args.data_dir, normalize=True)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]

    # 规范化模型
    # models = ['/home/yangjirui/paper-code/model/zhongyuan/0_.pth', '/home/yangjirui/paper-code/model/zhongyuan/1_.pth']
    # 规范化模型
    models = ['/home/yangjirui/paper-code/model/zhongyuan/2layers_0.pth', '/home/yangjirui/paper-code/model/zhongyuan/2layers_1.pth']


    # 非规范化模型
    # models = ['/home/yangjirui/paper-code/model/zhongyuan/not-normalization_0.pth',
    #           '/home/yangjirui/paper-code/model/zhongyuan/not-normalization_1.pth']
    # 非规范化简化模型
    # models = ['/home/yangjirui/paper-code/model/zhongyuan/not-normalization-2layers_0.pth',
    #           '/home/yangjirui/paper-code/model/zhongyuan/not-normalization-2layers_1.pth']

    # 指定rebuild的表格特征
    tab = {
        'boolList':[7, 15],
        'intList':[1, 4, 5, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20]
    }

    # 训练并生成
    # 白盒攻击本身并不需要训练数据
    rebuild(train_data=train, models=models, tab=tab, device=device, args=args)


# “双生成器生成模型（Dual-Generator Generative Model）”，
# 或简称为“双生成器模型（Dual-Generator Model）”。