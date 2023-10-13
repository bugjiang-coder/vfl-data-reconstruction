import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.preprocess.zhongyuan.preprocess_zhongyuan import preprocess_zhongyuan
from fedml_core.utils.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import zhongyuan_dataset, tensor2df, Similarity, bool_loss, int_loss, neg_loss, normalize_loss

import torch
import argparse


def initData(dataShape, granularity, dataRange, indices=None):
    # 更具indices的列表，生产初始化好的xGen值。
    # 返回最优的初始值列表
    # 注意indices的格式是[beam_size,维度]
    # eg:[[0], [1]]

    # 空数据 用于恢复数据
    xGen = torch.zeros(dataShape, requires_grad=False)
    if indices == None:
        return [xGen]
    else:
        xGenList = [xGen.clone() for i in range(len(indices))]
        # print(xGenList)
        for i, index in enumerate(indices):
            for j, setnum in enumerate(index):
                if dataRange:
                    x = dataRange[j]*(setnum / granularity)
                else:
                    # 规范化的数据，遵循3sigma原则
                    x = 2 * (setnum / granularity) - 1
                xGenList[i][:, j] = x
    return xGenList


def MSELOSS(net, NIters, xGen, refFeature, learningRate, eps):
    optimizer = torch.optim.Adam(params=[xGen], lr=learningRate, eps=eps, amsgrad=True)
    for j in range(NIters):  # 迭代优化
        optimizer.zero_grad()
        xFeature = net.forward(xGen)
        loss = ((xFeature - refFeature) ** 2).mean() # 欧几里得距离 损失函数
        loss.backward(retain_graph=True)
        optimizer.step()
    return loss

def find_best_initial_value(originDataShape, net, protocolData, dataRange, device, args):
    # step 取得向量的维度，以此获得搜索的次数
    step = originDataShape[-1]

    # TODO:注意这里批次大小为1

    # 最优初始值的索引
    indices = None

    # 注意所有的数据的范围是不一致的，所以初始化也不一样。
    for i in range(step):
        allLoss = None
        for j in range(args.granularity):
            xGenList = initData(originDataShape, args.granularity, dataRange, indices)
            for xGen in xGenList:
                if dataRange:
                    x = dataRange[i] * (j / args.granularity)
                else:
                    # 规范化的数据，遵守3sigma原则
                    x = 2 * (j / args.granularity) - 1
                xGen[:, i] = x
                xGen = xGen.to(device)
                #  一定要修改完成后再设置需要梯度  设置使用梯度
                xGen.requires_grad = True

                loss = MSELOSS(net, int(args.NIters / 500), xGen, protocolData, args.lr, args.eps)
                if allLoss == None:
                    allLoss = loss.unsqueeze(-1)
                else:
                    allLoss = torch.cat([allLoss, loss.unsqueeze(-1)], dim=-1)
        # 找到loss最小的beam_size个
        _, topIndices = allLoss.topk(args.beam_size, largest=False)


        if indices == None:
            topIndices = topIndices.cpu().detach().numpy().tolist()
            indices = []
            for num in range(args.beam_size):
                indices.append([topIndices[num]])
        else:
            # 按照10粒度*5束 里面选择最优的5个
            # 同时要更新这5个里面的最优路径
            totopIndices = topIndices.cpu().detach().numpy()
            row = (totopIndices % args.beam_size).tolist()
            index = (totopIndices // args.beam_size).tolist()
            new_indices = []
            for num in range(args.beam_size):
                row_i = indices[row[num]].copy()
                row_i.append(index[num])
                new_indices.append(row_i)
            indices = new_indices

    return indices


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
    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, drop_last=False)
    train_queue = iter(train_queue)

    # 加载VFL框架
    vfltrainer = VFLTrainer()
    vfltrainer.load_model(models)
    net = vfltrainer.model[1].to(device)   # 需要恢复数据的网络

    print("################################ recovery data ############################")


    for i in range(args.Ndata):
        (trn_X, trn_y) = train_queue.next()
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        protocolData = net.forward(originData).clone()

        print("originData:",originData)

        # --------------------------------

        indices = find_best_initial_value(originDataShape=originData.size(), net=net, protocolData=protocolData,dataRange=tab['range'], device=device, args=args)

        # 获得最优的初始值，这里初始的xGen为0
        xGenList = initData(originData.size(), args.granularity, tab['range'], indices)
        xGen = xGenList[0].to(device)
        xGen.requires_grad = True

        # --------------------------------
        optimizer = torch.optim.Adam(params=[xGen], lr=args.lr, eps=args.eps, amsgrad=True)

        for j in range(args.NIters):  # 迭代优化
            optimizer.zero_grad()

            xProtocolData = net.forward(xGen)
            featureLoss = ((xProtocolData - protocolData) ** 2).mean() # 欧几里得距离 损失函数

            # 计算loss
            bloss = bool_loss(xGen,tab['boolList'])
            iloss = int_loss(xGen,tab['intList'])
            nloss = neg_loss(xGen)
            norloss = normalize_loss(xGen)

            totalLoss = featureLoss + args.iloss * iloss + args.bloss * bloss + args.nloss * nloss + args.norloss * norloss
            totalLoss.backward(retain_graph=True)
            optimizer.step()

            similarity = Similarity(xGen, originData) # 余弦相似度
            euclidean_dist = torch.nn.functional.pairwise_distance(xGen, originData) # 现在换用欧几里得距离

            # wandb.log({"totalLoss_" + str(i): totalLoss})
            # wandb.log({"featureLoss_" + str(i): featureLoss})
            # wandb.log({"iloss_" + str(i): iloss})
            # wandb.log({"bloss_" + str(i): bloss})
            # wandb.log({"nloss_" + str(i): nloss})
            # wandb.log({"norloss_" + str(i): norloss})
            # wandb.log({"euclidean_dist_" + str(i): euclidean_dist})
            # wandb.log({"similarity_" + str(i): similarity})


        similarity = Similarity(xGen, originData)
        euclidean_dist = torch.nn.functional.pairwise_distance(xGen, originData)

        print("xGen:", xGen)
        print(f"Similarity: {similarity}")
        print(f"euclidean_dist: {euclidean_dist}")
        
        print("################################ save data ############################")
        if args.origin_data_output:
            # 保存元素数据
            origin_data = tensor2df(originData.detach())
            origin_data.to_csv(args.origin_data_output, mode='a', header=False, index=False)
        if args.inverse_data_output:
            inverse_data = tensor2df(xGen.detach())
            inverse_data.to_csv(args.inverse_data_output, mode='a', header=False, index=False)



if __name__ == '__main__':
    print("################################ prepare Data ############################")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("TabRebuild")
    parser.add_argument('--name', type=str, default='VFL-beam-search-sigma-beam-loss', help='experiment name')
    parser.add_argument('--data_dir', default="/home/yangjirui/feature-infer-workspace/dataset/zhongyuan/",
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--origin_data_output', default="/home/yangjirui/paper-code/data/zhongyuan/2layer/bs/origin_data.csv",
                        help='location of the data corpus')
    parser.add_argument('--inverse_data_output', default="/home/yangjirui/paper-code/data/zhongyuan/2layer/bs/inverse_data.csv",
                        help='location of the data corpus')

    # ==========下面是几个重要的超参数==========
    parser.add_argument('--NIters', type=int, default=5000, help="Number of times to optimize")
    parser.add_argument('--Ndata', type=int, default=100, help="Recovery data quantity")
    parser.add_argument('--iloss', type=float, default=0.1, help="Recovery data int loss intensity")
    parser.add_argument('--bloss', type=float, default=0.01, help="Recovery data boolean loss intensity")
    parser.add_argument('--nloss', type=float, default=1, help="Recovery data negative number loss intensity")
    parser.add_argument('--granularity', type=int, default=10, help="Beam search granularity")
    parser.add_argument('--beam_size', type=int, default=5, help="Beam search beam size (granularity>=beam_size)")
    parser.add_argument('--norloss', type=float, default=0, help="Recovery data negative number loss intensity")



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
    # wandb.init(project="VFL-TabRebuild-v3", entity="yang-test",
    #            name="VFL-{}".format(args.name),
    #            config=args)

    # 是否要规范化
    train, test = preprocess_zhongyuan(args.data_dir,normalize=False)
    # train, test = preprocess_zhongyuan(args.data_dir, normalize=True)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]

    base_path = '/home/yangjirui/paper-code/model/zhongyuan/'
    # models = [
    #     base_path + '2layer_normal_0.pth',
    #     base_path + '2layer_normal_1.pth']
    models = [
        base_path + '2layer_not_normal_0.pth',
        base_path + '2layer_not_normal_1.pth']
    # models = [
    #     base_path + '3layer_normal_0.pth',
    #     base_path + '3layer_normal_1.pth']
    # models = [
    #     base_path + '3layer_not_normal_0.pth',
    #     base_path + '3layer_not_normal_1.pth']
    # models = [
    #     base_path + '4layer_normal_0.pth',
    #     base_path + '4layer_normal_1.pth']
    # models = [
    #     base_path + '4layer_not_normal_0.pth',
    #     base_path + '4layer_not_normal_1.pth'
    # ]
    # 非规范化模型
    # models = ['/home/yangjirui/paper-code/model/zhongyuan/not-normalization_0.pth',
    #           '/home/yangjirui/paper-code/model/zhongyuan/not-normalization_1.pth']
    # 非规范化简化模型
    # models = ['/home/yangjirui/paper-code/model/zhongyuan/not-normalization-2layers_0.pth',
    #           '/home/yangjirui/paper-code/model/zhongyuan/not-normalization-2layers_1.pth']

    # 指定rebuild的表格特征
    # tab = {
    #     'boolList':[7, 15],
    #     'intList':[1, 4, 5, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20],
    #     'range':[1000,20,1000,1000,100,10,10,1,10,10,10,3000,500000,200,40,1,100,100,100,10,20000,6000]
    # }
    # 规范化后使用下面的表格初始化
    tab = {
        'boolList':[7, 15],
        'intList':[1, 4, 5, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20],
        'range':None
    }

    # 训练并生成
    # 白盒攻击本身并不需要训练数据
    rebuild(train_data=train, models=models, tab=tab, device=device, args=args)
