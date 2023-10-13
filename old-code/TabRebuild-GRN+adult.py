import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.preprocess.adult.preprocess_adult import preprocess
from fedml_core.utils.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import zhongyuan_dataset, tensor2df, Similarity, bool_loss, int_loss, neg_loss, normalize_loss
from fedml_core.model.net import generator_model

import torch
import argparse


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

        localData = trn_X[0]
        originData = trn_X[1]
        protocolData = net.forward(originData).clone()

        print("originData:",originData)

        # initdata = torch.randn(originData.size()).to(device)
        initdata = torch.randn(originData.size()).to(device)
        # initdata2 = torch.randn(localData.size()).to(device)
        inputEntry = torch.cat((localData, initdata), dim=1)
        # inputEntry = torch.cat((initdata2, initdata), dim=1)
        # inputEntry = initdata

        # 构建生成模型
        GRN = generator_model(input_dim=inputEntry.shape[-1],intern_dim=inputEntry.shape[-1],output_dim=originData.shape[-1]).to(device)


        optimizer = torch.optim.Adam(params=GRN.parameters(), lr=args.lr, eps=args.eps, amsgrad=True)

        for j in range(args.NIters):  # 迭代优化

            # if j%100==0:
            #     print(inputEntry)
            # initdata = torch.zeros(originData.size()).to(device)
            # initdata2 = torch.zeros(localData.size()).to(device)
            # inputEntry = torch.cat((initdata2, initdata), dim=1)
            # inputEntry = torch.cat((localData, initdata), dim=1)

            optimizer.zero_grad()

            xGen = GRN(inputEntry)
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
    parser.add_argument('--name', type=str, default='GRN+noloss', help='experiment name')
    parser.add_argument('--data_dir', default='/home/yangjirui/VFL/feature-infer-workspace/dataset/adult/adult.data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--origin_data_output', default="/home/yangjirui/paper-code/data/adult/2layer-enhance2/GRN/origin_data.csv",
                        help='location of the data corpus')
    parser.add_argument('--inverse_data_output', default="/home/yangjirui/paper-code/data/adult/2layer-enhance2/GRN/inverse_data.csv",
                        help='location of the data corpus')
    # parser.add_argument('--origin_data_output',default="",help='location of the data corpus')
    # parser.add_argument('--inverse_data_output', default="", help='location of the data corpus')
    # ==========下面是几个重要的超参数==========
    parser.add_argument('--NIters', type=int, default=5000, help="Number of times to optimize")
    parser.add_argument('--Ndata', type=int, default=100, help="Recovery data quantity")
    parser.add_argument('--iloss', type=float, default=0.1, help="Recovery data int loss intensity")
    parser.add_argument('--bloss', type=float, default=0.01, help="Recovery data boolean loss intensity")
    parser.add_argument('--nloss', type=float, default=1, help="Recovery data negative number loss intensity")
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

    train, test = preprocess(args.data_dir)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]

    base_path = '/home/yangjirui/paper-code/model/adult/'
    models = [
        base_path + '2layers-enhance2_0.pth',
        base_path + '2layers-enhance2_1.pth']
    # models = [
    #     base_path + '3layers-enhance2_0.pth',
    #     base_path + '3layers-enhance2_1.pth']
    # models = [
    #     base_path + '4layers-enhance2_0.pth',
    #     base_path + '4layers-enhance2_1.pth']


    # 指定rebuild的表格特征
    tab = {
        'boolList':[i for i in range(1, 51)],
        'intList':[0]
    }

    # 训练并生成
    # 白盒攻击本身并不需要训练数据
    rebuild(train_data=train, models=models, tab=tab, device=device, args=args)


# 超参数的最佳记录
# --zhognyuan 规范化
# 不使用norlosss 最佳
# 为什么拼上另外一半的数据，loss可以下降，我很不理解，这是为什么？
# 如果是随机的呢？