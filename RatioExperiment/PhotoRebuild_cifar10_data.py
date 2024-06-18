import logging
import os
import random
import sys
import pandas as pd

import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import torchvision.transforms as transforms
from pytorch_msssim import ssim

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
# 加入模块的搜索路径
sys.path.append("/data/yangjirui/vfl/vfl-tab-reconstruction")

from fedml_core.preprocess.cifar10.preprocess_cifar10 import IndexedCIFAR10
from fedml_core.model.cifar10Models import BottomModelForCifar10, TopModelForCifar10, CIFAR10CNNDecoder
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import over_write_args_from_file, save_tensor_as_image, PSNR, test_rebuild_psnr, keep_predict_loss

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


    if os.path.isfile(args.shadow_model):
        print("=> loading decoder mode '{}'".format(args.shadow_model))
        shadow_model = torch.load(args.shadow_model, map_location=device)
        return shadow_model

    # 这里现实现一个完全一致的
    print("################################ load Federated Models ############################")
    # 加载VFL框架
    top_model = TopModelForCifar10(A_ratio=args.A_ratio)
    bottom_model_list = [BottomModelForCifar10(),
                         BottomModelForCifar10()]

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

    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss

    for epoch in range(0, 120):
        train_loss = vfltrainer.train_single_model(train_queue, criterion, bottom_criterion, optimizer_list, 1,False, device, args)

        test_loss, top1_acc, top5_acc = vfltrainer.test_mul(train_queue, criterion, device)
        print("--- epoch: {0}, train_loss: {1}, test_loss: {2}, test_top1_acc: {3}, test_top5_acc: {4}".format(epoch, train_loss, test_loss, top1_acc, top5_acc))
            

    shadow_model = vfltrainer.bottom_model_list[1]

    # 检查args.decoder_mode目录是否存在
    save_dir = os.path.dirname(args.shadow_model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(shadow_model, args.shadow_model)
    print("model saved")

    return shadow_model



def train_decoder(net, train_queue, test_queue, device, args):
    # 注意这个decoder 需要使用测试集进行训练


    print("################################ Set Federated Models, optimizer, loss ############################")

    # net_output = net(torch.zeros_like(next(iter(train_queue))[0][1]).to(device))
    # print(net_output.shape)
    decoder = CIFAR10CNNDecoder(A_ratio=args.A_ratio).to(device)


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


def rebuild(train_data, test_data, device, args):
    print("################################ load Federated Models ############################")

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
    top_model = TopModelForCifar10(A_ratio=args.A_ratio)
    bottom_model_list = [BottomModelForCifar10(),
                         BottomModelForCifar10()]

    vfltrainer = VFLTrainer(top_model, bottom_model_list, args)


    checkpoint = torch.load(args.base_mode, map_location=device)
    args.start_epoch = checkpoint['epoch']
    vfltrainer.load_model(args.base_mode, device)
    print("=> loaded model '{}' (epoch: {} test_top1_acc: {})"
          .format(args.base_mode, checkpoint['epoch'], checkpoint['auc']))

    # net = vfltrainer.bottom_model_list[1].to(device)  # 需要恢复数据的网络
    net = train_shadow_model(test_queue, device, args)  # 需要恢复数据的网络

    decoder = train_decoder(net, train_queue, test_queue, device, args)

    print("################################ recovery data ############################")

    psnr_list = []
    ssim_list = []
    euclidean_dist_list = []

    #  最后测试重建准确率需要在训练集上进行
    for trn_X, trn_y in tqdm(train_queue):
        trn_X = [x.float().to(device) for x in trn_X]

        originData = trn_X[1]
        
        # save_path = './image/original'
        # os.makedirs(save_path, exist_ok=True)
        # originDataAll = torch.cat((trn_X[0], trn_X[1]), dim=3)
        # for i, data in enumerate(originDataAll):
        #     save_tensor_as_image(data, os.path.join(save_path, f'origin_{i}.png'))
        # # print(trn_X[1].shape)
        # sys.exit(0)
        
        # save_path = './image/original-half'
        # os.makedirs(save_path, exist_ok=True)
        # for i, data in enumerate(originData):
        #     save_tensor_as_image(data, os.path.join(save_path, f'origin_{i}.png'))

        # sys.exit(0)
        protocolData = net.forward(originData).clone().detach()

        xGen = decoder(protocolData)
        
        # save_path = './image/data'
        # os.makedirs(save_path, exist_ok=True)
        # for i, data in enumerate(xGen):
        #     save_tensor_as_image(data, os.path.join(save_path, f'xGen_{i}.png'))

        # sys.exit(0)


        average_psnr = PSNR(originData, xGen)
        average_ssim = ssim(originData, xGen, data_range=1, size_average=True).item()

        euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(xGen, originData)).item()

        psnr_list.append(average_psnr)
        ssim_list.append(average_ssim)
        euclidean_dist_list.append(euclidean_dist)

    psnr = np.mean(psnr_list)
    ssim_value  = np.mean(ssim_list)
    euclidean_dist = np.mean(euclidean_dist_list)
    print("PSNR:", psnr)
    print("SSIM:", ssim_value )
    print("euclidean_dist", euclidean_dist)
    print("################################ save data ############################")


            # # 保存元素数据
            # origin_data = tensor2df(originData.detach())
            # # 判断路径是否存在
            # origin_data.to_csv(args.save + "origin.csv", mode='a', header=False, index=False)
            #
            # inverse_data = tensor2df(xGen.detach())
            # inverse_data.to_csv(args.save + "inverse.csv", mode='a', header=False, index=False)

    return psnr, euclidean_dist




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
    parser.add_argument('--A_ratio', type=float, default=0.5, help="Recovery data negative number loss intensity")
    
    # config file
    parser.add_argument('--c', type=str, default='./configs/attack/cifar10/data.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    # 指定rebuild的表格特征
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

    decoder_mode = args.decoder_mode
    shadow_model = args.shadow_model
    base_mode = args.base_mode
    radio_list = [0.1, 0.3, 0.7, 0.9]
    # radio_list = [0.3, 0.7, 0.9]
    # radio_list = [0.1]

    for r in radio_list:
        print("radio: ", r)
        freeze_rand(args.seed)
        # train, test = preprocess(args.data_dir, A_ratio=r)
            # Load CIFAR-10 dataset
        trainset = IndexedCIFAR10(A_ratio=r, root=args.data_dir, train=True, download=True, transform=train_transform)
        testset = IndexedCIFAR10(A_ratio=r, root=args.data_dir, train=False, download=True, transform=train_transform)


        args.A_ratio = r

        args.base_mode = base_mode + str(r)
        args.decoder_mode = decoder_mode + str(r)
        args.shadow_model = shadow_model + str(r)


        rebuild(train_data=trainset, test_data=testset, device=device, args=args)


