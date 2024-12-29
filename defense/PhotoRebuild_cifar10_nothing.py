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
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
# 加入模块的搜索路径
sys.path.append("/data/yangjirui/vfl/vfl-tab-reconstruction")

from fedml_core.preprocess.cifar10.preprocess_cifar10 import IndexedCIFAR10
from fedml_core.model.cifar10Models import BottomModelForCifar10, TopModelForCifar10, CIFAR10CNNDecoder
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import over_write_args_from_file, Similarity, save_tensor_as_image, test_rebuild_psnr, PSNR

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
    # while True:
    for i in range(0, 120):
        # train and update
        epoch_loss = []
        for step, (trn_X, trn_y) in enumerate(train_queue):
            if step == 1:
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
                print("loss:", loss.item())


        epoch += 1

        # if epoch % 10 == 0:
        #     # acc, onehot_acc, num_acc, similarity, euclidean_dist = test_rebuild_acc(train_queue, net, decoder, tab,
        #     #                                                                         device, args)
            
        #     psnr, euclidean_dist = test_rebuild_psnr(train_queue, net, decoder, device, args)
        #     print(
        #         f"psnr: {psnr}, euclidean_dist: {euclidean_dist}")

        #     if psnr >= bestPsnr:
        #         if abs(psnr-bestPsnr) < 0.01:
        #             # 检查args.decoder_mode目录是否存在
        #             save_dir = os.path.dirname(args.decoder_mode)
        #             if not os.path.exists(save_dir):
        #                 os.makedirs(save_dir)

        #             torch.save(decoder, args.decoder_mode)
        #             break

        #         consecutive_decreases = 0
        #         bestPsnr = psnr
        #         # 检查args.decoder_mode目录是否存在
        #         save_dir = os.path.dirname(args.decoder_mode)
        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)
        #         torch.save(decoder, args.decoder_mode)


        #     else:
        #         consecutive_decreases += 1

        #     if consecutive_decreases >= 2:
        #         break



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
        
        # save_path = './image/nothing'
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


def set_args(parser):
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
    parser.add_argument('--c', type=str, default='./configs/attack/cifar10/nothing.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args

def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    print("################################ prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("TabRebuild")
    
    
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

    save_path = "./model/cifar10/defense/"
    list_of_args = []

    # 列出所有的防御方法
    # protectMethod = ['non', 'max_norm', 'iso', 'dp']
    # protectMethod = ['iso', 'dp']
    # protectMethod = ['vfldefender']
    protectMethod = ['PA_iMFL']
    # protectMethod = ['iso', 'dp']
    # protectMethod = ['iso']

    iso_range = [0.001, 0.01, 0.1, 0.5, 1.0]

    # iso_range = [1.0,1.5,2.0,2.5,3.0,3.5,4.0]
    # iso_range = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    # iso_range = [1.0]

    # dp_range = [0.1]
    dp_range = [0.5]
    # dp_range = [0.001, 0.01, 0.1, 0.5, 1.0]

    for method in protectMethod:
        if method == 'max_norm':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'max_norm'
            args.base_mode = save_path + 'max_norm' + '/best.pth.tar'
            args.decoder_mode = save_path + 'max_norm' + "/non" + '/decoder.pth.tar'
            freeze_rand(args.seed)
            list_of_args.append(args)
        elif method == 'dp':
            for dp in dp_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.save = save_path + 'DP' + str(dp)
                args.base_mode = save_path + 'DP' + str(dp) + '/best.pth.tar'
                args.decoder_mode = save_path + 'DP' + str(dp) + "/non" + '/decoder.pth.tar'+str(dp)
                freeze_rand(args.seed)
                list_of_args.append(args)
        elif method == 'iso':
            for iso in iso_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.save = save_path + 'iso' + str(iso)
                args.base_mode = save_path + 'iso' + str(iso) + '/best.pth.tar'
                args.decoder_mode = save_path + 'iso' + str(iso) + "/non" + '/decoder.pth.tar'+str(iso)
                freeze_rand(args.seed)
                list_of_args.append(args)
        elif method == 'non':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'non'
            args.base_mode = save_path + 'non' + '/best.pth.tar'
            args.decoder_mode = save_path + 'non' + "/non" + '/decoder.pth.tar'
            freeze_rand(args.seed)
            list_of_args.append(args)
        elif method == 'vfldefender':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'vfldefender'
            args.base_mode = save_path + 'vfldefender' + '/best.pth.tar'
            args.decoder_mode = save_path + 'vfldefender' + "/vfldefender" + '/decoder.pth.tar'
            freeze_rand(args.seed)
            list_of_args.append(args)
        elif method == 'PA_iMFL':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'PA_iMFL'
            args.base_mode = save_path + 'PA_iMFL' + '/best.pth.tar'
            args.decoder_mode = save_path + 'PA_iMFL' + "/PA_iMFL" + '/decoder.pth.tar'
            freeze_rand(args.seed)
            list_of_args.append(args)

    # args.decoder_mode = decoder_mode + str(r)
    # args.shadow_model = shadow_model + str(r)

    for arg in list_of_args:
        print("################################ start experiment ############################")
        print(arg.save)
        print(device)
        if not os.path.exists(arg.save):
            os.makedirs(arg.save)
        txt_name = f"saved_attack_nothing"
        # savedStdout = sys.stdout

        # with open(arg.save + '/' + txt_name + '.txt', 'a') as file:
        #     sys.stdout = file
        #     trainset = IndexedCIFAR10(root=arg.data_dir, train=True, download=True, transform=train_transform)
        #     testset = IndexedCIFAR10(root=arg.data_dir, train=False, download=True, transform=train_transform)
        #     freeze_rand(arg.seed)
        #     rebuild(train_data=trainset, test_data=testset, device=device, args=arg)
        #     sys.stdout = savedStdout
        # print("################################ end experiment ############################")
        
        trainset = IndexedCIFAR10(root=arg.data_dir, train=True, download=True, transform=train_transform)
        testset = IndexedCIFAR10(root=arg.data_dir, train=False, download=True, transform=train_transform)
        freeze_rand(arg.seed)
        rebuild(train_data=trainset, test_data=testset, device=device, args=arg)
