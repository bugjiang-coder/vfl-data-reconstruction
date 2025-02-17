import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
# 加入模块的搜索路径
sys.path.append("/data/yangjirui/vfl/vfl-tab-reconstruction")


from fedml_core.preprocess.cifar10.preprocess_cifar10 import IndexedCIFAR10
from fedml_core.model.cifar10Models import BottomModelForCifar10, TopModelForCifar10
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import adult_dataset, over_write_args_from_file, keep_predict_loss

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import time
import glob
import wandb
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import logging


def run_experiment(device, args):
    print("hyper-parameters:")
    print("batch size: {0}".format(args.batch_size))
    print("learning rate: {0}".format(args.lr))

    print("################################ Load Data ############################")
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

    train_queue = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers
        )
    test_queue = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

    # print(train_queue.dataset.data.shape)
    # sys.exit(0)

    print("################################ Set Federated Models, optimizer, loss ############################")

    top_model = TopModelForCifar10()
    bottom_model_list = [BottomModelForCifar10(), BottomModelForCifar10()]
    model_list = bottom_model_list + [top_model]

    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model
        in model_list
    ]

    stone1 = 50 # 50 int(args.epochs * 0.5)0.1
    stone2 = 85  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[2],
                                                                    milestones=[stone1, stone2],
                                                                    gamma=0.1)
    lr_scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0],
                                                            milestones=[stone1, stone2], gamma=0.1)
    lr_scheduler_b = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[1],
                                                            milestones=[stone1, stone2], gamma=0.1)
    # change the lr_scheduler to the one you want to use
    lr_scheduler_list = [lr_scheduler_a, lr_scheduler_b, lr_scheduler_top_model]


    vfltrainer = VFLTrainer(top_model, bottom_model_list, args)

    # loss function
    # criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss

    # optionally resume from a checkpoint

    print("################################ Train Federated Models ############################")

    best_top1_acc = 0.0

    for epoch in range(100):

        # logging.info('epoch %d args.lr %e ', epoch, args.lr)

        train_loss = vfltrainer.train(train_queue, criterion, bottom_criterion, optimizer_list, device, args)

        # [optimizer_list[i].zero_grad() for i in range(k)]

        lr_scheduler_list[0].step()
        lr_scheduler_list[1].step()
        lr_scheduler_list[2].step()
        
        test_loss, top1_acc, top5_acc = vfltrainer.test_mul(test_queue, criterion, device)

        # wandb.log({"train_loss": train_loss[0],
        #            "test_loss": test_loss,
        #            "test_acc": acc,
        #            "test_precision": precision,
        #            "test_recall": recall,
        #            "test_f1": f1,
        #            "test_auc": auc
        #            })

        # print(
        #     "--- epoch: {0}, train_loss: {1},test_loss: {2}, test_acc: {3}, test_precison: {4}, test_recall: {5}, test_f1: {6}, test_auc: {7}"
        #     .format(epoch, train_loss, test_loss, acc, precision, recall, f1, auc))

        print("--- epoch: {0}, train_loss: {1}, test_loss: {2}, test_top1_acc: {3}, test_top5_acc: {4}".format(epoch, train_loss, test_loss, top1_acc, top5_acc))
        # logger.info(
        #     "--- epoch: {0}, train_loss: {1}, test_loss: {2}, test_acc: {3}, test_precision: {4}, test_recall: {5}, test_f1: {6}, test_auc: {7}"
        #     .format(epoch, train_loss[0], test_loss, acc, precision, recall, f1, auc))


        ## save partyA and partyB model parameters
        vfltrainer.save_model(args.save, 'best.pth.tar', epoch, top1_acc)


    # vfltrainer.save_model('/data/yangjirui/vfl-tab-reconstruction/model/adult/', 'final.pth.tar')


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_args(parser):
    parser.add_argument('--data_dir', default='/home/yangjirui/VFL/feature-infer-workspace/dataset/adult/adult.data',
                        help='location of the data corpus')
    parser.add_argument('--name', type=str, default='adult-2layer', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    parser.add_argument('--layers', type=int, default=18, help='total number of layers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
    parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--encoder', default='./model/VIME/checkpoint_0095.pth.tar', type=str, metavar='PATH',
                        help='path to trained encoder checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save', default='./model/adult/base', type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: none)')
    
    parser.add_argument('--iso', action='store_true', default=False, help='iso defense')
    parser.add_argument('--DP', action='store_true', default=False, help='iso defense')
    parser.add_argument('--max_norm', action='store_true', default=False, help='maxnorm defense')

    parser.add_argument('--iso_ratio', type=float, default=0.01, help='iso defense ratio')
    parser.add_argument('--DP_ratio', type=float, default=0.01, help='iso defense ratio')
    
    # parser.add_argument('--tmax', type=float, default=1.00e-05, help='Maximum clipping threshold for gradients')
    # parser.add_argument('--tmin', type=float, default=-1.00e-05, help='Minimum clipping threshold for gradients')
    parser.add_argument('--Pmax', type=float, default=1.00e-05)
    parser.add_argument('--Pmin', type=float, default=-1.00e-05)
    parser.add_argument('--Pepsilon', type=float, default=0.1)
    parser.add_argument('--Pgamma', type=float, default=0.07)


    # config file
    parser.add_argument('--c', type=str, default='./configs/train/cifar10_base.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args

def run(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("################################ start experiment ############################")
    print(args.save)
    print(device)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # txt_name = f"saved_process_data"
    # savedStdout = sys.stdout
    # with open(args.save + '/' + txt_name + '.txt', 'a') as file:
    #     sys.stdout = file
    #     run_experiment(device=device, args=args)
    #     sys.stdout = savedStdout
    # print("################################ end experiment ############################")
    run_experiment(device=device, args=args)



if __name__ == '__main__':
    print("################################ Prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(1)

    save_path = "./model/cifar10/defense/"

    # # 这个是一个类似tensorboard的东西,可视化实验过程
    # wandb.init(project="vfl-tab-reconstruction", entity="potatobugjiang",
    #            name="VFL-{}".format(args.name),
    #            config=args)

    list_of_args = []

    # 列出所有的防御方法
    # protectMethod = ['non', 'max_norm', 'iso', 'dp']
    # protectMethod = ['non', 'iso', 'dp']
    # protectMethod = ['dp']
    # protectMethod = ['iso', 'dp']
    # protectMethod = ['vfldefender']
    protectMethod = ['PA_iMFL']

    iso_range = [0.001, 0.01, 0.1, 0.5, 1.0]

    # iso_range = [1.0,1.5,2.0,2.5,3.0,3.5,4.0]
    # iso_range = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    # iso_range = [1.0]

    # dp_range = [0.1]
    # dp_range = [0.3]
    dp_range = [0.001, 0.01, 0.1, 0.5, 1.0]

    for method in protectMethod:
        if method == 'max_norm':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.max_norm = True
            args.save = save_path + 'max_norm'
            freeze_rand(args.seed)
            list_of_args.append(args)
        elif method == 'dp':
            for dp in dp_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.DP = True
                args.DP_ratio = dp
                args.save = save_path + 'DP' + str(dp)
                freeze_rand(args.seed)
                list_of_args.append(args)
        elif method == 'iso':
            for iso in iso_range:
                parser = argparse.ArgumentParser("vflmodelnet")
                args = set_args(parser)
                args.iso = True
                args.iso_ratio = iso
                args.save = save_path + 'iso' + str(iso)
                freeze_rand(args.seed)
                list_of_args.append(args)
        elif method == 'non':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'non'
            freeze_rand(args.seed)
            list_of_args.append(args)
        elif method == 'vfldefender':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'vfldefender'
            freeze_rand(args.seed)
            list_of_args.append(args)
        elif method == 'PA_iMFL':
            parser = argparse.ArgumentParser("vflmodelnet")
            args = set_args(parser)
            args.save = save_path + 'PA_iMFL'
            freeze_rand(args.seed)
            list_of_args.append(args)

    for arg in list_of_args:
        run(arg)
    # with Pool(processes=1) as pool: