import sys
import os
import copy
import argparse
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.append('/data/yangjirui/vfl-tab-reconstruction')
from TabRebuild_white_adult import *
from concurrent.futures import ProcessPoolExecutor


def set_args(parser):
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
    parser.add_argument('--c', type=str, default='/home/yangjirui/data/vfl-tab-reconstruction/configs/attack/whitebox/adult_base.yml', help='config file')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)



    return args



def run_experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("################################ start experiment ############################")
    # print(args.save)
    print(device)

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
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    freeze_rand(args.seed)

    txt_name = "recoder_output"
    savedStdout = sys.stdout
    savedStderr = sys.stderr
    with open(args.save + txt_name + '.txt', 'a') as file:
        sys.stdout = file
        sys.stderr = file
        rebuild(train_data=train, tab=tab, device=device, args=args)
        sys.stdout = savedStdout
        sys.stderr = savedStderr
    print("################################ end experiment ############################")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 列出所有的要运行的参数
    # bloss2_range = [10, 1, 0.1, 0.01]

    NIters_range = [1000]
    # NIters_range = [0, 10, 100, 1000]
    bloss2_v2_range = [10, 1, 0.1, 0.01]
    # bloss2_v2_range = [10,1,0]
    numloss_range = [10, 1, 0.1, 0.01]
    # numloss_range = [1]
    lr_range = [10, 1, 0.1, 0.01, 0.001]
    # lr_range = [1]

    iloss_range = [100, 10,  0]

    list_of_args = []

    # save_path = './model/CIFAR10/noaug/'

    parser = argparse.ArgumentParser("TabRebuild")
    args = set_args(parser)
    print(args)
    list_of_args.append(args)

    temp_list_of_args = []
    for bloss2_v2 in bloss2_v2_range:
        for args in list_of_args:
            args = copy.deepcopy(args)
            args.bloss2_v2 = bloss2_v2
            temp_list_of_args.append(args)
    list_of_args = temp_list_of_args

    temp_list_of_args = []
    for NIters in NIters_range:
        for args in list_of_args:
            args = copy.deepcopy(args)
            args.NIters = NIters
            temp_list_of_args.append(args)
    list_of_args = temp_list_of_args

    temp_list_of_args = []
    for numloss in numloss_range:
        for args in list_of_args:
            args = copy.deepcopy(args)
            args.numloss = numloss
            temp_list_of_args.append(args)
    list_of_args = temp_list_of_args

    temp_list_of_args = []
    for lr in lr_range:
        for args in list_of_args:
            args = copy.deepcopy(args)
            args.lr = lr
            temp_list_of_args.append(args)
    list_of_args = temp_list_of_args

    temp_list_of_args = []
    for iloss in iloss_range:
        for args in list_of_args:
            args = copy.deepcopy(args)
            args.iloss = iloss
            temp_list_of_args.append(args)
    list_of_args = temp_list_of_args

    print(len(list_of_args))
    # for args in list_of_args:
    #     print(args)

    # run_experiment(list_of_args[0])
    # 是否要规范化



    # Create a pool of workers and run experiments in parallel
    # 同时最大运行3个进程
    # with Pool(processes=3) as pool:
    #     pool.map(run_experiment, list_of_args)

    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     executor.map(run_experiment, list_of_args)

    for i in range(len(list_of_args)):
        # print("################################ start experiment ############################")
        print("progress: ", i, "/", len(list_of_args))

        run_experiment(list_of_args[i])

