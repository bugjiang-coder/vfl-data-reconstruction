import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.preprocess.zhongyuan.preprocess_zhongyuan import preprocess_zhongyuan
from fedml_core.model.net import active_model, passive_model
from fedml_core.utils.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import zhongyuan_dataset

#from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import wandb
import shutil


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def run_experiment(train_data, test_data, device, args):
    print("hyper-parameters:")
    print("batch size: {0}".format(args.batch_size))
    print("learning rate: {0}".format(args.lr))

    # input cuda
    '''
    train_data = [torch.from_numpy(x).to(device) for x in train_data]
    test_data = [torch.from_numpy(x).to(device) for x in test_data]
    '''
    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data


    print("################################ Wire Federated Models ############################")

    active_party = active_model(input_dim=Xa_train.shape[1], intern_dim=Xb_train.shape[1]+2, num_classes=1, k=args.k)


    #model_list = [active_party]+ [passive_model(input_dim=Xb_train.shape[1], intern_dim=20, output_dim=10) for _ in range(args.k-1)]
    model_list = [active_party] + [passive_model(input_dim=Xb_train.shape[1], intern_dim=Xb_train.shape[1]+1, output_dim=Xb_train.shape[1]+2) for _ in
                                   range(args.k - 1)]



    # dataloader
    train_dataset = zhongyuan_dataset(train_data)
    test_dataset = zhongyuan_dataset(test_data)


    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model
        in model_list
    ]

    #model_list = [model.train() for model in model_list]

    vfltrainer = VFLTrainer(model_list)

    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            for i in range(len(model_list)):
                model_list[i].load_state_dict(checkpoint['state_dict'][i])
                #optimizer_list[i].load_state_dict(checkpoint['optimizer'][i])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    print("################################ Train Federated Models ############################")

    best_auc = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        logging.info('epoch %d args.lr %e ', epoch, args.lr)

        train_loss = vfltrainer.train(train_queue, criterion, optimizer_list, device, args)

        #[optimizer_list[i].zero_grad() for i in range(k)]

        acc, auc, test_loss, precision, recall, f1 = vfltrainer.test(test_queue, criterion, device)


        wandb.log({"train_loss": train_loss[0],
                   "test_loss": test_loss,
                   "test_acc": acc,
                   "test_precision": precision,
                   "test_recall": recall,
                   "test_f1": f1,
                   "test_auc": auc
        })

        print("--- epoch: {0}, train_loss: {1},test_loss: {2}, test_acc: {3}, test_precison: {4}, test_recall: {5}, test_f1: {6}, test_auc: {7}"
              .format(epoch, train_loss[0], test_loss, acc, precision, recall, f1, auc))

        ## save partyA and partyB model parameters
        # if epoch % args.report_freq == 0:
        #     is_best = auc > best_auc
        #     best_auc = max(auc, best_auc)
        #
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'best_auc': best_auc,
        #         'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
        #         'optimizer': [optimizer_list[i].state_dict() for i in range(len(optimizer_list))],
        #     }, is_best, './model/zhongyuan/Tab-baseline',  'checkpoint_{:04d}.pth.tar'.format(epoch))

    vfltrainer.save_model('/home/yangjirui/paper-code/model/zhongyuan/',"4layer_normal_enhance")



if __name__ == '__main__':
    print("################################ Prepare Data ############################")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("vflmodelnet")
    parser.add_argument('--data_dir', default="/home/yangjirui/VFL/feature-infer-workspace/dataset/zhongyuan/", help='location of the data corpus')
    parser.add_argument('--name', type=str, default='4layer_normal_enhance', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--workers', type=int, default=2, help='num of workers')
    parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
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

    # 这个是一个类似tensorboard的东西,可视化实验过程
    wandb.init(project="F-VFL-basemodel", entity="yang-test",
               name="VFL-{}".format(args.name),
               config=args)

    #data_dir = "../../../data/zhongyuan/"
    train, test = preprocess_zhongyuan(args.data_dir)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train]
    test = [Xa_test, Xb_test, y_test]
    run_experiment(train_data=train, test_data=test, device=device, args=args)

    # reference training result:
    # --- epoch: 99, batch: 1547, loss: 0.11550658332804839, acc: 0.9359105089400196, auc: 0.8736984159409958
    # --- (0.9270889578726378, 0.5111934752243287, 0.5054099033579607, None)

    # --- epoch: 99, batch: 200, loss: 0.09191526211798191, acc: 0.9636565918783608, auc: 0.9552342451916291
    # --- (0.9754657898538487, 0.7605652456769234, 0.8317858679682943, None)

