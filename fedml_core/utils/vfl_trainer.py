import os

import torch
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from fedml_core.utils.utils import ModelTrainer
import numpy as np




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
    # 将输出的0-1之间的值根据0.5为分界线分为两类,同时统计正确率
    y_hat_lbls = []
    pred_pos_count = 0
    pred_neg_count = 0
    correct_count = 0
    for y_prob, y_t in zip(y_prob_preds, y_targets):
        if y_prob <= threshold:
            pred_neg_count += 1
            y_hat_lbl = 0
        else:
            pred_pos_count += 1
            y_hat_lbl = 1
        y_hat_lbls.append(y_hat_lbl)
        if y_hat_lbl == y_t:
            correct_count += 1

    return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]

class VFLTrainer(ModelTrainer):
    def get_model_params(self):
        return [self.active_model.cpu().state_dict()] + [model.cpu().state_dict() for model in self.passive_model_list]

    def set_model_params(self, model_parameters):
        self.active_model.load_state_dict(model_parameters[0])
        for i in range(len(self.passive_model_list)):
            self.passive_model_list[i].load_state_dict(model_parameters[i+1])

    def save_model(self, dir, name='', epoch=None, auc=None):
        if not os.path.exists(dir):
            os.makedirs(dir)

        model_list = self.passive_model_list
        optimizer_list = self.passive_optimizer_list

        state = {
            'epoch': epoch,
            'auc': auc,
            'state_dict': [self.active_model.state_dict()]+[model_list[i].state_dict() for i in range(len(model_list))],
            'optimizer': [self.active_optimizer.state_dict()]+[optimizer_list[i].state_dict() for i in range(len(optimizer_list))],
        }

        if name:
            filename = os.path.join(dir, name)
        else:
            filename = os.path.join(dir, 'checkpoint.pth.tar')
        torch.save(state, filename)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.active_model.load_state_dict(checkpoint['state_dict'][0])
        self.active_optimizer.load_state_dict(checkpoint['optimizer'][0])
        for i in range(len(self.passive_model_list)):
            self.passive_model_list[i].load_state_dict(checkpoint['state_dict'][i+1])
            self.passive_optimizer_list[i].load_state_dict(checkpoint['optimizer'][i+1])

        # only model to device
        # self.active_model.to(device)
        # for i in range(len(model_list)):
        #     model_list[i].to(device)

        return checkpoint['epoch'], checkpoint['auc']




    def train(self, train_data, criterion, device, args):

        model_list = [self.active_model] + self.passive_model_list
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        optimizer_list = [self.active_optimizer] + self.passive_optimizer_list

        # # 打印model的device
        # for model in model_list:
        #     # print(model.device)
        #     print(model.state_dict())

        # train and update
        epoch_loss = []
        for step, (trn_X, trn_y) in enumerate(train_data):
            trn_X = [x.float().to(device) for x in trn_X]
            target = trn_y.float().to(device)
            batch_loss = []
            # logging.info("x.size = " + str(x.size()))
            # logging.info("labels.size = " + str(labels.size()))

            #Xb_encoding, _, _ = tabular_encoder(trn_X[1])

            [optimizer_list[i].zero_grad() for i in range(len(model_list))]

            # 把被动方法的数据放进去
            U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]
            # 把输出复制一份拷贝出来
            U_B_clone_list = [U_B.detach().clone() for U_B in U_B_list]
            U_B_clone_list = [U_B.requires_grad_(True) for U_B in U_B_clone_list]

            #m = nn.Sigmoid()
            # Q:很奇怪训练的时候不用sigmoid 测推理的使用用sigmoid 为什么要这样设计?

            logits = model_list[0](trn_X[0], U_B_clone_list)

            loss = criterion(logits, target)

            # 1. 在主动方的模型上，计算非主动方的梯度
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in U_B_clone_list]

            # add max_norm noise
            #U_B_gradients_list = gradient_masking(U_B_gradients_list)
            # add iso gaussian noise
            #U_B_gradients_list = gradient_gaussian_noise_masking(U_B_gradients_list, ratio=1.0)
            # add marvell noise
            #U_B_gradients_list = marvell_g(U_B_gradients_list, target)

            # grad_outputs: 一个与输出形状相同的张量，表示输出相对于某个标量张量的梯度
            # 让几个模型连接起来
            model_B_weights_gradients_list = [
                torch.autograd.grad(U_B_list[i], model_list[i + 1].parameters(), grad_outputs=U_B_gradients_list[i],
                                    retain_graph=True) for i in range(len(U_B_gradients_list))]

            for i in range(len(model_B_weights_gradients_list)):
                for w, g in zip(model_list[i + 1].parameters(), model_B_weights_gradients_list[i]):
                    w.grad = g.detach()
                #  将梯度张量的范数（Norm）限制在一个指定的阈值范围内,防止过拟合
                nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)
                optimizer_list[i + 1].step()

            # 2. 计算主动方的梯度
            loss.backward()
            #  将梯度张量的范数（Norm）限制在一个指定的阈值范围内,防止过拟合
            nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
            optimizer_list[0].step()


            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss

    def train_passive_mode(self, train_data, criterion, device, args):
        # 注意着个函数只能处理1个参与方

        model_list = [self.active_model] + self.passive_model_list
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        optimizer = torch.optim.SGD(self.passive_model_list[0].parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        # # 打印model的device
        # for model in model_list:
        #     # print(model.device)
        #     print(model.state_dict())

        # train and update
        epoch_loss = []
        batch_loss = []
        for step, (trn_X, trn_y) in enumerate(train_data):
            trn_X = [x.float().to(device) for x in trn_X]
            target = trn_y.float().to(device)
            # batch_loss = []

            optimizer.zero_grad()

            # 把被动方法的数据放进去
            U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]
            # 把输出复制一份拷贝出来
            U_B_clone_list = [U_B.detach().clone() for U_B in U_B_list]
            U_B_clone_list = [U_B.requires_grad_(True) for U_B in U_B_clone_list]

            #m = nn.Sigmoid()
            # Q:很奇怪训练的时候不用sigmoid 测推理的使用用sigmoid 为什么要这样设计?

            logits = model_list[0](trn_X[0], U_B_clone_list)

            loss = criterion(logits, target)

            # 1. 在主动方的模型上，计算非主动方的梯度
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in U_B_clone_list]

            # add max_norm noise
            #U_B_gradients_list = gradient_masking(U_B_gradients_list)
            # add iso gaussian noise
            #U_B_gradients_list = gradient_gaussian_noise_masking(U_B_gradients_list, ratio=1.0)
            # add marvell noise
            #U_B_gradients_list = marvell_g(U_B_gradients_list, target)

            # grad_outputs: 一个与输出形状相同的张量，表示输出相对于某个标量张量的梯度
            # 让几个模型连接起来
            model_B_weights_gradients_list = [
                torch.autograd.grad(U_B_list[i], model_list[i + 1].parameters(), grad_outputs=U_B_gradients_list[i],
                                    retain_graph=True) for i in range(len(U_B_gradients_list))]

            for i in range(len(model_B_weights_gradients_list)):
                for w, g in zip(model_list[i + 1].parameters(), model_B_weights_gradients_list[i]):
                    w.grad = g.detach()
                #  将梯度张量的范数（Norm）限制在一个指定的阈值范围内,防止过拟合
                nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)


            # 2. 计算主动方的梯度
            loss.backward()
            #  将梯度张量的范数（Norm）限制在一个指定的阈值范围内,防止过拟合
            nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
            optimizer.step()


            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss


    def test(self, test_data, criterion, device):
        model_list = [self.active_model] + self.passive_model_list

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        m = nn.Sigmoid()

        Loss = AverageMeter()
        AUC = AverageMeter()
        ACC = AverageMeter()
        Precision = AverageMeter()
        Recall = AverageMeter()
        F1 = AverageMeter()

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        with torch.no_grad():
            for batch_idx, (trn_X, target) in enumerate(test_data):
                trn_X = [x.float().to(device) for x in trn_X]
                target = target.float().to(device)

                #Xb_encoding, _, _ = tabular_encoder(trn_X[1])


                U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]
                U_B_clone_list = [U_B.detach().clone() for U_B in U_B_list]
                U_B_clone_list = [U_B.requires_grad_(True) for U_B in U_B_clone_list]

                pred = m(model_list[0](trn_X[0], U_B_clone_list))


                loss = criterion(pred, target)

                y_hat_lbls, statistics = compute_correct_prediction(y_targets=target,
                                                                    y_prob_preds=pred,
                                                                    threshold=0.5)

                acc = accuracy_score(target.cpu().numpy(), y_hat_lbls)
                auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
                metrics = precision_recall_fscore_support(target.cpu().numpy(), y_hat_lbls, average="macro",
                                                          warn_for=tuple())
                # print(classification_report(target.cpu().numpy(), y_hat_lbls))
                '''
                predicted = (pred > .5).int()
                correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                recall = true_positive / (target.sum(axis=-1) + 1e-13)
                metrics['test_precision'] += precision.sum().item()
                metrics['test_recall'] += recall.sum().item()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                '''
                ACC.update(acc)
                AUC.update(auc)
                Loss.update(loss)
                Precision.update(metrics[0])
                Recall.update(metrics[1])
                F1.update(metrics[2])

        return ACC.avg, AUC.avg, Loss.avg, Precision.avg, Recall.avg, F1.avg


    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
