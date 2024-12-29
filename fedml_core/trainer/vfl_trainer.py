import torch
import os
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve
import numpy as np
import wandb
import copy
import torch.nn.functional as F
# from fedml_core.utils.utils import AverageMeter, gradient_masking, gradient_gaussian_noise_masking, marvell_g, backdoor_truepostive_rate, apply_noise_patch, gradient_compression, laplacian_noise_masking
from fedml_core.utils.utils import AverageMeter, gaussian_noise_masking, smashed_data_masking, tabRebuildAcc, Similarity, reconstruct_input, reconstruct_input_count, VFLDefender, PA_iMFL
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


from .model_trainer import ModelTrainer

def update_model_one_batch(optimizer, model, output, batch_target, loss_func, args):
    # print(output)
    # print(batch_target)
    # print(loss_func)
    # CrossEntropyLoss()
    # 如果是CrossEntropyLoss
    # 如果loss_func是CrossEntropyLoss的一个实例
    if isinstance(loss_func, nn.CrossEntropyLoss):
        # 为了避免错误，确保目标是长整型
        # print('CrossEntropyLoss')
        batch_target = batch_target.long()
        # loss = loss_func(output, batch_target)
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    # 裁剪梯度,防止梯度爆炸
    #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss


def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
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

def split_data(data, args):
    if args.dataset == 'Yahoo':
        x_b = data[1]
        x_a = data[0]
    elif args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    elif args.dataset == 'Criteo':
        x_b = data[:, args.half:D_]
        x_a = data[:, 0:args.half]
    elif args.dataset == 'BCW':
        x_b = data[:, args.half:28]
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b

class InvertingGradientsAttack:
    def __init__(self, model, criterion, device, input_dim):
        """
        初始化攻击类
        :param model: 目标 BottomModel
        :param criterion: 损失函数
        :param device: 设备（CPU或GPU）
        :param input_dim: 输入数据维度
        :param target_output: 真实标签，用于计算损失
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.input_dim = input_dim
        # self.target_output = target_output

    def invert(self, observed_grads, target_output, num_iterations=1000, lr=0.1):
        """
        执行输入重建
        :param observed_grads: 目标模型参数的梯度字典
        :param num_iterations: 优化迭代次数
        :param lr: 学习率
        :return: 重建的输入张量
        """
        # 确保 target_output 在正确的设备上，并且需要梯度
        target_output = target_output.detach().clone().to(self.device)
        # if not target_output.requires_grad:
        # target_output.requires_grad_(False)
        # print(self.input_dim)
        # 初始化一个伪输入张量，requires_grad=True 以便优化
        # dummy_input = torch.randn(self.input_dim, device=self.device, requires_grad=True)
        # 初始化为0
        dummy_input = torch.zeros(self.input_dim, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([dummy_input], lr=lr)
        # 使用sgd优化器
        # optimizer = torch.optim.SGD([dummy_input], lr=lr)
        
        loss_fn = nn.MSELoss()

        for i in range(num_iterations):
            optimizer.zero_grad()
            output = self.model(dummy_input)
            loss = self.criterion(output, target_output)
            loss.backward()
            # 使用 torch.autograd.grad 计算当前梯度
            # current_grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)


            # # 收集当前伪输入生成的参数梯度
            current_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    current_grads[name] = param.grad.clone().detach()
            # print(observed_grads)
            # print(observed_grads.keys())
            # print(current_grads)
            # print(current_grads.keys())

            # 计算伪梯度与观察到梯度的差异
            total_loss = 0.0
            for name in observed_grads:
                if name in current_grads:
                    # 确保 observed_grads 中的梯度张量可导且需要梯度
                    if not observed_grads[name].requires_grad:
                        observed_grads[name].requires_grad_(True)
                    total_loss += loss_fn(current_grads[name], observed_grads[name])

            # 反向传播以优化伪输入
            total_loss.backward()
            optimizer.step()

            # if i % 1 == 0 or i == num_iterations - 1:
            #     print(f"\tIteration {i+1}/{num_iterations}, Gradient Loss: {total_loss.item()}")

        return dummy_input.detach()

class VFLTrainer(ModelTrainer):
    def update_model(self, new_model):
        # 更新self.model
        self.model = new_model

    def get_model_params(self):
        return [self.active_model.cpu().state_dict()] + [model.cpu().state_dict() for model in self.passive_model_list]

    def set_model_params(self, model_parameters):
        self.active_model.load_state_dict(model_parameters[0])
        for i in range(len(self.passive_model_list)):
            self.passive_model_list[i].load_state_dict(model_parameters[i+1])

    def save_model(self, dir, name='', epoch=None, auc=None):
        if not os.path.exists(dir):
            os.makedirs(dir)

        state = {
            'epoch': epoch,
            'auc': auc,
            'state_dict': [self.top_model.state_dict()]+[model.state_dict() for model in self.bottom_model_list]
            # 'optimizer': [self.active_optimizer.state_dict()]+[optimizer_list[i].state_dict() for i in range(len(optimizer_list))],
        }

        if name:
            filename = os.path.join(dir, name)
        else:
            filename = os.path.join(dir, 'checkpoint.pth.tar')
        torch.save(state, filename)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.top_model.load_state_dict(checkpoint['state_dict'][0])
        # self.active_optimizer.load_state_dict(checkpoint['optimizer'][0])
        for i in range(len(self.bottom_model_list)):
            self.bottom_model_list[i].load_state_dict(checkpoint['state_dict'][i+1])
            # self.passive_optimizer_list[i].load_state_dict(checkpoint['optimizer'][i+1])

        return checkpoint['epoch'], checkpoint['auc']


    def train(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):

        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []


        for step, (trn_X, trn_y) in enumerate(train_data):

            trn_X = [x.float().to(device) for x in trn_X]
            target = trn_y.float().to(device)

            #input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            #input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](trn_X[1])
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](trn_X[0])

            # output_tensor_bottom_model_b = output_tensor_bottom_model_b.to(device)

            # 对smashed data扰动
            # if hasattr(args, 'DP') and args.DP:
            # if args.max_norm:
            if hasattr(args, 'max_norm') and args.max_norm:
                output_tensor_bottom_model_b = smashed_data_masking(output_tensor_bottom_model_b)
            # 增加iso gaussian noise
            # if args.iso:
            if hasattr(args, 'iso') and args.iso:
                output_tensor_bottom_model_b = gaussian_noise_masking(output_tensor_bottom_model_b,
                                                                             ratio=args.iso_ratio)
            # test
            # output_tensor_bottom_model_a = torch.zeros_like(output_tensor_bottom_model_a)
            # output_tensor_bottom_model_b = torch.zeros_like(output_tensor_bottom_model_b)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                                    model=model_list[2],
                                                    output=output,
                                                    batch_target=target,
                                                    loss_func=criterion,
                                                    args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # 差分隐私防御方法
            # if args.DP:
            if hasattr(args, 'DP') and args.DP:
                grad_output_bottom_model_b = gaussian_noise_masking(grad_output_bottom_model_b,
                                                                      ratio=args.DP_ratio)

            # 合并所有 δo（假设 δo 是针对每个底层模型的）
            # 应用 VFLDefender 到每个 δo
            # perturbed_grad_a = VFLDefender(grad_output_bottom_model_a, tmax, tmin)
                # 提取 VFLDefender 的参数
            if hasattr(args, 'tmax') and hasattr(args, 'tmin'):
                tmax = getattr(args, 'tmax', 1.0)
                tmin = getattr(args, 'tmin', -1.0)
                grad_output_bottom_model_b = VFLDefender(grad_output_bottom_model_b, tmax, tmin)
            
            if hasattr(args, 'Pepsilon') and hasattr(args, 'Pgamma') and hasattr(args, 'Pmax') and hasattr(args, 'Pmin'):
                Pepsilon = getattr(args, 'Pepsilon', 0.1)
                Pgamma = getattr(args, 'Pgamma', 0.1)
                Pmax = getattr(args, 'Pmax', 1.0)
                Pmin = getattr(args, 'Pmin', -1.0)
                # print(grad_output_bottom_model_b)
                # print("PA_iMFL")
                grad_output_bottom_model_b = PA_iMFL(grad_output_bottom_model_b, Pepsilon, Pgamma, Pmax, Pmin)
                # print(grad_output_bottom_model_b)
                # sys.exit(0)
            
            
            # Theoretical reference:https://github.com/FuChong-cyber/label-inference-attacks
            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                                    model=model_list[1],
                                                    output=output_tensor_bottom_model_b,
                                                    batch_target=grad_output_bottom_model_b,
                                                    loss_func=bottom_criterion,
                                                    args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            epoch_loss.append(loss.item())
        # epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return np.mean(epoch_loss)
    
    def train_GI(self, train_data, criterion, bottom_criterion, optimizer_list, device, tab, args):

        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        acc_list = []
        onehot_acc_list = []
        num_acc_list = []
        similarity_list = []
        euclidean_dist_list = []
        


        for step, (trn_X, trn_y) in  tqdm(enumerate(train_data), total=len(train_data)):

            trn_X = [x.float().to(device) for x in trn_X]
            target = trn_y.float().to(device)

            #input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            #input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](trn_X[1])
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](trn_X[0])

            # output_tensor_bottom_model_b = output_tensor_bottom_model_b.to(device)

            # 对smashed data扰动
            # if hasattr(args, 'DP') and args.DP:
            # if args.max_norm:
            if hasattr(args, 'max_norm') and args.max_norm:
                output_tensor_bottom_model_b = smashed_data_masking(output_tensor_bottom_model_b)
            # 增加iso gaussian noise
            # if args.iso:
            if hasattr(args, 'iso') and args.iso:
                output_tensor_bottom_model_b = gaussian_noise_masking(output_tensor_bottom_model_b,
                                                                             ratio=args.iso_ratio)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                                    model=model_list[2],
                                                    output=output,
                                                    batch_target=target,
                                                    loss_func=criterion,
                                                    args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # 差分隐私防御方法
            # if args.DP:
            if hasattr(args, 'DP') and args.DP:
                grad_output_bottom_model_b = gaussian_noise_masking(grad_output_bottom_model_b,
                                                                             ratio=args.DP_ratio)

            # Theoretical reference:https://github.com/FuChong-cyber/label-inference-attacks
            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                                    model=model_list[1],
                                                    output=output_tensor_bottom_model_b,
                                                    batch_target=grad_output_bottom_model_b,
                                                    loss_func=bottom_criterion,
                                                    args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)
            
            # 捕获目标 BottomModel 的参数梯度
            model_list[1].capture_gradients()
            observed_grads = model_list[1].param_grads.copy()
            # test
            # output_tensor_bottom_model_a = torch.zeros_like(output_tensor_bottom_model_a)
            # output_tensor_bottom_model_b = torch.zeros_like(output_tensor_bottom_model_b)
            # copy_model = copy.deepcopy(model_list[1])
            attack = InvertingGradientsAttack(
                model=model_list[1],
                criterion=criterion,
                device=device,
                input_dim=trn_X[1].shape
            )
            
            # print(f"Performing inversion attack on BottomModel B")
            reconstructed_input = attack.invert(observed_grads, input_tensor_top_model_b, num_iterations=200, lr=0.1)
            acc, onehot_acc, num_acc = tabRebuildAcc(trn_X[1], reconstructed_input, tab)
            similarity = Similarity(reconstructed_input, trn_X[1])
            euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(reconstructed_input, trn_X[1])).item()
        
            # print(f"Reconstructed Input for BottomModel B at epoch {epoch}: {reconstructed_input}")

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            epoch_loss.append(loss.item())
            acc_list.append(acc)
            onehot_acc_list.append(onehot_acc)
            num_acc_list.append(num_acc)
            similarity_list.append(similarity)
            euclidean_dist_list.append(euclidean_dist)
        # epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return np.mean(epoch_loss),np.mean(acc_list), np.mean(onehot_acc_list),np.mean(num_acc_list),np.mean(similarity_list),np.mean(euclidean_dist_list)

    def train_LLL(self, train_data, criterion, bottom_criterion, optimizer_list, device, tab, args):

        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        acc_list = []
        onehot_acc_list = []
        num_acc_list = []
        similarity_list = []
        euclidean_dist_list = []
        


        for step, (trn_X, trn_y) in  tqdm(enumerate(train_data), total=len(train_data)):

            trn_X = [x.float().to(device) for x in trn_X]
            target = trn_y.float().to(device)

            #input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            #input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](trn_X[1])
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](trn_X[0])

            # output_tensor_bottom_model_b = output_tensor_bottom_model_b.to(device)

            # 对smashed data扰动
            # if hasattr(args, 'DP') and args.DP:
            # if args.max_norm:
            if hasattr(args, 'max_norm') and args.max_norm:
                output_tensor_bottom_model_b = smashed_data_masking(output_tensor_bottom_model_b)
            # 增加iso gaussian noise
            # if args.iso:
            if hasattr(args, 'iso') and args.iso:
                output_tensor_bottom_model_b = gaussian_noise_masking(output_tensor_bottom_model_b,
                                                                             ratio=args.iso_ratio)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                                    model=model_list[2],
                                                    output=output,
                                                    batch_target=target,
                                                    loss_func=criterion,
                                                    args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # 差分隐私防御方法
            # if args.DP:
            if hasattr(args, 'DP') and args.DP:
                grad_output_bottom_model_b = gaussian_noise_masking(grad_output_bottom_model_b,
                                                                             ratio=args.DP_ratio)

            # Theoretical reference:https://github.com/FuChong-cyber/label-inference-attacks
            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                                    model=model_list[1],
                                                    output=output_tensor_bottom_model_b,
                                                    batch_target=grad_output_bottom_model_b,
                                                    loss_func=bottom_criterion,
                                                    args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)
            
            # 捕获目标 BottomModel 的参数梯度
            model_list[1].capture_gradients()
            reconstructed = reconstruct_input(model_list[1], target_layer=0)  # Assuming first FC layer is target
            # if reconstructed is not None:
            #     print("Reconstructed Input:", reconstructed)
            #     # print("Original Input:", input_data)
            # else:
            #     print("Reconstruction failed due to multiple activations.")
            
            # print("Reconstructed Input:", reconstructed.shape)
            # print("Original Input:", trn_X[1].shape)
            # sys.exit(0)
            reconstructed = reconstructed.unsqueeze(0)
            acc, onehot_acc, num_acc = tabRebuildAcc(trn_X[1], reconstructed, tab)
            similarity = Similarity(reconstructed, trn_X[1])
            euclidean_dist = torch.mean(torch.nn.functional.pairwise_distance(reconstructed, trn_X[1])).item()
        
            # print(f"Reconstructed Input for BottomModel B at epoch {epoch}: {reconstructed_input}")

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            epoch_loss.append(loss.item())
            acc_list.append(acc)
            onehot_acc_list.append(onehot_acc)
            num_acc_list.append(num_acc)
            similarity_list.append(similarity)
            euclidean_dist_list.append(euclidean_dist)
        # epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return np.mean(epoch_loss),np.mean(acc_list), np.mean(onehot_acc_list),np.mean(num_acc_list),np.mean(similarity_list),np.mean(euclidean_dist_list)

    def train_LLL_count(self, train_data, criterion, bottom_criterion, optimizer_list, device, tab, args):

        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        acc_list = []
        onehot_acc_list = []
        num_acc_list = []
        similarity_list = []
        euclidean_dist_list = []
        
        all_num, reconstructed_num = 0, 0

        for step, (trn_X, trn_y) in  tqdm(enumerate(train_data), total=len(train_data)):

            trn_X = [x.float().to(device) for x in trn_X]
            target = trn_y.float().to(device)

            #input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            #input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](trn_X[1])
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](trn_X[0])

            # output_tensor_bottom_model_b = output_tensor_bottom_model_b.to(device)

            # 对smashed data扰动
            # if hasattr(args, 'DP') and args.DP:
            # if args.max_norm:
            if hasattr(args, 'max_norm') and args.max_norm:
                output_tensor_bottom_model_b = smashed_data_masking(output_tensor_bottom_model_b)
            # 增加iso gaussian noise
            # if args.iso:
            if hasattr(args, 'iso') and args.iso:
                output_tensor_bottom_model_b = gaussian_noise_masking(output_tensor_bottom_model_b,
                                                                             ratio=args.iso_ratio)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                                    model=model_list[2],
                                                    output=output,
                                                    batch_target=target,
                                                    loss_func=criterion,
                                                    args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # 差分隐私防御方法
            # if args.DP:
            if hasattr(args, 'DP') and args.DP:
                grad_output_bottom_model_b = gaussian_noise_masking(grad_output_bottom_model_b,
                                                                             ratio=args.DP_ratio)

            # Theoretical reference:https://github.com/FuChong-cyber/label-inference-attacks
            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                                    model=model_list[1],
                                                    output=output_tensor_bottom_model_b,
                                                    batch_target=grad_output_bottom_model_b,
                                                    loss_func=bottom_criterion,
                                                    args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)
            
            # 捕获目标 BottomModel 的参数梯度
            model_list[1].capture_gradients()
            reconstructed_num += reconstruct_input_count(model_list[1], target_layer=0) 
            all_num += trn_X[1].shape[0]
            
        print("all_num:", all_num, "reconstructed_num:", reconstructed_num)
        return reconstructed_num/all_num


    def train_single_model(self, train_data, criterion, bottom_criterion, optimizer_list, bottom_model_index=None, top_model=False, device=None, args=None):

        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []


        for step, (trn_X, trn_y) in enumerate(train_data):

            trn_X = [x.float().to(device) for x in trn_X]
            #target = torch.argmax(trn_y,dim=1).long().to(device)
            target = trn_y.float().to(device)

            #input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            #input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](trn_X[1])
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](trn_X[0])

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            if top_model:
                loss = update_model_one_batch(optimizer=optimizer_list[2],
                                                        model=model_list[2],
                                                        output=output,
                                                        batch_target=target,
                                                        loss_func=criterion,
                                                        args=args)
            else:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    # 为了避免错误，确保目标是长整型
                    # print('CrossEntropyLoss')
                    target = target.long()
                loss = criterion(output, target)
                loss.backward()


            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # Theoretical reference:https://github.com/FuChong-cyber/label-inference-attacks
            # -- bottom model b backward/update--
            if bottom_model_index and bottom_model_index==1:
                _ = update_model_one_batch(optimizer=optimizer_list[1],
                                                        model=model_list[1],
                                                        output=output_tensor_bottom_model_b,
                                                        batch_target=grad_output_bottom_model_b,
                                                        loss_func=bottom_criterion,
                                                        args=args)
            if bottom_model_index and bottom_model_index==0:
                # -- bottom model a backward/update--
                _ = update_model_one_batch(optimizer=optimizer_list[0],
                                           model=model_list[0],
                                           output=output_tensor_bottom_model_a,
                                           batch_target=grad_output_bottom_model_a,
                                           loss_func=bottom_criterion,
                                           args=args)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            epoch_loss.append(loss.item())
        # epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return np.mean(epoch_loss)

    def extract_features(self, train_data, device, args):

        victim_model = self.model[1].eval().to(device)

        features = []
        labels = []
        with torch.no_grad():
            for step, (trn_X, trn_y) in enumerate(train_data):
                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    #trn_X = [x.float().to(device) for x in trn_X]
                    Xb = trn_X.float().to(device)
                    target = torch.argmax(trn_y, dim=1).long().to(device)
                # target = trn_y.float().to(device)
                batch_loss = []

                # bottom model B
                output_tensor_bottom_model_b = victim_model(Xb)
                features.append(output_tensor_bottom_model_b.detach().cpu().numpy())
                labels.append(target.detach().cpu().numpy())

        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


    def train_cluster(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):

        #model_list = self.model

        # first three models adopt train(), last model adopts eval()
        model_list = [model.eval() for model in self.model[:-1]]
        model_list.append(self.model[-1].train())
        model_list = [model.to(device) for model in model_list]

        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                #trn_X = [x.float().to(device) for x in trn_X]
                Xb = trn_X.float().to(device)
                #target = torch.argmax(trn_y, dim=1).long().to(device)
                target = trn_y.long().to(device)
            batch_loss = []

            # input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            # input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # model completion head
            pseudo_output = model_list[3](output_tensor_bottom_model_b)

            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_b.requires_grad_(True)


            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[0],
                                          model=model_list[3],
                                          output=pseudo_output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            '''
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)
            '''

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]

    def train_poisoning(self, train_data, criterion, bottom_criterion, optimizer_list, device, args, delta, poisoned_indices):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        poisoned_grads = []
        batch_poisoned_count = 0

        selected_set = set(poisoned_indices)

        # delta = torch.clamp(delta, -args.eps*2, args.eps*2)

        for step, (trn_X, trn_y, indices) in enumerate(train_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []



            '''
            selected_values = Xb[~mask]
            random_indices = torch.randint(0, (~mask).sum(), (mask.sum(),))
            Xb[mask] = selected_values[random_indices] + delta.detach().clone()
            #Xb[selected_values[random_indices]] = selected_values[random_indices] + delta.detach().clone()
            '''

            indices = indices.cpu().numpy()

            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            num_masked = mask.sum().item()

            # 假设Xb是你的数据张量，mask是你的掩码张量
            non_mask_indices = torch.where(~mask)[0]  # 找到非mask的索引
            random_indices = torch.randperm(non_mask_indices.size(0))[:num_masked]  # 从非mask的索引中随机选择5个

            Xb[non_mask_indices[random_indices]] += delta.detach().clone()  # 将delta添加到选定的Xb值

            # batch_delta.grad.zero_()
            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()

            # random-valued vector to replace the gradient of poisoned data
            random_grad = torch.randn(num_masked, input_tensor_top_model_b.shape[1]).to(device)
            input_tensor_top_model_b[non_mask_indices[random_indices]] = random_grad

            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)


            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # Target Label replacement
            #将grad_output_bottom_model_b中按mask中为True的元素索引赋值给non_mask_indices[random_indices]中的元素
            grad_output_bottom_model_b[non_mask_indices[random_indices]] = args.corruption_amp*grad_output_bottom_model_b[mask]


            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(grad_output_bottom_model_b)
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(grad_output_bottom_model_b, ratio=args.iso_ratio)
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(grad_output_bottom_model_b, target)

            # ----yjr-----
            # gradient compression
            if args.gc:
                grad_output_bottom_model_b = gradient_compression(grad_output_bottom_model_b, preserved_perc=args.gc_ratio)

            # differential privacy
            if args.lap_noise:
                grad_output_bottom_model_b = laplacian_noise_masking(grad_output_bottom_model_b, beta=args.lap_noise_ratio)

            # sign SGD
            # TODO: 这里的signSGD只是完成了在g上的操作，理论上应该使用optimizer.step()的方法来优化
            if args.signSGD:
                torch.sign(grad_output_bottom_model_b, out=grad_output_bottom_model_b)

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)



            # poison_optimizer.step()

            # poisoned_grads.append(batch_delta_grad)

            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]


    def train_mul(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):

        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                #trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []

            # input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            # input_tensor_top_model_b = torch.tensor([], requires_grad=True)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            # TODO：！！！这里的grad用法很特别，需要学习！！！
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]

    def test_cluster(self, test_data, criterion, device, args):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    #trn_X = [x.float().to(device) for x in trn_X]
                    Xb = trn_X.float().to(device)
                    target = torch.argmax(trn_y, dim=1).long().to(device)
                    #target = trn_y.long().to(device)


                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb)

                # complete head
                output = model_list[-1](output_tensor_bottom_model_b)


                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total


        print('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc

    def test_copur_target(self, test_data, criterion, device, args, poison_target_label):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0
        asr_top5_correct = 0
        asr_correct = 0

        # batch_delta = delta.unsqueeze(0).repeat([poison_target_label.shape[0], 1, 1, 1]).detach().clone()


        for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)

            target_class = torch.tensor([poison_target_label]).repeat(target.shape[0]).to(device)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb )
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)
            for i in range(50):
                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

                loss = criterion(output, target_class)

                # compute gradients of the loss with respect to output_tensor_bottom_model_b
                grad_output_bottom_model_b = torch.autograd.grad(loss, output_tensor_bottom_model_b)[0]

                adv_feat = output_tensor_bottom_model_b - 0.1 * grad_output_bottom_model_b.detach().clone()

                output_tensor_bottom_model_b = adv_feat.detach().clone()

            output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)


            probs = F.softmax(output, dim=1)
            # Top-1 accuracy
            total += target.size(0)
            _, pred = probs.topk(1, 1, True, True)
            _, top5_preds = probs.topk(5, 1, True, True)



            asr_correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

            # Top-5 accuracy

            asr_top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()


        asr_top1_acc = 100. * asr_correct / total
        asr_top5_acc = 100. * asr_top5_correct / total

        print(
            'CoPur Test set: ASR Top-1 Accuracy: {}/{} ({:.4f}%), ASR Top-5 Accuracy: {}/{} ({:.4f}%)'
            .format(
                 asr_correct, total, asr_top1_acc, asr_top5_correct, total, asr_top5_acc ))

        return asr_top1_acc, asr_top5_acc


    def test_mul(self, test_data, criterion, device):
        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y) in enumerate(test_data):

                # if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                #     trn_X = trn_X.float().to(device)
                #     Xa, Xb = split_data(trn_X, args)
                #     target = trn_y.long().to(device)
                # else:
                #     #trn_X = [x.float().to(device) for x in trn_X]
                #     Xa = trn_X[0].float().to(device)
                #     Xb = trn_X[1].float().to(device)
                #     target = trn_y.long().to(device)
                trn_X = [x.float().to(device) for x in trn_X]
                # if isinstance(criterion, nn.CrossEntropyLoss):
                    # 为了避免错误，确保目标是长整型
                trn_y = trn_y.long().to(device)



                # bottom model B
                output_tensor_bottom_model_b = model_list[1](trn_X[1])
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](trn_X[0])

                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)


                    # 如果loss_func是CrossEntropyLoss的一个实例

                loss = criterion(output, trn_y)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += trn_y.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, trn_y.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, trn_y.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total


        print('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
            test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc

    def test_backdoor_mul(self, test_data, criterion, device, args, delta, poison_target_label):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0


        #batch_delta = delta.unsqueeze(0).repeat([poison_target_label.shape[0], 1, 1, 1]).detach().clone()

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    #trn_X = [x.float().to(device) for x in trn_X]
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                target_class = torch.tensor([poison_target_label]).repeat(target.shape[0]).to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb+delta)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)


                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()




        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total



        print('Backdoor Test set: Average loss: {:.4f}, ASR Top-1 Accuracy: {}/{} ({:.4f}%, ASR Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
            test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc


    def test(self, test_data, criterion, device):
        model_list = self.bottom_model_list + [self.top_model]

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        m = nn.Sigmoid()

        Loss = AverageMeter()
        # AUC = AverageMeter()
        ACC = AverageMeter()
        Precision = AverageMeter()
        Recall = AverageMeter()
        F1 = AverageMeter()
        # ASR_0 = AverageMeter()
        # ASR_1 = AverageMeter()
        # ARR_0 = AverageMeter()
        # ARR_1 = AverageMeter()

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        # all_target = np.array([])
        target_list = []
        pred_list = []
        with torch.no_grad():
            for batch_idx, (trn_X, target) in enumerate(test_data):
                trn_X = [x.float().to(device) for x in trn_X]
                target = target.float().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](trn_X[1])
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](trn_X[0])

                # top model
                pred = m(model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b))

                if isinstance(criterion, nn.CrossEntropyLoss):
                    # 为了避免错误，确保目标是长整型
                    target = target.long()

                loss = criterion(pred, target)

                y_hat_lbls, statistics = compute_correct_prediction(y_targets=target,
                                                                    y_prob_preds=pred,
                                                                    threshold=0.5)

                acc = accuracy_score(target.cpu().numpy(), y_hat_lbls)
                #auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())

                target_list.append(target.cpu().numpy())
                pred_list.append(pred.cpu().numpy())
                # try:
                #
                #     auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
                # except  ValueError:
                #     auc = 0
                #     pass

                metrics = precision_recall_fscore_support(target.cpu().numpy(), y_hat_lbls, average="binary",
                                                          warn_for=tuple())
                # print(classification_report(target.cpu().numpy(), y_hat_lbls))
                # cm = confusion_matrix(target.cpu().numpy(), y_hat_lbls)

                # poisoned_fn_rate, poisoned_tn_rate, poisoned_fp_rate, poisoned_tp_rate = backdoor_truepostive_rate(cm)
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
                # AUC.update(auc)
                Loss.update(loss)
                Precision.update(metrics[0])
                Recall.update(metrics[1])
                F1.update(metrics[2])
                # ASR_0.update(poisoned_fn_rate)
                # ASR_1.update(poisoned_fp_rate)
                # ARR_0.update(poisoned_tn_rate)
                # ARR_1.update(poisoned_tp_rate)
        # print(len(target_list))

        traget_list = np.concatenate(target_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        
        # 使用 numpy.nan_to_num() 函数将 NaN 值替换为 0
        pred_list_cleaned = np.nan_to_num(pred_list)

        auc =  roc_auc_score(traget_list, pred_list_cleaned)
        return Loss.avg, ACC.avg, auc, Precision.avg, Recall.avg, F1.avg

    def train_narcissus(self, train_data, criterion, bottom_criterion, optimizer_list, device, args, delta, poisoned_indices):
        # poisoned_indices: 是一个要毒害的index的list
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        poisoned_grads = []
        batch_poisoned_count= 0

        selected_set = set(poisoned_indices)

        #delta = torch.clamp(delta, -args.eps*2, args.eps*2)

        for step, (trn_X, trn_y, indices) in enumerate(train_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                #trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []

            indices = indices.cpu().numpy()

            batch_delta = torch.zeros_like(Xb).to(device)
            # mask标记哪些是要毒害的
            mask=torch.from_numpy(np.isin(indices, poisoned_indices))
            batch_delta[mask] = delta.detach().clone()
            batch_delta.requires_grad_()

            #batch_delta.grad.zero_()
            # bottom model B
            # 直接在上面加上随机扰动？
            output_tensor_bottom_model_b = model_list[1](Xb+batch_delta)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            # corruption_amp
            #output_tensor_bottom_model_a[mask]*=args.corruption_amp

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)


            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(grad_output_bottom_model_b)
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(grad_output_bottom_model_b, ratio=args.iso_ratio)
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(grad_output_bottom_model_b, target)
            # ----yjr-----
            # gradient compression
            if args.gc:
                grad_output_bottom_model_b = gradient_compression(grad_output_bottom_model_b, preserved_perc=args.gc_ratio)

            # differential privacy
            if args.lap_noise:
                grad_output_bottom_model_b = laplacian_noise_masking(grad_output_bottom_model_b, beta=args.lap_noise_ratio)

            # sign SGD
            # TODO: 这里的signSGD只是完成了在g上的操作，理论上应该使用optimizer.step()的方法来优化
            if args.signSGD:
                torch.sign(grad_output_bottom_model_b, out=grad_output_bottom_model_b)

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)



            # norm_attack_

            # Calculate the squared sum of the gradients along dim=1
            #squared_sum = grad_output_bottom_model_b.pow(2).sum(dim=1)

            # Ensure that the squared sum has no negative values
            #non_negative_squared_sum = torch.clamp(squared_sum, min=0)

            # Take the square root of the non-negative squared sum
            #g_norm = torch.sqrt(non_negative_squared_sum)



            # UAB attack
            batch_delta_grad = batch_delta.grad[mask]
            poisoned_grads.append(batch_delta_grad)

            #poisoned_grads = torch.cat(poisoned_grads, dim=0)

            #poison_optimizer.zero_grad()

            grad_sign = batch_delta_grad.detach().mean(dim=0).sign()

            delta = delta - grad_sign * args.alpha

            # delta.grad = poisoned_avg_grads.unsqueeze(0)
            #delta.backward(poisoned_avg_grads)

            delta = torch.clamp(delta, -args.eps, args.eps)


            #poison_optimizer.step()



            #poisoned_grads.append(batch_delta_grad)
            # l_inf

            '''
            # l2
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha

            delta_norms = torch.norm(delta.view(trn_X[1].shape[1], -1), p=2, dim=1)
            factor = args.eps/ delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor
            '''

            '''
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad = grad + momentum * args.decay
            momentum = grad.detach()

            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha
            '''

            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())

        '''
        poisoned_grads = torch.cat(poisoned_grads, dim=0)
        
        #poisoned_avg_grads = poisoned_grads.detach().mean(dim=0)

        #delta.grad = poisoned_avg_grads.unsqueeze(0)
        #delta.backward(poisoned_avg_grads.unsqueeze(0))

        #poison_optimizer.step()
        
        grad_sign = poisoned_grads.detach().mean(dim=0).sign()
        delta = delta - grad_sign * args.alpha
        # revise hyper-parameters self.alpha, self.eps and final adv_Xb clamp
        delta = torch.clamp(delta, -args.eps, args.eps)
        '''

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0], delta


    def train_backdoor(self, train_data, criterion, bottom_criterion, optimizer_list, device, args, delta, poisoned_indices):

        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]


        # train and update
        epoch_loss = []

        for step, (trn_X, trn_y, indices) in enumerate(train_data):

            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                #trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = torch.argmax(trn_y, dim=1).long().to(device)
            # target = trn_y.float().to(device)
            batch_loss = []


            batch_delta = delta.unsqueeze(0).repeat([target.shape[0], 1]).detach().clone()
            batch_delta.requires_grad_()

            #batch_delta.grad.zero_()
            # bottom model B
            # 这里是在每一个batch里面，对每一个样本进行都进行+delta的操作
            output_tensor_bottom_model_b = model_list[1](Xb+batch_delta)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # print(grad_output_bottom_model_a.size())
            # print(grad_output_bottom_model_b.size())
            # 下面内容貌似没有用到
            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(grad_output_bottom_model_b)
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(grad_output_bottom_model_b, ratio=args.iso_ratio)
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(grad_output_bottom_model_b, target)
                # U_B_gradients_list = marvell_g(U_B_gradients_list, target)
            # ----yjr-----
            # gradient compression
            if args.gc:
                U_B_gradients_list = gradient_compression(U_B_gradients_list, preserved_perc=args.gc_ratio)

            # differential privacy
            if args.lap_noise:
                U_B_gradients_list = laplacian_noise_masking(U_B_gradients_list, beta=args.lap_noise_ratio)

            # sign SGD
            # TODO: 这里的signSGD只是完成了在g上的操作，理论上应该使用optimizer.step()的方法来优化
            if args.signSGD:
                torch.sign(U_B_gradients_list, out=U_B_gradients_list)


            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)



            # norm_attack_
            g_norm = grad_output_bottom_model_b.pow(2).sum(dim=1).sqrt()
            # Calculate the squared sum of the gradients along dim=1
            #squared_sum = grad_output_bottom_model_b.pow(2).sum(dim=1)

            # Ensure that the squared sum has no negative values
            #non_negative_squared_sum = torch.clamp(squared_sum, min=0)

            # Take the square root of the non-negative squared sum
            #g_norm = torch.sqrt(non_negative_squared_sum)

            score = roc_auc_score(target.cpu().numpy(), g_norm.cpu().numpy())

            print("norm attack score: ", score)
            # target =0
            if args.negative_target:
                topk_values, topk_indices = torch.topk(g_norm.view(-1), k=args.Topk)
            else:
                # target =1
                topk_values, topk_indices = torch.topk(g_norm.view(-1), k=args.Topk, largest=False)

            # UAB attack
            batch_delta_grad = batch_delta.grad
            # l_inf
            grad_sign = batch_delta_grad[topk_indices].data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha
            # revise hyper-parameters self.alpha, self.eps and final adv_Xb clamp
            delta = torch.clamp(delta, -args.eps, args.eps)
            '''
            # l2
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha

            delta_norms = torch.norm(delta.view(trn_X[1].shape[1], -1), p=2, dim=1)
            factor = args.eps/ delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor
            '''



            '''
            grad_norms = torch.norm(adv_gradients[0][0].view(n, -1), p=2, dim=1) + 1e-10
            grad = adv_gradients[0][0] / grad_norms.view(n, 1)
            grad = grad + momentum * args.decay
            momentum = grad.detach()

            grad_sign = grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * args.alpha
            '''

            # to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
            #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss, delta

    def test_backdoor(self, test_data, criterion, device, delta):
        model_list = self.model

        model_list = [model.to(device) for model in model_list]

        model_list = [model.eval() for model in model_list]

        m = nn.Sigmoid()

        Loss = AverageMeter()
        AUC = AverageMeter()
        ACC = AverageMeter()
        Precision = AverageMeter()
        Recall = AverageMeter()
        F1 = AverageMeter()
        ASR_0 = AverageMeter()
        ASR_1 = AverageMeter()
        ARR_0 = AverageMeter()
        ARR_1 = AverageMeter()

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

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](trn_X[1]+delta)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](trn_X[0])

                # top model
                pred = m(model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b))

                loss = criterion(pred, target)

                y_hat_lbls, statistics = compute_correct_prediction(y_targets=target,
                                                                    y_prob_preds=pred,
                                                                    threshold=0.5)

                acc = accuracy_score(target.cpu().numpy(), y_hat_lbls)
                # auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())

                try:
                    auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
                except  ValueError:
                    pass

                metrics = precision_recall_fscore_support(target.cpu().numpy(), y_hat_lbls, average="binary",
                                                          warn_for=tuple())
                # print(classification_report(target.cpu().numpy(), y_hat_lbls))
                cm = confusion_matrix(target.cpu().numpy(), y_hat_lbls)

                poisoned_fn_rate, poisoned_tn_rate, poisoned_fp_rate, poisoned_tp_rate = backdoor_truepostive_rate(cm)
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
                ASR_0.update(poisoned_fn_rate)
                ASR_1.update(poisoned_fp_rate)
                ARR_0.update(poisoned_tn_rate)
                ARR_1.update(poisoned_tp_rate)

        return Loss.avg, ACC.avg, AUC.avg, Precision.avg, Recall.avg, F1.avg, ASR_0.avg, ASR_1.avg, ARR_0.avg, ARR_1.avg

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
