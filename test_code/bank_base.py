import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

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


class bank_dataset(Dataset):
    def __init__(self, X, y):
        # data = data
        self.X = X
        self.y = y
    def __getitem__(self, item):
        X = self.X.iloc[[item]].values.reshape(-1)
        y = self.y.iloc[[item]].values.reshape(-1)
        return np.float32(X), np.float32(y)

    def __len__(self):
        return len(self.X)

class fcn(nn.Module):
    def __init__(self):
        super(fcn, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.model(x)

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

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("################################ Load Data ############################")
    dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/bank/bank-additional/bank_cleaned.csv'
    df = pd.read_csv(dataPath, delimiter=',')

    df_train = df.sample(frac=0.8, random_state=0)
    df_test = df.drop(df_train.index)

    X_train = df_train.drop(df_train.columns[-1], axis=1)
    y_train = df_train.iloc[:, -1:]
    X_test = df_test.drop(df_train.columns[-1], axis=1)
    y_test = df_test.iloc[:, -1:]

    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    train_dataset = bank_dataset(X_train, y_train)
    test_dataset = bank_dataset(X_test, y_test)

    train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                              drop_last=False)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,
                                            drop_last=False)

    print("################################ Set Models, optimizer, loss ############################")

    model = fcn()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    print("################################ Start Training ############################")

    for epoch in range(100):
        model.train()

        batch_loss = []
        for i, (X, y) in enumerate(train_queue):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        print("epoch: {0}, loss: {1}".format(epoch, np.mean(batch_loss)))
        model.eval()
        with torch.no_grad():
            Loss = AverageMeter()
            AUC = AverageMeter()
            ACC = AverageMeter()
            Precision = AverageMeter()
            Recall = AverageMeter()
            F1 = AverageMeter()

            for i, (X, y) in enumerate(test_queue):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)

                y_hat_lbls, statistics = compute_correct_prediction(y_targets=y,
                                                                    y_prob_preds=y_pred,
                                                                    threshold=0.5)
                acc = accuracy_score(y.cpu().numpy(), y_hat_lbls)
                # auc = roc_auc_score(y.cpu().numpy(), y_pred.cpu().numpy())

                ACC.update(acc)
                # AUC.update(auc)
                Loss.update(loss)
                # Precision.update(metrics[0])
                # Recall.update(metrics[1])
                # F1.update(metrics[2])
                #
            # print("y_pred:", y_pred)
                # if i % 100 == 0:
            print("epoch: {0}, loss: {1}, acc: {2}".format(epoch, Loss.avg, ACC.avg))
                #     print("epoch: {0}, batch: {1}, loss: {2}".format(epoch, i, loss.item()))



