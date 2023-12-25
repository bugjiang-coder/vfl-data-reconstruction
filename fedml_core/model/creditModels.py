import torch
import torch.nn as nn
import torch.nn.init as init
import collections


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class active_model(nn.Module):
    # 主动方负责将两方的模型拼接起来，同时计算最后的分类
    def __init__(self, input_dim, output_dim, k=2):
        super(active_model, self).__init__()

        self.bottom_model = nn.Sequential(
            nn.Linear(input_dim, 300),
            # nn.LeakyReLU(),
            # nn.Linear(300, 100)
        )

        self.top_model = nn.Sequential(
            # nn.LeakyReLU(),
            nn.Linear(300 * k, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 10),
            nn.LeakyReLU(),
            nn.Linear(10, output_dim)
        )

    def forward(self, input, U_B):
        out = self.bottom_model(input)

        if U_B is not None:
            out = torch.cat([out] + [U for U in U_B], dim=1)
            # out = torch.cat((out, U_B), dim=1)
        logits = self.top_model(out)
        return logits

    def getLayerOutput(self, x, targetLayer):
        # 注意这个是主动方的本地模型
        return self.bottom_model[0:targetLayer](x)

    def getLayerNum(self):
        return len(self.bottom_model)


class passive_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(passive_model, self).__init__()
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 300),
        )

    def forward(self, input):
        return self.local_model(input)
        # return torch.zeros_like(self.local_model(input))*self.local_model(input)

    def getLayerOutput(self, x, targetLayer):
        # 注意这个是主动方的本地模型
        return self.local_model[0:targetLayer](x)

    def getLayerNum(self):
        return len(self.local_model)


class BottomModel(nn.Module):
    def __init__(self, input_dim, output_dim=100):
        super(BottomModel, self).__init__()
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 100),
            nn.LeakyReLU(),
            nn.Linear(100, output_dim),
            # nn.Linear(input_dim, output_dim)
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.local_model(input)
        # return torch.zeros_like(self.local_model(input))*self.local_model(input)

    def getLayerOutput(self, x, targetLayer):
        # 注意这个是主动方的本地模型
        return self.local_model[0:targetLayer](x)

    def getLayerNum(self):
        return len(self.local_model)


class TopModel(nn.Module):
    def __init__(self, input_dim=200, output_dim=1):
        super(TopModel, self).__init__()

        self.local_model = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, 100),
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 10),
            # nn.BatchNorm1d(10),
            nn.LeakyReLU(),
            nn.Linear(10, output_dim),
        )

        self.apply(weights_init)

    def forward(self, model_a_output, model_b_output):
        input = torch.cat((model_a_output, model_b_output), dim=1)
        return self.local_model(input)

    def getLayerOutput(self, x, targetLayer):
        # 注意这个是主动方的本地模型
        return self.local_model[0:targetLayer](x)

    def getLayerNum(self):
        return len(self.local_model)


class BottomModelDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BottomModelDecoder, self).__init__()
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.Linear(300, output_dim),
            # nn.Linear(input_dim, output_dim)
        )

    def forward(self, input):
        return self.local_model(input)


class BottomModelDecoder_layer1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BottomModelDecoder_layer1, self).__init__()
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, input):
        return self.local_model(input)

class BottomModelDecoder_layer2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BottomModelDecoder_layer2, self).__init__()
        self.local_model = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.LeakyReLU(),
            nn.Linear(300, output_dim),
            # nn.Linear(input_dim, output_dim)
        )

    def forward(self, input):
        return self.local_model(input)