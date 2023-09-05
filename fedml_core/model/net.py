import torch
import torch.nn as nn
import collections
import torch.optim as optim

# =================================VFL基础net================================
class active_model(nn.Module):
    # 主动方负责将两方的模型拼接起来，同时计算最后的分类
    def __init__(self, input_dim, intern_dim, num_classes, k=2):
        super(active_model, self).__init__()

        self.local_model = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(intern_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(intern_dim, intern_dim),
            nn.LeakyReLU()
        )
        self.classifier = nn.Linear(in_features=intern_dim*k, out_features=num_classes)

    def forward(self, input, U_B):
        out = self.local_model(input)

        if U_B is not None:
            out = torch.cat([out] + [U for U in U_B], dim=1)
            #out = torch.cat((out, U_B), dim=1)
        logits = self.classifier(out)
        return logits

class passive_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model, self).__init__()
        self.layerDict = collections.OrderedDict()

        self.linear1 = nn.Linear(input_dim, intern_dim)
        self.layerDict['linear1'] = self.linear1

        self.ReLU1 = nn.LeakyReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.linear2 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear2'] = self.linear2

        self.ReLU2 = nn.LeakyReLU()
        self.layerDict['ReLU2'] = self.ReLU2

        self.linear3 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear3'] = self.linear3

        self.ReLU3 = nn.LeakyReLU()
        self.layerDict['ReLU3'] = self.ReLU3

        self.linear4 = nn.Linear(intern_dim, output_dim, bias=False)
        self.layerDict['linear4'] = self.linear4

    def forward(self, input):
        for layer in self.layerDict:
            input = self.layerDict[layer](input)
        return input

    def getLayerOutput(self, x, targetLayer):
        # 输出targetLayer层之前的结果
        if targetLayer == 'input':
            return x

        for layer in self.layerDict:
            if layer == targetLayer:
                return x
            x = self.layerDict[layer](x)

    def fromLayerForward(self, x, targetLayer):  # 从某层往后推
        if targetLayer == 'output':
            return x
        elif targetLayer == 'input':
            return self.forward(x)

        flag = False
        for layer in self.layerDict:
            if layer == targetLayer:
                flag = True
            if flag:
                x = self.layerDict[layer](x)
        return x

    def getLayerOut(self, x, targetLayer):
        # 输出某一层的结果
        if targetLayer == 'output':
            return x
        for layer in self.layerDict:
            if layer == targetLayer:
                return self.layerDict[layer](x)
        return x

class passive_model_layer3(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model_layer3, self).__init__()
        self.layerDict = collections.OrderedDict()

        self.linear1 = nn.Linear(input_dim, intern_dim)
        self.layerDict['linear1'] = self.linear1

        self.ReLU1 = nn.LeakyReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.linear2 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear2'] = self.linear2

        self.ReLU2 = nn.LeakyReLU()
        self.layerDict['ReLU2'] = self.ReLU2


        self.linear4 = nn.Linear(intern_dim, output_dim, bias=False)
        self.layerDict['linear3'] = self.linear4

    def forward(self, input):
        for layer in self.layerDict:
            input = self.layerDict[layer](input)
        return input

    def getLayerOutput(self, x, targetLayer):
        # 输出targetLayer层之前的结果
        if targetLayer == 'input':
            return x

        for layer in self.layerDict:
            if layer == targetLayer:
                return x
            x = self.layerDict[layer](x)

    def fromLayerForward(self, x, targetLayer):  # 从某层往后推
        if targetLayer == 'output':
            return x
        elif targetLayer == 'input':
            return self.forward(x)

        flag = False
        for layer in self.layerDict:
            if layer == targetLayer:
                flag = True
            if flag:
                x = self.layerDict[layer](x)
        return x

    def getLayerOut(self, x, targetLayer):
        # 输出某一层的结果
        if targetLayer == 'output':
            return x
        for layer in self.layerDict:
            if layer == targetLayer:
                return self.layerDict[layer](x)
        return x

class passive_model_layer5(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model_layer5, self).__init__()
        self.layerDict = collections.OrderedDict()

        self.linear1 = nn.Linear(input_dim, intern_dim)
        self.layerDict['linear1'] = self.linear1

        self.ReLU1 = nn.LeakyReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.linear2 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear2'] = self.linear2

        self.ReLU2 = nn.LeakyReLU()
        self.layerDict['ReLU2'] = self.ReLU2

        self.linear3 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear3'] = self.linear3

        self.ReLU3 = nn.LeakyReLU()
        self.layerDict['ReLU3'] = self.ReLU3

        self.linear4 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear4'] = self.linear4

        self.ReLU4 = nn.LeakyReLU()
        self.layerDict['ReLU4'] = self.ReLU4


        self.linear5 = nn.Linear(intern_dim, output_dim, bias=False)
        self.layerDict['linear5'] = self.linear5

    def forward(self, input):
        for layer in self.layerDict:
            input = self.layerDict[layer](input)
        return input

    def getLayerOutput(self, x, targetLayer):
        # 输出targetLayer层之前的结果
        if targetLayer == 'input':
            return x

        for layer in self.layerDict:
            if layer == targetLayer:
                return x
            x = self.layerDict[layer](x)

    def fromLayerForward(self, x, targetLayer):  # 从某层往后推
        if targetLayer == 'output':
            return x
        elif targetLayer == 'input':
            return self.forward(x)

        flag = False
        for layer in self.layerDict:
            if layer == targetLayer:
                flag = True
            if flag:
                x = self.layerDict[layer](x)
        return x

    def getLayerOut(self, x, targetLayer):
        # 输出某一层的结果
        if targetLayer == 'output':
            return x
        for layer in self.layerDict:
            if layer == targetLayer:
                return self.layerDict[layer](x)
        return x

class passive_model_layer6(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model_layer6, self).__init__()
        self.layerDict = collections.OrderedDict()

        self.linear1 = nn.Linear(input_dim, intern_dim)
        self.layerDict['linear1'] = self.linear1

        self.ReLU1 = nn.LeakyReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.linear2 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear2'] = self.linear2

        self.ReLU2 = nn.LeakyReLU()
        self.layerDict['ReLU2'] = self.ReLU2

        self.linear3 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear3'] = self.linear3

        self.ReLU3 = nn.LeakyReLU()
        self.layerDict['ReLU3'] = self.ReLU3

        self.linear4 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear4'] = self.linear4

        self.ReLU4 = nn.LeakyReLU()
        self.layerDict['ReLU4'] = self.ReLU4

        self.linear5 = nn.Linear(intern_dim, intern_dim)
        self.layerDict['linear5'] = self.linear5

        self.ReLU5 = nn.LeakyReLU()
        self.layerDict['ReLU5'] = self.ReLU5


        self.linear6 = nn.Linear(intern_dim, output_dim, bias=False)
        self.layerDict['linear6'] = self.linear6

    def forward(self, input):
        for layer in self.layerDict:
            input = self.layerDict[layer](input)
        return input

    def getLayerOutput(self, x, targetLayer):
        # 输出targetLayer层之前的结果
        if targetLayer == 'input':
            return x

        for layer in self.layerDict:
            if layer == targetLayer:
                return x
            x = self.layerDict[layer](x)

    def fromLayerForward(self, x, targetLayer):  # 从某层往后推
        if targetLayer == 'output':
            return x
        elif targetLayer == 'input':
            return self.forward(x)

        flag = False
        for layer in self.layerDict:
            if layer == targetLayer:
                flag = True
            if flag:
                x = self.layerDict[layer](x)
        return x

    def getLayerOut(self, x, targetLayer):
        # 输出某一层的结果
        if targetLayer == 'output':
            return x
        for layer in self.layerDict:
            if layer == targetLayer:
                return self.layerDict[layer](x)
        return x


class passive_model_easy(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model_easy, self).__init__()
        self.layerDict = collections.OrderedDict()

        self.linear1 = nn.Linear(input_dim, intern_dim)
        self.layerDict['linear1'] = self.linear1

        # self.ReLU1 = nn.LeakyReLU()
        # self.layerDict['ReLU1'] = self.ReLU1

        self.linear4 = nn.Linear(intern_dim, output_dim, bias=False)
        self.layerDict['linear2'] = self.linear4

    def forward(self, input):
        for layer in self.layerDict:
            input = self.layerDict[layer](input)
        return input

    def getLayerOutput(self, x, targetLayer):
        # 输出targetLayer层之前的结果
        if targetLayer == 'input':
            return x

        for layer in self.layerDict:
            if layer == targetLayer:
                return x
            x = self.layerDict[layer](x)

    def fromLayerForward(self, x, targetLayer):  # 从某层往后推
        if targetLayer == 'output':
            return x
        elif targetLayer == 'input':
            return self.forward(x)

        flag = False
        for layer in self.layerDict:
            if layer == targetLayer:
                flag = True
            if flag:
                x = self.layerDict[layer](x)
        return x

    def getLayerOut(self, x, targetLayer):
        # 输出某一层的结果
        if targetLayer == 'output':
            return x
        for layer in self.layerDict:
            if layer == targetLayer:
                return self.layerDict[layer](x)
        return x


# 该类未使用
class passive_classfication_head(nn.Module):
    def __init__(self, model, intern_dim, num_classes):
        super(passive_classfication_head, self).__init__()
        self.model = model
        '''
        for p in self.parameters():
            p.requires_grad = False
        '''
        #self.classifier = nn.Linear(in_features=intern_dim, out_features=num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(intern_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=num_classes, bias=False)
        )
        # Q:神经网络最后一层的bias为什么要设置为false?
        # 如果最后一层的bias不为0，那么Softmax函数将不再保证所有类别的概率之和为1。
        # 因此，为了确保模型的输出满足条件，我们需要将最后一层的bias设置为False
        # 在一些情况下我们可以使用带bias的Linear层，然后在最后接一个Softmax函数来实现同样的效果
        # 但是为了代码的简洁和效率，有时候我们会选择将bias直接设置为False

    def forward(self, input):
        out = self.classifier(self.model(input))
        return out

# =================================VFL基础net================================



# =================================VFL攻击net================================
class generator_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(generator_model, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(intern_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(intern_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(intern_dim, output_dim)
        )


    def forward(self, input):
        out = self.MLP(input)
        return out
# =================================VFL攻击net================================