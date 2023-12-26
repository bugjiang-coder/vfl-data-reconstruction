import torch
import torch.nn as nn
import torch.nn.init as init
import collections
import torch.nn.functional as F


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)



# class BottomModelForCifar10(nn.Module):
#     def __init__(self):
#         super(BottomModelForCifar10, self).__init__()
#         self.resnet20 = resnet20(num_classes=10)

#     def forward(self, x):
#         x = self.resnet20(x)
#         return x

class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the first convolutional layer and ReLU activation function
        x = F.relu(self.conv11(x))
        split1 = x  # Split point after 1st convolutional layer

        # Apply the second convolutional layer
        x = self.conv12(x)
        x = self.pool(F.relu(x))
        return x


class CIFAR10CNNDecoder(nn.Module):
    def __init__(self):
        super(CIFAR10CNNDecoder, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.deconv11 = nn.ConvTranspose2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv11'] = self.deconv11

        self.ReLU11 = nn.ReLU()
        self.layerDict['ReLU11'] = self.ReLU11

        self.deconv21 = nn.ConvTranspose2d(
            in_channels = 64,
            out_channels = 3,
            kernel_size = 3,
            padding = 1
        )

        self.layerDict['deconv21'] = self.deconv21

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x

# class TopModelForCifar10(nn.Module):
#     def __init__(self):
#         super(TopModelForCifar10, self).__init__()
#         self.fc1top = nn.Linear(20, 20)
#         self.fc2top = nn.Linear(20, 10)
#         self.fc3top = nn.Linear(10, 10)
#         self.fc4top = nn.Linear(10, 10)
#         self.bn0top = nn.BatchNorm1d(20)
#         self.bn1top = nn.BatchNorm1d(20)
#         self.bn2top = nn.BatchNorm1d(10)
#         self.bn3top = nn.BatchNorm1d(10)

#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         x = self.fc4top(F.relu(self.bn3top(x)))
#         return F.log_softmax(x, dim=1)

class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Define the 2 fully connected layers
        # The input features to the first fully connected layer will change
        # because the width of the image is halved
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # Adjusted for halved image width
        self.fc2 = nn.Linear(1024, 10)

        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        # print(input_tensor_top_model_a.shape) torch.Size([64, 64, 16, 8])
        # print(input_tensor_top_model_b.shape)
        # sys.exit(0)
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=3)
        # print(output_bottom_models.shape) torch.Size([64, 64, 16, 16])
        # sys.exit(0)
        x = output_bottom_models
        x = F.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.pool(F.relu(x))

        # Apply fifth and sixth convolutional layers
        x = F.relu(self.conv31(x))
        x = self.conv32(x)
        x = self.pool(F.relu(x))

        # Flatten the output for the fully connected layer
        # The flattening process needs to take into account the halved width
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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


