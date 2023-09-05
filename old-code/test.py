from torch.utils.tensorboard import SummaryWriter
import torch
import numpy

# 实例化SummaryWriter对象
# writer = SummaryWriter('log')

base_path = '/home/yangjirui/paper-code/model/adult/'

models =  base_path + '2layers_1.pth'
# 加载模型51-->10
model = torch.load(models)
# 记录模型参数
for name, param in model.named_parameters():
    print(name, param)
    print(param.shape)
    # writer.add_histogram(name, param, 0)

# 构造全0输入数据
input = torch.zeros(1, 51)

model.to('cpu')
# 记录模型输出
output0 = model(input)
output0 = output0.detach().numpy()
outputlist =[]

diff = []

for i in range(51):
    input = torch.zeros(1, 51)
    input[0][i] = 1
    output = model(input)
    output = output.detach().numpy()
    diff.append(numpy.linalg.norm(output0 - output))
    outputlist.append(output)


# 计算每个特征的影响
# for i in range(51):
#     # 计算欧式距离
#
#
#     print(i, numpy.linalg.norm(outputlist[i] - output))
#     # print(i, outputlist[i] - outputlist[0])

for i, j in enumerate(diff):
    print(i, j)

# 找出outputlist中最大的10个数的索引
print(numpy.argsort(diff)[-10:])




