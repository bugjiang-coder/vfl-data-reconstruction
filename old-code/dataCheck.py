from sklearn.mixture import GaussianMixture
import pandas as pd


# pandas 读取数据
file = "/home/yangjirui/paper-code/data/zhongyuan/norm-2layer/DGM/inverse_data-g3.csv"
file2 = "/home/yangjirui/paper-code/data/zhongyuan/norm-2layer/DGM/origin_data-g.csv"
data = pd.read_csv(file, header=None,index_col=None)



bool1 = data.iloc[:, 15].to_frame()

# # 生成数据
# data = pd.DataFrame({
#                      'feature2': [0, 1, 1, 0, 1, 0]})

# 使用 GMM 模型拟合数据
gmm = GaussianMixture(n_components=2)
gmm.fit(bool1)

# 获取每个高斯分布的参数
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

print("Means:", means)
print("Covariances:", covariances)
print("Weights:", weights)

# 生成：
# Means:[[-0.118672061] [2.12427654]]
# Covariances：[0.269046351] [[3.00914274]]
# Weights: [0.68093547 0.31906453]

# 真实
# Means: [[-0.8406997] [ 1.1894854]]
# Covariances: [[[1.e-06]] [[1.e-06]]]
# Weights: [0.58718134 0.41281866]