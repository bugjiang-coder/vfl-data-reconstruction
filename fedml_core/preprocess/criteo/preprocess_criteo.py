import os
import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def preprocess(data_dir, fraction_to_keep=0.01):
    # 13列整数特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    # 取出dense_features的奇数列和偶数列
    Xa_dense_features, Xb_dense_features = dense_features[::2], dense_features[1::2]

    # 26列分类特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    # 取出sparse_features的奇数列和偶数列
    Xa_sparse_features, Xb_sparse_features = sparse_features[::2], sparse_features[1::2]

    party_a_feat_list, party_b_feat_list = Xa_dense_features + Xa_sparse_features, Xb_dense_features + Xb_sparse_features

    file_path = data_dir + "processed_criteo.csv"

    if os.path.exists(file_path):
        # 读取缓存数据
        processed_criteo_df = pd.read_csv(file_path, low_memory=False)
        processed_criteo_df = processed_criteo_df.sample(frac=0.1, replace=False, random_state=1)
        criteo_df = shuffle(processed_criteo_df, random_state=47)


    else:
        file_path = data_dir + "train.txt"
        # 读取数据
        full_data = pd.read_csv(file_path, header=None, names=["label"] + dense_features + sparse_features, sep='\t')

        full_data[sparse_features] = full_data[sparse_features].fillna('', )
        full_data[dense_features] = full_data[dense_features].fillna(0, )

        # --------------分类特征的标签进行编码------------
        label_encoder_dict = {}
        for feat in sparse_features:
            lbe = LabelEncoder()  # 对值介于0和n_classes-1之间的目标标签进行编码。
            full_data.loc[:, feat] = lbe.fit_transform(full_data[feat])  # fit label encoder and return encoded label
            full_data.loc[:, feat] = full_data[feat].astype(np.int32)  # convert from float64 to float32
            label_encoder_dict[feat] = lbe  # store the fitted label encoder


        # -----------------------对整数特征执行简单转换-------------------------
        mms = MinMaxScaler(feature_range=(0, 1))
        full_data.loc[:, dense_features] = mms.fit_transform(full_data[dense_features])
        full_data.loc[:, dense_features] = full_data[dense_features].astype(np.float32)

        # for key in dense_features:
        #     print(key)
        #     print(np.max(full_data[key]), np.min(full_data[key]))

        # 对处理好的数据进行采样、同时保存
        criteo_df = full_data.sample(frac=fraction_to_keep, replace=False, random_state=1)
        criteo_df.to_csv(path_or_buf=data_dir + "processed_criteo.csv", index=False)


    # -----------------------划分训练集和测试集-------------------------

    X, Y = criteo_df[dense_features+sparse_features], criteo_df['label']
    Xa, Xb, y = X[party_a_feat_list].values, X[party_b_feat_list].values, Y.values
    y = np.expand_dims(y, axis=1)

    n_train = int(0.8 * Xa.shape[0])
    print("# of train samples:", n_train)

    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]

    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]

    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]




if __name__ == "__main__":
    data_path = "/home/yangjirui/feature-infer-workspace/dataset/criteo/"

    # 直接输出到源文件夹下
    # 这里只选取了0.001的数据，并没有使用全部数据
    # 由于数据过大，有约10个GB
    preprocess(data_dir=data_path)



