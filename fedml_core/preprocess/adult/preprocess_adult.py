import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/adult/adult.data'


def to_onehot(df, col_features):
    # 对类别型特征进行one-hot编码,并返回离散特征的索引
    if len(col_features) == 0:
        return None, {}
    onehot_df = pd.get_dummies(df[col_features])
    onehot_features = onehot_df.columns.values
    discrete_index = {s: [i for i in range(len(onehot_features)) if s in onehot_features[i]] for s in col_features}

    return onehot_df, discrete_index


def preprocess(dataPath, A_ratio=0.5):
    # 注意默认都是重建B
    print("===============processing data===============")

    df = pd.read_csv(dataPath, header=None,
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                            'native-country', 'income'])
    # df.head()
    df.info()
    # 对数据进行打乱
    df = df.sample(frac=1, random_state=0)

    # df.apply(lambda x: print(np.sum(x == " ?")))  有缺失值的共三列

    df.replace(" ?", pd.NaT, inplace=True)
    df.replace(" >50K", 1, inplace=True)
    df.replace(" <=50K", 0, inplace=True)

    trans = {'workclass': df['workclass'].mode()[0], 'occupation': df['occupation'].mode()[0],
             'native-country': df['native-country'].mode()[0]}
    df.fillna(trans, inplace=True)

    # 去除相关性低的列
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('capital-gain', axis=1, inplace=True)
    df.drop('capital-loss', axis=1, inplace=True)

    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']
    target = df["income"]

    # 为了复现GRNA算法准备数据集
    # # 创建LabelEncoder对象
    # label_encoder = LabelEncoder()
    #
    # # 针对每个分类变量，使用LabelEncoder进行标签编码
    # for col in df_object_col:
    #     df[col] = label_encoder.fit_transform(df[col])
    #
    # # 确定 object 列和 int 列的数量
    # num_object_cols = len(df_object_col)
    # num_int_cols = len(df_int_col)
    #
    # # 确定生成 DataFrame 的行数
    # num_rows = max(num_object_cols, num_int_cols)
    #
    # # 创建一个空的 DataFrame
    # new_df = pd.DataFrame()
    #
    # # 依次从 object 列和 int 列中取值，交替填充新的 DataFrame
    # for i in range(num_rows):
    #     if i < num_object_cols:
    #         new_df[df_object_col[i]] = df[df_object_col[i]]
    #     if i < num_int_cols:
    #         new_df[df_int_col[i]] = df[df_int_col[i]]
    #
    # # 添加目标列
    # new_df["income"] = target
    #
    # # 打印新的 DataFrame
    # print(new_df.head())
    #
    # # 报存新的 DataFrame
    # new_df.to_csv("adult-2.csv", index=False)
    # sys.exit(0)

    # 连续列缩放到[-1,1]之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[df_int_col] = scaler.fit_transform(df[df_int_col])

    # -----------------------划分训练集和测试集-------------------------

    if A_ratio == 0.5:
        Xa, Xa_index = to_onehot(df, df_object_col[::2])
        Xa = pd.concat([Xa, df[df_int_col[::2]]], axis=1).values

        Xb, Xb_index = to_onehot(df, df_object_col[1::2])
        Xb = pd.concat([Xb, df[df_int_col[1::2]]], axis=1).values
    else:
        Xa, Xa_index = to_onehot(df, df_object_col[:int(A_ratio * len(df_object_col))])
        Xa = pd.concat([Xa, df[df_int_col[:int(A_ratio * len(df_int_col))]]], axis=1).values

        Xb, Xb_index = to_onehot(df, df_object_col[int(A_ratio * len(df_object_col)):])
        Xb = pd.concat([Xb, df[df_int_col[int(A_ratio * len(df_int_col)):]]], axis=1).values

    y = target.values
    y = np.expand_dims(y, axis=1)

    n_train = int(0.8 * Xa.shape[0])

    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]
    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xa_onehot_index:", Xa_index)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("Xb_onehot_index:", Xb_index)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    print("===============processing data end===============")

    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]


if __name__ == "__main__":
    dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/adult/adult.data'

    # print("radio 0.5")
    # [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(dataPath)
    # for i in range(1):
    #     print(Xb_train[i])


    print("radio 0.1")

    [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(dataPath, A_ratio=0.13)
    for i in range(1):
        print(Xb_train[i])

    print("radio 0.9")

    [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(dataPath, A_ratio=0.9)
    for i in range(1):
        print(Xb_train[i])


