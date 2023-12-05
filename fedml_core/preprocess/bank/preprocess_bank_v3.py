import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/bank/bank-additional/bank_cleaned.csv'

# 注意该版本的preprocess 不使用one-hot编码
#
def to_int(df, col_features):
    # 创建一个LabelEncoder对象
    # label_encoder = LabelEncoder()
    # 使用LabelEncoder对特征进行整数编码
    int_df = df[col_features]
    # onehot_df = pd.get_dummies(df[col_features])
    int_features = int_df.columns.values
    discrete_index = {s: [i for i in range(len(int_features)) if s in int_features[i]] for s in col_features}

    # 将离散特征缩放到[0,1]之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    # int_df = scaler.fit_transform(int_df)
    int_df = pd.DataFrame(scaler.fit_transform(int_df), columns=int_features)

    return int_df, discrete_index


def preprocess(dataPath):
    print("===============processing data===============")

    df = pd.read_csv(dataPath, delimiter=',')
    # df.head()
    df.info()
    # 对数据进行打乱
    df = df.sample(frac=1, random_state=0)

    # 处理分类特征
    cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                 'poutcome']

    # 对cate_cols进行打乱
    # cate_cols = shuffle(cate_cols, random_state=0)
    # random.shuffle(cate_cols)
    # print("cate_cols:", cate_cols)

    # df['y'] = (df['y'] == 'yes').astype(int)

    df_object_col = cate_cols
    df_num_col = [col for col in df.columns if col not in cate_cols and col != 'y']
    target = df['y']

    # 连续列缩放到[0,1]之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[df_num_col] = scaler.fit_transform(df[df_num_col])

    # -----------------------划分训练集和测试集-------------------------
    # Xa, Xa_index = to_int(df, df_object_col[::2])
    # print(df_object_col[::2])
    Xa = df[df_object_col[::2]]
    # Xa = df[df_object_col[:1]]
    Xa = pd.concat([Xa, df[df_num_col[::2]]], axis=1).values
    # Xa = pd.concat([Xa, df[df_num_col[:1]]], axis=1).values

    # Xb, Xb_index = to_int(df, df_object_col[1::2])
    Xb = df[df_object_col[1::2]]
    # Xb = df[df_object_col[1:]]
    Xb = pd.concat([Xb, df[df_num_col[1::2]]], axis=1).values
    # Xb = pd.concat([Xb, df[df_num_col[1:]]], axis=1).values

    y = target.values
    y = np.expand_dims(y, axis=1)

    n_train = int(0.8 * Xa.shape[0])

    # 对Xa和Xb进行打乱
    # np.random.shuffle(Xa)
    # np.random.shuffle(Xb)
    # np.random.shuffle(y)
    # Xb = shuffle(Xb, random_state=0)
    # y = shuffle(y, random_state=0)

    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]
    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    # print("Xa_onehot_index:", Xa_index)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xb_test.shape:", Xb_test.shape)
    # print("Xb_onehot_index:", Xb_index)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    print("===============processing data end===============")

    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]


if __name__ == "__main__":
    # dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/bank/bank-additional/bank-additional-full.csv'
    dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/bank/bank-additional/bank_cleaned.csv'
    # df = pd.read_csv(dataPath, delimiter=';')

    # print(df.sample(10))
    # print(df.shape)
    # print(df.info())
    # print(df.describe())
    # print(df.head())
    # print(df.columns.values)


    [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(dataPath)
    for i in range(1):
        print(Xa_train[i])
        print(Xb_train[i])
        print(y_train[i])

    # print(Xa_train[10:])
    # print(y_train[10:])