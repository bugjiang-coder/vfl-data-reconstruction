import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/credit/clients.xls'


def to_onehot(df, col_features):
    # 对类别型特征进行one-hot编码,并返回离散特征的索引
    onehot_df = pd.get_dummies(df[col_features])
    onehot_features = onehot_df.columns.values
    discrete_index = {s: [i for i in range(len(onehot_features)) if s in onehot_features[i]] for s in col_features}

    return onehot_df, discrete_index


def preprocess(dataPath):
    print("===============processing data===============")

    df = pd.read_excel(dataPath, index_col=0, header=1)
    # df = pd.read_csv(dataPath, delimiter=';')
    # df.head()
    df.info()
    # 对数据进行打乱
    df = df.sample(frac=1, random_state=0)

    # 处理分类特征
    cate_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
                 ]

    # df['y'] = (df['y'] == 'yes').astype(int)

    df[cate_cols] = df[cate_cols].astype('object')
    df_object_col = cate_cols
    df_num_col = [col for col in df.columns if col not in cate_cols and col != 'default payment next month']
    target = df['default payment next month']

    # 连续列缩放到[-1,1]之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[df_num_col] = scaler.fit_transform(df[df_num_col])

    # -----------------------划分训练集和测试集-------------------------
    Xa, Xa_index = to_onehot(df, df_object_col[::2])
    Xa = pd.concat([Xa, df[df_num_col[::2]]], axis=1).values

    Xb, Xb_index = to_onehot(df, df_object_col[1::2])
    Xb = pd.concat([Xb, df[df_num_col[1::2]]], axis=1).values

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
    dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/credit/clients.xls'

    # df = pd.read_csv(dataPath, delimiter=';')

    df = pd.read_excel(dataPath, index_col=0, header=1)
    # df = pd.read_csv(dataPath, delimiter=';')
    # df.head()
    df.info()

    print(df.sample(10))
    # print(df.shape)
    # print(df.info())
    print(df.describe())
    # print(df.head())
    print(df.iloc[0:10, :])
    print(df.columns.values)

    [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(dataPath)
    # for i in range(10):
    #     print(Xb_train[i])
    #
    # print(Xa_train[10:])
    # print(y_train[10:])
