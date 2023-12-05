import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataPath = '/home/yangjirui/data/vfl-tab-reconstruction/dataset/adult/adult.data'


def to_onehot(df, col_features):
    # 对类别型特征进行one-hot编码,并返回离散特征的索引
    onehot_df = pd.get_dummies(df[col_features])
    onehot_features = onehot_df.columns.values
    discrete_index = {s: [i for i in range(len(onehot_features)) if s in onehot_features[i]] for s in col_features}

    return onehot_df, discrete_index


def preprocess(dataPath):
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

    # 连续列缩放到[-1,1]之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[df_int_col] = scaler.fit_transform(df[df_int_col])

    # -----------------------划分训练集和测试集-------------------------
    Xa, Xa_index = to_onehot(df, df_object_col[::2])
    Xa = pd.concat([Xa, df[df_int_col[::2]]], axis=1).values

    Xb, Xb_index = to_onehot(df, df_object_col[1::2])
    Xb = pd.concat([Xb, df[df_int_col[1::2]]], axis=1).values

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

    [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(dataPath)
    for i in range(1):
        print(Xb_train[i])
