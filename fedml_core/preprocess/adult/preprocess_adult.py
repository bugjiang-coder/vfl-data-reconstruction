import pandas as pd
import numpy as np


dataPath = '/home/yangjirui/feature-infer-workspace/dataset/adult/adult.data'

def preprocess(dataPath):
    df = pd.read_csv(dataPath, header = None, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
    df.head()
    df.info()

    df.apply(lambda x : np.sum(x == " ?"))


    df.replace(" ?", pd.NaT, inplace = True)
    df.replace(" >50K", 1, inplace = True)
    df.replace(" <=50K", 0, inplace = True)
    trans = {'workclass' : df['workclass'].mode()[0], 'occupation' : df['occupation'].mode()[0], 'native-country' : df['native-country'].mode()[0]}
    df.fillna(trans, inplace = True)
    print(df.describe())


    df.drop('fnlwgt', axis = 1, inplace = True)
    df.drop('capital-gain', axis = 1, inplace = True)
    df.drop('capital-loss', axis = 1, inplace = True)



    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']
    target = df["income"]
    dataset = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis = 1)

    headers = list(dataset.columns.values)
    Xa_features, Xb_features = headers[::2], headers[1::2]

    # -----------------------划分训练集和测试集-------------------------

    X, Y = dataset, df["income"]
    Xa, Xb, y = X[Xa_features].values, X[Xb_features].values, Y.values
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
    data_dir = '/home/yangjirui/VFL/feature-infer-workspace/dataset/adult/adult.data'

    [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test] = preprocess(data_dir)
    for i in range(100):
        print(Xb_train[i])