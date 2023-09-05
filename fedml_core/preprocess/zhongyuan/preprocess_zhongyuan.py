import os
import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

try:
    from preprocess.zhongyuan.zhongyuan_feature_group import all_feature_list, qualification_feat, \
        loan_feat, debt_feat, repayment_feat
except ImportError:
    from zhongyuan_feature_group import all_feature_list, qualification_feat, \
        loan_feat, debt_feat, repayment_feat

work_year_map = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
}

class_map = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7
}


def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_scaled = (x - mu) / sigma
    '''
    mu_dir = data_dir + "normalize_mu.npy"
    sigma_dir = data_dir + "normalize_sigma.npy"
    np.save(mu_dir, mu)
    np.save(sigma_dir, sigma)
    '''
    return x_scaled, mu, sigma


def normalize_df(df, data_dir):
    column_names = df.columns
    x = df.values
    x_scaled, mu, sigma = normalize(x)
    mu = mu.reshape(1, -1)
    sigma = sigma.reshape(1, -1)

    normalize_mu = pd.DataFrame(data=mu, columns=column_names)
    normalize_sigma = pd.DataFrame(data=sigma, columns=column_names)
    normalize_param = pd.concat([normalize_mu, normalize_sigma])
    normalize_param.to_csv(data_dir + "normalize_param.csv")
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def digitize_columns(data_frame):
    print("[INFO] digitize columns")

    data_frame = data_frame.replace({"work_year": work_year_map, "class": class_map})
    return data_frame


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


def prepare_data(file_path):
    print("[INFO] prepare loan data.")

    df_loan = pd.read_csv(file_path, low_memory=False)
    # print(f"[INFO] loaded loan data with shape:{df_loan.shape} to :{file_path}")
    df_loan = digitize_columns(df_loan)

    df_loan['issue_date'] = pd.to_datetime(df_loan['issue_date'])

    df_loan['issue_date_month'] = df_loan['issue_date'].dt.month

    df_loan['issue_date_dayofweek'] = df_loan['issue_date'].dt.dayofweek

    cols = ['employer_type', 'industry']
    for col in cols:
        lbl = LabelEncoder().fit(df_loan[col])
        df_loan[col] = lbl.transform(df_loan[col])

    df_loan['earlies_credit_mon'] = pd.to_datetime(df_loan['earlies_credit_mon'].map(findDig))

    df_loan['earliesCreditMon'] = df_loan['earlies_credit_mon'].dt.month
    df_loan['earliesCreditYear'] = df_loan['earlies_credit_mon'].dt.year

    df_loan.fillna(method='bfill', inplace=True)

    col_to_drop = ['issue_date', 'earlies_credit_mon']
    df_loan = df_loan.drop(col_to_drop, axis=1)

    return df_loan


def process_data(loan_df, data_dir, normalize=True):
    loan_feat_df = loan_df[all_feature_list]

    if normalize:
        loan_feat_df = normalize_df(loan_feat_df, data_dir)

    loan_target_df = loan_df[['isDefault']]
    loan_target = loan_target_df.values
    reindex_loan_target_df = pd.DataFrame(loan_target, columns=loan_target_df.columns)
    processed_loan_df = pd.concat([loan_feat_df, reindex_loan_target_df], axis=1)
    # processed_loan_df = pd.concat([loan_feat_df, loan_target_df], axis=1)
    return processed_loan_df


def load_processed_data(data_dir, normalize=True):
    if normalize:
        file_path = data_dir + "processed_loan.csv"
    else:
        file_path = data_dir + "processed_loan_not_normalize.csv"
    if os.path.exists(file_path):
        print(f"[INFO] load processed loan data from {file_path}")
        processed_loan_df = pd.read_csv(file_path, low_memory=False)
    else:
        # print(f"[INFO] start processing loan data.")

        file_path = data_dir + "train_public.csv"
        processed_loan_df = process_data(prepare_data(file_path), data_dir, normalize)
        # processed_loan_df = process_unnormalized_data(prepare_data(file_path))
        if normalize:
            file_path = data_dir + "processed_loan.csv"
        else:
            file_path = data_dir + "processed_loan_not_normalize.csv"
        processed_loan_df.to_csv(file_path, index=False)
        print(f"[INFO] save processed loan data to: {file_path}")
    return processed_loan_df


def preprocess_zhongyuan(data_dir, normalize=True):
    # 输入：数据存放的文件夹
    # 输出：VFL的训练集和测试集，数据格式为numpy
    print("[INFO] load two party data")
    # 从指定的数据集中加载文件
    processed_loan_df = load_processed_data(data_dir,normalize)
    # TODO:为了测试暂时去除随机性,算了随机的地方太多了，去除不了
    processed_loan_df = shuffle(processed_loan_df)

    party_a_feat_list = qualification_feat + loan_feat
    party_b_feat_list = debt_feat + repayment_feat

    X, Y = processed_loan_df[all_feature_list], processed_loan_df['isDefault']
    smo = SMOTE(sampling_strategy=0.25, random_state=42)
    X_smo, Y_smo = smo.fit_resample(X, Y)

    Xa, Xb, y = X_smo[party_a_feat_list].values, X_smo[party_b_feat_list].values, Y_smo.values

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


if __name__ == '__main__':
    data_dir = "/home/yangjirui/feature-infer-workspace/dataset/zhongyuan/"
    preprocess_zhongyuan(data_dir,normalize=True)
