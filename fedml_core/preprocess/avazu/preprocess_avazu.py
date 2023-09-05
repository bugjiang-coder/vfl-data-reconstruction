import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle


'''
注意数据处理的时候一定不要正则化，规范化，会加大数据的恢复难度
'''

def preprocess_avazu(data_dir, frac=0.04, using_processed=True):
    # 训练集和测试集文件位置
    # train_data_file_route = data_dir + 'train_0.9_smote_1_norm.csv'
    # test_data_file_route = data_dir + 'test_0.1_smote_1_norm.csv'
    train_data_file_route = data_dir + 'train_0.9_vfl.csv'
    test_data_file_route = data_dir + 'test_0.1_vfl.csv'


    # 一共有30个被挑选出来的特征
    partA = ['app_category_0', 'site_category_20', 'site_category_3', 'site_id_users', 'C21', 'site_category_11', 'C_site_domain', 'C_app_id', 'banner_pos', 'site_category_22', 'device_conn_type', 'app_domain', 'app_category_19', 'site_category_2', 'app_category_26']
    partB = ['C_pix', 'app_id_users', 'site_category_1', 'site_category_23', 'site_category_5', 'C18', 'C_site_id', 'C_device_type_1', 'is_device', 'app_category_24', 'site_category_0', 'C19', 'C20', 'app_category_4', 'C14']

    all_columns = partA + partB
    x_columns_selected_y = all_columns + ['click']


    # 如果已经有划分好的数据, 直接读取数据
    if using_processed and os.path.exists(train_data_file_route) and os.path.exists(test_data_file_route):
        train_set = pd.read_csv(train_data_file_route)
        test_set = pd.read_csv(test_data_file_route)

    else:
        train_origin_df = pd.read_csv(data_dir + 'tr_FE.csv')
        X_y = train_origin_df[x_columns_selected_y]

        # 根据 click 列分组，对每组数据进行抽样，抽出相同的数据量
        nums = min(X_y['click'].sum(), (1 - X_y['click']).sum())
        samples = X_y.groupby('click').apply(lambda x: x.sample(n=nums))
        # 合并所有抽样结果
        X_y = pd.concat([samples.loc[samples['click'] == 0], samples.loc[samples['click'] == 1]])

        # 对所有数据进行洗牌
        X_y = shuffle(X_y)

        # 注意总的数据量大小为2,021,448
        # 训练过于费时，这里直接选择其中20,000左右比较好
        X_y = X_y.sample(frac=frac, random_state=24)

        # 得到训练集和测试集的dataframe：9：1
        train_set, test_set = train_test_split(X_y, test_size=0.1, random_state=42)

        # 储存训练集和测试集（in csv）
        train_set.to_csv(path_or_buf=os.path.join(train_data_file_route), index=False)
        test_set.to_csv(path_or_buf=os.path.join(test_data_file_route), index=False)

    # 训练集和测试集 特征标签划分
    Xa_train, Xb_train = train_set[partA].values, train_set[partB].values

    Xa_test, Xb_test = test_set[partA].values, test_set[partB].values

    y_train, y_test = train_set['click'].to_frame().values, test_set['click'].to_frame().values

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))

    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]


if __name__ == "__main__":
    # data_dir = '/home/yangjirui/kaggle-dataset/avazu/-Kaggle-Click-Through-Rate-Prediction-/result/'
    data_dir = '/home/yangjirui/feature-infer-workspace/dataset/avazu/'
    processed_dir = '/home/yangjirui/feature-infer-workspace/dataset/avazu/'
    preprocess_avazu(data_dir, frac=0.04)