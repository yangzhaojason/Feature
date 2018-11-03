# -*-coding:utf-8-*-
# 数据集使用KDD99，我们从所有数据集中的正常数据集
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy import *
import numpy as np
import csv
import os
import math
import xgboost as xgb
from xgboost import XGBClassifier

# 数据预处理
path_KDD_test = '/home/apple/Documents/MacToLinux/KDDcup_study.csv'

# 算法部分
params = {
    'max_depth': 10,
    'subsample': 1,
    'verbose_eval': True,
    'seed': 12,
    'objective':'binary:logistic'
}

def zeroMean(dataMat):
    meanVal = np.mean(dataMat,axis=0)
    newData = dataMat-meanVal
    return newData,meanVal

def max_min_normalization(arr):
    arr = arr.tolist()
    list = []
    for x in arr:
        x = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
        list.append(x)
    return list

def fileOpenWrite(stu, path):
    with open(path, mode='a', encoding='utf-8', newline='') as out:
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(stu)

def Processing_data():
    print("Processing data")
    df = pd.read_csv(path)
    Y_data = df.iloc[:, 41].values
    Dict = {}
    for inlist in Y_data:
        # print(inlist)
        if Dict.get(inlist) == None:
            Dict[inlist] = 1
        else:
            Dict[inlist] += 1
    # {'normal.': 595797, 'buffer_overflow.': 5, 'loadmodule.': 2, 'perl.': 2, 'neptune.': 204815, 'smurf.': 227524, 'guess_passwd.': 53, 'pod.': 40, 'teardrop.': 199, 'portsweep.': 2782, 'ipsweep.': 7579, 'land.': 17, 'ftp_write.': 8, 'back.': 2002, 'imap.': 12, 'satan.': 5393, 'phf.': 3, 'nmap.': 2316, 'multihop.': 6,'warezmaster.': 20}
    Dict_abnormal = []
    csv_file = csv.reader(open(path, 'r'))
    for i, stu in enumerate(csv_file):
        if stu[41] == 'normal.':
            # out = open(path_normal, mode='a', newline='', encoding='utf-8')
            # with open(path1, mode='a', encoding='utf-8', newline='') as out:
            #     csv_write = csv.writer(out, dialect='excel')
            #     csv_write.writerow(stu)
            fileOpenWrite(stu, path_normal)
        # else:
        #     if stu[41] in Dict_abnormal:
        #         csvname = stu[41] + ".csv"
        #         path_abnormal = os.path.join(base, csvname)
        #         # with open(path_abnormal, mode='a', encoding='utf-8', newline='') as out:
        #         #     csv_write = csv.writer(out, dialect='excel')
        #         #     csv_write.writerow(stu)
        #         fileOpenWrite(stu, path_abnormal)
        #     else:
        #         Dict_abnormal.append(stu[41])
        #         csvname = stu[41] + ".csv"
        #         path_abnormal = os.path.join(base, csvname)
        #         # with open(path_abnormal, mode='a', encoding='utf-8', newline='') as out:
        #         #     csv_write = csv.writer(out, dialect='excel')
        #         #     csv_write.writerow(stu)
        #         fileOpenWrite(stu, path_abnormal)

def calc(data):
    n=len(data) # 10000个数
    niu=0.0 # niu表示平均值,即期望.
    niu2=0.0 # niu2表示平方的平均值
    niu3=0.0 # niu3表示三次方的平均值
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [niu,sigma,niu3]

def calc_stat(data):
    [niu, sigma, niu3]=calc(data)
    n=len(data)
    niu4=0.0 # niu4计算峰度计算公式的分子
    for a in data:
        a -= niu
        niu4 += a**4
    niu4 /= n
    skew =(niu3 -3*niu*sigma**2-niu**3)/(sigma**3) # 偏度计算公式
    kurt=niu4/(sigma**4) # 峰度计算公式:下方为方差的平方即为标准差的四次方
    return [niu, sigma,skew,kurt]

def test_pca(data, N):
    print("PCA Processing")
    newData, meanVal = zeroMean(data)
    covMat = np.cov(newData, rowvar=0)
    Diag = np.diag(covMat)
    tot = sum(Diag)
    var_exp = [(Diag[i]) / tot for i in range(len(Diag))]
    eigValIndice = np.argsort(Diag)
    n = int(round(len(Diag)*N))
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    New_List = []
    for i in range(n):
        New_List.append((n_eigValIndice[i]+1, var_exp[n_eigValIndice[i]]))
    return New_List






def test_Xgboost(x, y1):
    print("Xgboost processing")
    y = max_min_normalization(y1)
    xgtrain = xgb.DMatrix(x, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=10)
    fmap = 'weight'
    importance = bst.get_score(fmap='', importance_type=fmap)
    print(fmap,importance)
    print('\n')
    # print(bst.get_dump(with_stats=False))
    fmap = 'gain'
    importance = bst.get_score(fmap='', importance_type=fmap)
    print(fmap,importance)
    print('\n')
    # print(bst.get_dump(with_stats=True))
    fmap = 'cover'
    importance = bst.get_score(fmap='', importance_type=fmap)
    print(fmap,importance)
    print('\n')
    # print(bst.get_dump(with_stats=True))

def XGB(x,y,N):
    print("XGBoost importance of feature score")
    model = XGBClassifier()
    model.fit(x, y)
    # feature importance
    featureImportant = model.feature_importances_
    # print(featureImportant)
    list_feature = featureImportant.tolist()
    xgb_Dictionary = {}
    for i in range(len(featureImportant)):
        xgb_Dictionary[i+1] = list_feature[i]
    # print(xgb_Dictionary)
    Tuple_xgbSorted = sorted(xgb_Dictionary.items(), key = lambda item:item[1], reverse = True)
    length_xgbSelected = int(round(len(featureImportant) * N))
    Result_xgbSelected = []
    for j in range(length_xgbSelected):
        Result_xgbSelected.append(Tuple_xgbSorted[j])
    return Result_xgbSelected
    # print(type(featureImportant)) #<type 'numpy.ndarray'>
    # print(len(featureImportant))
    

#峰度：概率密度在均值处峰值高低的特征，假设满足高斯分布
def kurtosis(data):
    print("kurtosis processing")
    dic_kurtosis = {}
    for i in range(40):
        [niu, sigma, skew, kurt] = calc_stat(data[:, i])
        dic_kurtosis[i+1] = kurt
    return dic_kurtosis

if __name__ == '__main__':
    print("Main processing")
    if os.path.exists(path_KDD_test):
        if os.path.getsize(path_KDD_test):
            df = pd.read_csv(path_KDD_test, header=None)
            # print(df.shape)
            X = df.iloc[0:df.shape[0], 1:42].values  # X是数据
            Y1 = df.iloc[0:df.shape[0], 0].values  # Y是标签
            # print(type(Y1)) #<class 'numpy.ndarray'>
            N = 0.8
            result_xgboost = XGB(X, Y1, N)
            print('XGBoost',result_xgboost)
            # test_Xgboost(X, Y1)
            result_pca = test_pca(X, N)
            print('PCA',result_pca)
            # result_kurtosis = kurtosis(X_data)
            # print(result_kurtosis)
        else:
            print('File is null!')
            Processing_data()
            print('Extract csv file data end!')

    # list = os.listdir(base)  # 列出文件夹下所有的目录与文件
    # for i in range(0, len(list)):
    #     dir_path = os.path.join(base, list[i])
    #     df = pd.read_csv(dir_path)
    #     X_data = df.iloc[:, 5:41].values
    #     test_pca(X_data)

