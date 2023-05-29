import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
from utils import TransferLabel
import copy
import warnings

warnings.filterwarnings('ignore')

""" 小波阈值去噪 """


# sgn函数
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def sig_wavelet_noising(new_df):
    data = new_df
    data = data.values.T.tolist()
    w = pywt.Wavelet('dB10')  # 选择dB10小波基
    ca3, cd3, cd2, cd1 = pywt.wavedec(data, w, level=3)  # 3层小波分解
    ca3 = ca3.squeeze()  # ndarray数组减维：(1，a)->(a,)
    cd3 = cd3.squeeze()
    cd2 = cd2.squeeze()
    cd1 = cd1.squeeze()
    length1 = len(cd1)
    length0 = len(data)

    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = [ca3]

    # 软阈值方法
    for k in range(length1):
        if abs(cd1[k]) >= lamda / np.log2(2):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda / np.log2(2))
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if abs(cd2[k]) >= lamda / np.log2(3):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda / np.log2(3))
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if abs(cd3[k]) >= lamda / np.log2(4):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda / np.log2(4))
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构

    return np.round(recoeffs, 3)


def wavelet_noising(data):
    newdata = copy.deepcopy(data)
    column = newdata.columns
    m = len(newdata)
    for col in column:
        data_denoising = sig_wavelet_noising(newdata[col])  # 调用小波阈值方法去噪
        n = len(data_denoising)
        if m < n:
            data_denoising = data_denoising[:m]

        newdata[col] = data_denoising

    # newdata.to_csv('Denoised/' + name + '.csv', index=False, encoding='gbk')

    return newdata


if __name__ == '__main__':

    name = 'W1'

    p = 'datasets/'
    data = pd.read_csv(p + name + '.csv', index_col=False, encoding='gbk')

    column = list(data.columns)

    data = pd.DataFrame(data, columns=column)
    c = column[1: len(column) - 1]
    depths = pd.DataFrame(data['Depths'], columns=['Depths'])
    liths = pd.DataFrame(data['Liths'], columns=['Liths'])
    label, liths_map = TransferLabel(liths)
    x = data['Depths']

    data = data.iloc[:, 1:-1]

    data_denoised = wavelet_noising(data)
    for col in c:
        plt.figure(figsize=(16, 8))

        n = 800
        plt.plot(x.iloc[:n], data[col].iloc[:n], 'r')
        plt.plot(x.iloc[:n], data_denoised[col].iloc[:n], 'b--', )
        # plt.scatter(x.iloc[500:2500], data[col].iloc[:n], s=2, color='r')

        plt.legend(['Original ' + col, 'Denoisied ' + col])
        plt.show()
