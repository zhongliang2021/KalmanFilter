import numpy

import os
import pandas as pd
import numpy as np
import re
def KalmanFilter(z, n_iter=20):
    # 这里是假设A=1，H=1的情况

    # intial parameters

    sz = (n_iter,)  # size of array

    # Q = 1e-5 # process variance
    Q = 1e-6  # process variance
    # allocate space for arrays
    xhat = numpy.zeros(sz)  # a posteri estimate of x
    P = numpy.zeros(sz)  # a posteri error estimate
    xhatminus = numpy.zeros(sz)  # a priori estimate of x
    Pminus = numpy.zeros(sz)  # a priori error estimate
    K = numpy.zeros(sz)  # gain or blending factor

    R = 0.1 ** 2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    A = 1
    H = 1

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat


#设置Excel文件夹的路径
filePath = r"D:\Code\Rawdata"
#获取文件夹下的所有文件名称
nameList = os.listdir(filePath)


for i in nameList:
    #使用pandas中的read_excel函数读取文件 我这里只读取一行数据 nrows=n的含义为读取第n行数据(注意不是读取前n数据)
    temp = pd.read_excel(r"D:\\Code\\Rawdata\\" + i, index_col=0)
    print("表格名：")
    print(i)  # 文件名
    print('表格内容:')
    print(temp)#文件内容
    rawdata = temp
    row, column = rawdata.shape
    print(rawdata.shape)
    data_kalman = np.zeros((row, 16), dtype=np.float)
    rawdata = rawdata.to_numpy()
    raw_data = list()
    for j in range(16):
        for k in range(row):
            raw_data.append(rawdata[k][j])  # 0-15
        xhat = KalmanFilter(raw_data, n_iter=len(raw_data))

        # xhat = xhat.tolist()
        print(xhat)
        print(type(xhat))

        # pylab.plot(raw_data, 'k-', label='raw measurement')  # 测量值
        # pylab.plot(xhat, 'b-', label='Kalman estimate')  # 过滤后的值
        # pylab.legend()
        # pylab.xlabel('Iteration')
        # pylab.ylabel('ADC reading')
        # pylab.show()
        for m in range(row):
            data_kalman[m][j] = xhat[m]
        raw_data = list()
    print(data_kalman)
    print(data_kalman.shape)

    data_kalman = np.delete(data_kalman, 0, axis=0)

    print(data_kalman)
    print(data_kalman.shape)

    data_kalman = pd.DataFrame(data_kalman)
    print(data_kalman)

    data_kalman.to_excel(r"D:\\Code\\KalmanFilter_output\\" + i, sheet_name='sheet1')  # 此处修改EXCEL的名称和SHEET
