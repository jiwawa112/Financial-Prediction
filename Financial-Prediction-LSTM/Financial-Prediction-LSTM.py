#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

# 生成标签值：下一天收盘价
def get_label(df):
    next_close = list()
    for i in range(len(df['close']) - 1):
        next_close.append(df['close'][i + 1])
    next_close.append(0)
    df['next_close'] = next_close
    df.drop(df.index[-1], inplace=True)
    return df

# 归一化处理
def normalized(df):
    for i in ['open', 'close', 'high', 'low', 'vol','next_close']:
        df[i] = scaler.fit_transform(np.reshape(np.array(df[i]), (-1, 1)))
    return df

# 生成训练和测试数据
def generate_model_data(df,alpha,days):
    data_length = int((len(df['close']) - days + 1))
    train_data,train_label = list(),list()
    # 生成时序数据
    for i in range(data_length):
        train_label.append(df['next_close'][i + days - 1])
        for j in range(days):
            for m in ['open','close','high','low','vol']:
                train_data.append(df[m][i + j])

    train_data = np.reshape(np.array(train_data),(-1,5 * days))  # 5表示特征数量*天数
    train_length = int(len(train_label) * alpha)

    train_X = np.reshape(np.array(train_data[:train_length]),(len(train_data[:train_length]),days,5))
    train_y = np.array(train_label[:train_length])

    test_X = np.reshape(np.array(train_data[train_length:]),(len(train_data[train_length:]),days,5))
    test_y = np.array(train_label[train_length:])

    return train_X,train_y,test_X,test_y

def evaluate(real,pred):
    RMSE = math.sqrt(mean_squared_error(real[:, 0], pred[:, 0]))
    MAE = mean_absolute_error(real[:, 0], pred[:, 0])
    return RMSE,MAE

def lstm_model(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(LSTM(units=20, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_y, epochs=100, batch_size=32, verbose=2)

    train_pred = model.predict(train_X)
    pred = scaler.inverse_transform(train_pred)
    y_train = scaler.inverse_transform(np.reshape(train_y, (-1, 1)))

    test_pred = model.predict(test_X)
    test_pred = scaler.inverse_transform(test_pred)
    y_test = scaler.inverse_transform(np.reshape(test_y, (-1, 1)))

    return y_train, train_pred, y_test, test_pred

if __name__ == '__main__':
    df = pd.read_csv('000858.SZ_stock.csv')
    drop_features = ['Unnamed: 0','ts_code','trade_date','pre_close','change','pct_chg','amount']
    df = df.drop(drop_features,axis=1)
    df.index = range(len(df))
    df.head()

    days = 15
    alpha = 0.8
    df = get_label(df)
    df.head()
    scaler = MinMaxScaler(feature_range=(0,1))
    df = normalized(df)
    train_X,train_y,test_X,test_y = generate_model_data(df,alpha,days)
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)
    train_y,train_pred,test_y,test_pred = lstm_model(train_X,train_y,test_X,test_y)

    plt.plot(list(train_pred),color='red',label='predict')
    plt.plot(list(train_y),color='blue',label='real')
    plt.legend(loc='upper left')
    plt.title('train data')
    plt.show()

    plt.plot(list(test_pred),color='red',label='prediction')
    plt.plot(list(test_y),color='blue',label='real')
    plt.legend(loc='upper left')
    plt.title('test data')
    plt.show()

    RMSE,MAE = evaluate(test_y,test_pred)
    print(RMSE)
    print(MAE)