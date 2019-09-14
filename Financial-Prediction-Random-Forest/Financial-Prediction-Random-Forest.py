#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import talib
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def get_label(data,days):
    label = [''] * len(data['close'])
    for i in range(len(data['close']) - days):
        if (data['close'][i + days] - data['close'][i]) > 0:
            label[i] = 1
        else:
            label[i] = -1
    # Save to typefile file
    data['label'] = label
    return data


def exponential_smoothing(alpha,stock_price):
    es = np.zeros(stock_price.shape)
    es[0] = stock_price[0]
    for i in range(1, len(es)):
        es[i] = alpha*float(stock_price[i])+(1-alpha)*float(es[i-1])
    return es

# print(exponential_smoothing(0.1,np.array(df['open'])))

# preprocess the stock data with exponential_smoothing
def em_stock_data(df,alpha):
    es_open = pd.DataFrame(exponential_smoothing(alpha,np.array(df['open'])))
    es_close = pd.DataFrame(exponential_smoothing(alpha, np.array(df['close'])))
    es_high = pd.DataFrame(exponential_smoothing(alpha, np.array(df['high'])))
    es_low = pd.DataFrame(exponential_smoothing(alpha, np.array(df['low'])))
    df['open'],df['close'],df['high'],df['low'] = es_open,es_close,es_high,es_low
    return df

# print(em_stock_data(df,0.1))

def cal_technical_indicators(df):
    # Simple Moving Average SMA 简单移动平均
    df['SMA5'] = talib.MA(df['close'], timeperiod=5)
    df['SMA10'] = talib.MA(df['close'], timeperiod=10)
    df['SMA20'] = talib.MA(df['close'], timeperiod=20)
    # Williams Overbought/Oversold Index WR 威廉指标
    df['WR14'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['WR18'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=18)
    df['WR22'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=22)
    # Moving Average Convergence / Divergence MACD 指数平滑移动平均线
    DIFF1, DEA1, df['MACD9'] = talib.MACD(np.array(df['close']), fastperiod=12, slowperiod=26, signalperiod=9)
    DIFF2, DEA2, df['MACD10'] = talib.MACD(np.array(df['close']), fastperiod=14, slowperiod=28, signalperiod=10)
    df['MACD9'] = df['MACD9'] * 2
    df['MACD10'] = df['MACD10'] * 2
    # Relative Strength Index RSI 相对强弱指数
    df['RSI15'] = talib.RSI(np.array(df['close']), timeperiod=15)
    df['RSI20'] = talib.RSI(np.array(df['close']), timeperiod=20)
    df['RSI25'] = talib.RSI(np.array(df['close']), timeperiod=25)
    df['RSI30'] = talib.RSI(np.array(df['close']), timeperiod=30)
    # Stochastic Oscillator Slow STOCH 常用的KDJ指标中的KD指标
    df['STOCH'] = \
    talib.STOCH(df['high'], df['low'], df['close'], fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3,
                slowd_matype=0)[1]
    # On Balance Volume OBV 能量潮
    df['OBV'] = talib.OBV(np.array(df['close']), df['vol'])
    # Simple moving average SMA 简单移动平均
    df['SMA15'] = talib.SMA(df['close'], timeperiod=15)
    df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA25'] = talib.SMA(df['close'], timeperiod=25)
    df['SMA30'] = talib.SMA(df['close'], timeperiod=30)
    # Money Flow Index MFI MFI指标
    df['MFI14'] = talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=14)
    df['MFI18'] = talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=18)
    df['MFI22'] = talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=22)
    # Ultimate Oscillator UO 终极指标
    df['UO7'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['UO8'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=8, timeperiod2=16, timeperiod3=22)
    df['UO9'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=9, timeperiod2=18, timeperiod3=26)
    # Rate of change Percentage ROCP 价格变化率
    df['ROCP'] = talib.ROCP(df['close'], timeperiod=10)
    return df

# print(cal_technical_indicators(df))
# print(cal_technical_indicators(df).shape)
# print(cal_technical_indicators(df).columns)

# drop_features = ['Unnamed: 0', 'ts_code', 'trade_date','pre_close', 'change', 'pct_chg','amount']
# # df = df.drop(drop_features,axis=1)
# # print(df[df.columns])
# # features = list(df.T.index)
# # print(features)

def normalization(df,pred_days):
    df = df[36:(len(df['vol']) - pred_days)]
    features = list(df.T.index)
    features.remove('label')
    # normalization
    min_max_scaler = MinMaxScaler()
    df[features] = min_max_scaler.fit_transform(df[features])
    return df,features

def split_data(df,features):
    # split data set
    df_len = len(df)
    train_data = df[:int(df_len * 0.8)]
    valid_data = df[int(df_len * 0.8):int(df_len * 0.9)]
    test_data = df[int(df_len * 0.9):]

    train_X = train_data[features].values
    train_y = train_data['label'].values

    valid_X = valid_data[features].values
    valid_y = valid_data['label'].values

    test_X = test_data[features].values
    test_y = test_data['label'].values

    return train_X,train_y,valid_X,valid_y,test_X,test_y

def model(train_X,train_y,valid_X,valid_y,test_X,test_y):
    rfg = RandomForestClassifier()
    rfg.fit(train_X,train_y)

    train_acc = rfg.score(train_X,train_y)
    valid_acc = rfg.score(valid_X,valid_y)

    predict = rfg.predict(test_X)
    features_important = sorted(zip(map(lambda x: round(x, 4), rfg.feature_importances_),df[features]), reverse=True)
    pred_accuracy = (test_y == predict).mean()

    return train_acc,valid_acc,features_important,pred_accuracy

if __name__ == '__main__':
    df = pd.read_csv('002202.SZ_stock.csv')
    df = df.sort_values('trade_date')
    # print(df.head())

    drop_features = ['Unnamed: 0', 'ts_code', 'trade_date','pre_close', 'change', 'pct_chg','amount']
    df = df.drop(drop_features,axis=1)
    df.index = range(len(df))
    #print(df)

    df = get_label(df,5)
    em_stock = em_stock_data(df,0.1)
    cal_stock = cal_technical_indicators(em_stock)
    #print(cal_stock)

    normalization_stock,features = normalization(cal_stock,5)
    #print(normalization_stock.columns)

    train_X,train_y,valid_X,valid_y,test_X,test_y = split_data(normalization_stock,features)

    train_y = train_y.astype('int')
    valid_y = valid_y.astype('int')
    test_y = test_y.astype('int')

    print(model(train_X,train_y,valid_X,valid_y,test_X,test_y))

