**基于LSTM预测股票价格**

**数据集**：
+ 000858.SZ 五粮液

**数据特征:**
+ 只选用原始数据特征（开盘价、收盘价、最高价、最低价、交易量）

**时间窗口**：
+ 15天

**流程：**
+ 读取数据->生成标签(下一天收盘价)->分割数据集->LSTM模型预测->可视化->预测结果评估


**函数介绍：**
+ 1、get_label 生成标签（下一天收盘价）
+ 2、normalized 归一化
+ 3、get_model_data 分割数据集
+ 4、evaluate 结果评估
+ 5、lstm_model LSTM预测模型


**训练集拟合效果：**
+ <img src="https://github.com/jiwawa112/Financial-Prediction/raw/master/Financial-Prediction-LSTM/images/train_pred.png" width="500">

**测试集拟合效果：**
+ <img src="https://github.com/jiwawa112/Financial-Prediction/raw/master/Financial-Prediction-LSTM/images/test_pred.png" width="500">

**评估指标：**
+ 1、RMSE
+ 2、MAE
