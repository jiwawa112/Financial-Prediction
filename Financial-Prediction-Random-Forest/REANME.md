
# 基于随机森林预测股票未来第d+k天相比于第d天的涨/跌

**参考论文：Predicting the direction of stock market prices using random forest**

**论文思路：**
![Alt text](https://github.com/jiwawa112/Financial-Prediction/tree/master/Financial-Prediction-Random-Forest/images/Methodology.jpg.png)

**算法流程：**
获取金融数据->指数平滑->计算技术指标->数据归一化->随机森林模型预测

**函数介绍：**
+ 1、get_stock_label 得到未来第d+k天相比于第d天的涨/跌的标签  1表示上涨 -1表示下跌
+ 2、exponential_smoothing 指数平滑公式
+ 3、es_stock_data 股票价格平滑处理
+ 4、cal_technical_indicators 计算常用股票技术指标
+ 5、normalization 数据归一化
+ 6、split_data 划分训练、验证、测试数据集
+ 7、model 随机森林模型并返回准确率和特征排名

**决策树：**
+（1）ID3: 基于信息增益大的特征划分层次
+（2）C4.5: 基于信息增益比=信息增益/特征熵划分层次
+（3）CART: 基于Gini划分层次

**基于Bagging集成学习算法，有多棵决策树组成（通常是CART决策树），其主要特性有：**
+ （1）样本随机采样
+ （2）对异常样本点不敏感
+ （3）适用于数据维度大的数据集
+ （4）可以并行训练（决策树间独立同分布）

**存在问题：**
+ 1.模型未进行参数寻优(树的棵树、树的最大深度、树节点最大特征数、叶子节点最小样本数等)
+ 2.未来第k天的选择问题

**归一化方法**
随机森林模型其实本身不需要数据归一化（如需要对数据集进行归一化也需要考虑对训练集、验证集、测试集独立归一化）

**股票预测考虑的数据特征：**
+ 原始数据特征（open/close/high/low）
+ 技术指标（Technical indicator）
+ 企业财务报表(利润表、资产负债表、现金流)
+ 宏观经济指数（Shibor利率、美元指数等）
+ 政治事件
+ 财经新闻
+ 社会舆论
+ 股民情绪
+ 国家政策
+ 股票间影响等
