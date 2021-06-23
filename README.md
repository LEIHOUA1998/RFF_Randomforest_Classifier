# RFF_Randomforest_Classifier
# RFF分类程序文档说明

本程序旨在对射频信号指纹进行分类，程序主要分为两个部分：特征提取和信号分类。
***feature_extract.py***是对射频信号进行特征提取，***random_forest_model.py***是使用随机森林作为分类器根据提取的特征对信号进行分类。

## 特征提取（***feature_extract.py***）

读入训练集、验证集、测试集的数据，提取信号的短时功率谱作为特征保存为npy文件。
**定义函数**
各函数作用：
***create_folder***：建立空的特征文件夹，用于存放提取后的特征。
***Power_spec***：将一个信号样本分成若干个等长的子块，进行短时分析，对每个子块进行傅里叶变换后，取模求平方得到信号的短时功率谱，取信号的单边功率谱作为特征。
**输入参数**：data1为一个一维的射频信号，size为短时分析的点数，即是子块的长度（也是fft的点数），这里的短时分析没有加窗也没有重叠部分。程序中将size设置为512点，共有16个子块，每个子块取单边功率谱共257点，即特征点数为4112.
其余定义函数均为求取信号的特征，但在后续没有使用（效果不好），故略。

**特征提取**
定义三个空矩阵用于存放提取特征，读取每个样本，调用***Power_spec***函数进行特征提取，循环结束后储存到用***create_folder***函数创建的文件夹中。

## 随机森林分类（***random_forest_model***）

读入和预处理数据：先将特征文件和标签文件读入,把特征和标签都张量化，然后通过shuffle把训练数据进行打乱（对于随机森林这一步似乎可以省略），最后检查输入特征是否有nan值，若有则进行0值替换。

模型训练：调用***RandomForestClassifier***函数定义随机森林模型，根据情况设置各参数，输入训练集特征和标签进行训练，将训练好的模型储存。随机森林可以给出输入特征的权重，将其保存，用于特征筛选。

预测：输入验证集的特征和标签观察预测准确率，并将测试集的特征输入得到预测标签存储为相应格式的csv文件。

## 特征筛选

在**特征提取**部分读取特征权重为***feature_imp***，计算权重平均值，提取权重高于平均权重的特征作为新的特征，可以降低特征维度和提高分类准确率。本程序一般特征筛选一轮即可达到最优。

```

```
