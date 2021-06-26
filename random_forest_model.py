import time

import joblib
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

#load data and preprocess data
train_feature = np.load("D:/BaiduNetdiskDownload/gjbc/feature6/train/trainfeature.npy")
val_feature = np.load("D:/BaiduNetdiskDownload/gjbc/feature6/val/valfeature.npy")
train_label = np.load("D:/BaiduNetdiskDownload/gjbc/train/10type_sort_train_label_8192.npy")
#train_label = np.load("D:/BaiduNetdiskDownload/gjbc/feature0/train_label/trainlabel.npy")
val_label = np.load("D:/BaiduNetdiskDownload/gjbc/val/10type_sort_eval_label_8192.npy")
test_feature = np.load("D:/BaiduNetdiskDownload/gjbc/feature6/test/testfeature.npy")

#feature selection
#feature_imp = np.load('D:/BaiduNetdiskDownload/gjbc/feature6/feature_imp.npy')
#ave = sum(feature_imp) / len(feature_imp)
#inds = np.where(feature_imp > ave)
#train_feature = train_feature[:][inds]
#val_feature = val_feature[:][inds]
#test_feature = test_feature[:][inds]

train_feature = torch.Tensor(train_feature)
val_feature = torch.Tensor(val_feature)
test_feature = torch.Tensor(test_feature)
#shuffle
index1 = [i for i in range(len(train_feature))]
#index2 = [i for i in range(len(val_feature))]
np.random.shuffle(index1)
#np.random.shuffle(index2)
train_feature = train_feature[index1]
train_label = train_label[index1]
#val_feature = val_feature[index2]
#val_label = val_label[index2]

#data cleaning
inds1 = np.where(np.isnan(train_feature))
inds2 = np.where(np.isnan(val_feature))
inds3 = np.where(np.isnan(test_feature))
train_feature[inds1] = 0
val_feature[inds2] = 0
test_feature[inds3] = 0
inds1 = np.where(np.isinf(train_feature))
inds2 = np.where(np.isinf(val_feature))
inds3 = np.where(np.isinf(test_feature))
train_feature[inds1] = 0
val_feature[inds2] = 0
test_feature[inds3] = 0
print('cleaning finished')

# randromforest train and predict
print("random forest")
rf = RandomForestClassifier(random_state=0,n_jobs=4,oob_score='True',n_estimators=200,max_depth=30,
                            class_weight='balanced_subsample',min_samples_leaf=12,min_samples_split=6,
                            )

# train
rf.fit(train_feature, train_label)
#file1 = r'D:/BaiduNetdiskDownload/gjbc/randomforest.joblib 191959'
#rf = joblib.load(file1)
print(rf.score(train_feature,train_label))
print(rf.oob_score_)
print(rf.score(val_feature,val_label))
#print(rf.feature_importances_)
print(len(rf.feature_importances_))
file0 = r'D:/BaiduNetdiskDownload/gjbc/feature6/feature_imp.npy'
#np.save(file0,rf.feature_importances_)

mdhms = time.strftime('%d%H%M', time.localtime(time.time()))
file1 = r'D:/BaiduNetdiskDownload/gjbc/feature6/randomforest.joblib'+' '+mdhms
joblib.dump(rf,file1)

# 使用训练的模型来预测验证集数据
val_pred = rf.predict(val_feature)
val_pred = np.array(val_pred)
print(val_feature.shape)
print(val_label.shape)
print(val_pred.shape)
print(metrics.classification_report(val_label, val_pred))
print(metrics.confusion_matrix(val_label, val_pred))
print("Accuracy:", metrics.accuracy_score(val_label, val_pred))

#save
test_pred = rf.predict(test_feature)
test_pred = np.array(test_pred)
index1 = [i for i in range(len(test_pred))]
np.set_printoptions(suppress=True)
t = np.zeros(shape=(len(index1),2))
t[:,0] = index1
t[:,1] = test_pred
file2 = r'D:/BaiduNetdiskDownload/gjbc/feature6/pre_label.csv'
np.savetxt(file2, t,delimiter=',',fmt='%s')

