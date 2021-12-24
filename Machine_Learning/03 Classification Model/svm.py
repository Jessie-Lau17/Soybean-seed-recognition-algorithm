
"""
@author:Lisa
@file:svm_Iris.py
@func:Use SVM to achieve Iris flower classification
@time:2018/5/30 0030上午 9:58
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  soyben_name=['AN5','DN59','DND252','DND254','HJ12','JC168','JN417','LD130','XX6']
#define converts(字典)
def Iris_label(s):
    it={b'AN5':0, b'DN59':1, b'DND252':2 , b'DND254':3, b'HJ12':4,b'JC168':5,b'JN417':6,b'LD130':7,b'XX6':8 }
    return it[s]
 
 
#1.读取数据集
path='data_1.txt'
data=np.loadtxt(path, dtype=float, delimiter='\t', converters={7:Iris_label} )
#converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
#print(data.shape)
#1-2.制作单个测试集
# path2='DND254_data.txt'
# data_2=np.loadtxt(path2, dtype=float, delimiter='\t', converters={7:Iris_label} )
# x_test,y_test=np.split(data_2,indices_or_sections=(7,),axis=1) #x为数据，y为标签
# x_test=x_test[:,0:7]
# #对单个测试集也做数据标准化
# std_test=StandardScaler()
# x_test = std_test.fit_transform(x_test)
# print(x_test[0:4,:])
# print(y_test[0:4,:])
#print(y_test)

#2.划分数据与标签
x,y=np.split(data,indices_or_sections=(7,),axis=1) #x为数据，y为标签
x=x[:,0:7]
#我们进行标准化看看效果
std = StandardScaler()
x = std.fit_transform(x)
print('SVM: 数据标准化：')
#提取单个测试集数据
x_test=x[840:1260,0:7]
y_test=y[840:1260,:]
train_data,test_data,train_label,test_label =train_test_split(x,y, random_state=1, train_size=0.7,test_size=0.3) #sklearn.model_selection.
#print(train_label.shape)
x_3d = train_data[:, 0]
y_3d = train_data[:, 2]
z_3d = train_data[:, 3] 
#3.训练svm分类器
classifier=svm.SVC(C=1,kernel='rbf',gamma=10,decision_function_shape='ovo') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
 
#4.计算svc分类器的准确率
print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))
print("单个DND252测试集：",classifier.score(x_test,y_test))
#也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
# print(tes_label[0:20])
# print(train_label[0:20])
# print('here')
# print("训练集：", accuracy_score(train_label,tra_label) )
# print("测试集：", accuracy_score(test_label,tes_label) )
