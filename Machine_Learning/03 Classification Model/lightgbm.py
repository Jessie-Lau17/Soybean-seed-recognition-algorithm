import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Variety_Path = 'dataset/data_ori/soybean/Varieties.txt'
DataSet_Path = 'dataset/data_feature/soybean/'
Feature_Num = 263

print('Loading the DataSet...')

f = open(Variety_Path, 'r')
varieties = f.read().splitlines()
f.close()

print('The number of varieties is %d. ' % len(varieties))
num = 1

data = np.zeros([1, Feature_Num + 1])
for var in varieties:
    open_path = DataSet_Path + var + '.csv'
    data_temp = np.loadtxt(open_path, delimiter=',')
    data_temp = np.append(data_temp, np.ones([data_temp.shape[0], 1]) * (num - 1), axis=1)
    data = np.append(data, data_temp, axis=0)
    num += 1
data = np.delete(data, 0, axis=0)
label = np.int_(data[:, Feature_Num])
data = np.delete(data, Feature_Num, axis=1)

# data = LinearDiscriminantAnalysis(n_components=3).fit_transform(data, label)
# X = np.delete(data, [0, 1, 2, 3, 4, 5, 6], axis=1)
# Y = label
# X = data[:, 6:7]
# Y = label
X = data
Y = label
X = data[label == 0]
Y = label[label == 0]
X = np.append(X, data[label == 1], axis=0)
Y = np.append(Y, label[label == 1], axis=0)
X = np.append(X, data[label == 2], axis=0)
Y = np.append(Y, label[label == 2], axis=0)
X = np.append(X, data[label == 6], axis=0)
Y = np.append(Y, label[label == 6]-3, axis=0)

print('Successfully Completed! \n')

print('Splitting the DataSet...')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
print('Successfully Completed! \n')

for i in range(0, len(varieties)):
    print('Class %d has %d TrainData, %d TestData' % (i, np.sum(Y_train == i), np.sum(Y_test == i)))
print('\n')

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'multiclass',  # 目标函数
    'num_class': 4,  # 类别个数len(varieties)
    'metric': {'multi_logloss'},  # 评估函数, 'multi_error'
    'num_leaves': 80,  # 叶子节点数
    'max_depth': 7,  # 最大深度
    # 'min_sum_hessian_in_leaf': 0.003,   # 防止过拟合的参数：
    'min_data_in_leaf': 100,            # 防止过拟合的参数：叶子中最小样本数
    'learning_rate': 0.01,  # 学习速率
    'feature_fraction': 0.8,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Training...')
classifier = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_eval], early_stopping_rounds=10)
print('Successfully Completed! \n')

print('Saving the Model...')
classifier.save_model('lgb_classifier.model')
print('Successfully Completed! \n')

print('Predicting the TestData...')
Y_clf_weight = classifier.predict(X_test, num_iteration=classifier.best_iteration)
Y_pred = Y_clf_weight.argmax(axis=1)
Y_clf_weight = classifier.predict(X_train, num_iteration=classifier.best_iteration)
Y_pred_train = Y_clf_weight.argmax(axis=1)
print('Successfully Completed! \n')

print('Computing the Accuracy...')
test_acc = accuracy_score(Y_test, Y_pred)
train_acc = accuracy_score(Y_train, Y_pred_train)
print('TestSet Accuracy: %f' % test_acc)
print('TrainSet Accuracy: %f' % train_acc)

print('混淆矩阵：')
print(confusion_matrix(Y_test, Y_pred, labels=range(4)))
