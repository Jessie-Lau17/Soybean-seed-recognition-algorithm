import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Variety_Path = 'dataset/data_ori/soybean/Varieties.txt'     # 分析的大豆种子种类名称
DataSet_Path = 'dataset/data_feature/soybean/'              # 数据集路径
Feature_Num = 263   # 分析的特征个数
n_components = 3    # LDA算法中所要保留下来的特征个数n

f = open(Variety_Path, 'r')
varieties = f.read().splitlines()
f.close()

print('The number of varieties is %d. ' % len(varieties))
num = 1

data = np.zeros([1, Feature_Num+1])
for var in varieties:
    open_path = DataSet_Path + var + '.csv'
    data_temp = np.loadtxt(open_path, delimiter=',')
    data_temp = np.append(data_temp, np.ones([data_temp.shape[0], 1]) * (num-1), axis=1)
    data = np.append(data, data_temp, axis=0)
    num += 1
data = np.delete(data, 0, axis=0)
label = np.int_(data[:, Feature_Num])
data = np.delete(data, Feature_Num, axis=1)
# data = data[:, 0:7]

t0 = time()
data_lda = LinearDiscriminantAnalysis(n_components=n_components).fit_transform(data, label)  # 训练一个pca模型
print("Train LDA in %0.3fs" % (time() - t0))

ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程

for i in range(0, len(varieties)):
    color = ['red', 'green', 'blue', 'black', 'cyan', 'magenta', 'orange', 'aquamarine',
             'darkviolet', 'teal', 'brown', 'chartreuse', 'coral', 'darkblue']
    d = data_lda[label == i]  # 找出聚类类别为0的数据对应的降维结果
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], c=color[i], marker='.', label=varieties[i])

ax.set_zlabel('LD3')  # 坐标轴
ax.set_ylabel('LD2')
ax.set_xlabel('LD1')

plt.legend()
plt.draw()
plt.show()
# plt.pause(10)
# plt.savefig('3D.jpg')
# plt.close()
