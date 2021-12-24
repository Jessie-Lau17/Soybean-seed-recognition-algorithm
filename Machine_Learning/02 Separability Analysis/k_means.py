import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D #3D的库
def kmeans():
    """
    Kmeams聚类分析啤酒
    :return:
    """
    #读取数据
    data = pd.read_table('test_data.txt',sep='\t',error_bad_lines=False)
    data.columns = ['name','area' ,'length', 'R','G','B','Sum']
    #print(data)

    #提取特征值
    X = data.loc[:,['area' ,'G', 'B',]]
    print('here')
    print(X)
    print('there')
    #我们进行标准化看看效果
    # std = StandardScaler()
    # X = std.fit_transform(X)
    #print(X)
    #print(X[2:, 1])
    #使用Kmeans聚类
    kmean = KMeans(n_clusters=5)
    km = kmean.fit(X)
    labels_real=np.arange(2137)
    labels_real[0:472]=0
    labels_real[472:896]=1
    labels_real[896:1317]=2
    labels_real[1317:1735]=3
    labels_real[1735:2137]=4
    #print(labels[0:250])
    #print(labels)

    km.labels_=labels_real
    data['cluster'] = km.labels_
    print(km.labels_)
    centers = data.groupby("cluster").mean().reset_index()
    # print(centers)
    #画图，四个特征量量比较
    plt.rcParams['font.size'] = 14
    colors = np.array(['yellow','green','blue',"black",'red'])
    plt.scatter(data['area'],data['Sum'],c=colors[data["cluster"]])
    plt.scatter(centers.area,centers.Sum,linewidths=3,marker='+',s=300,c='black')#中心点
    plt.xlabel("area")
    plt.ylabel("Sum")
    plt.show()

    ##3D图像绘制
    # fignum = 1
    # fig = plt.figure(fignum, figsize=(4, 3))
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # kmean.fit(X) #fit建立模型
    # labels = labels_real #获得模型聚类后的label
    # ax.scatter(data['R'],data['G'],data['B'],
    #            c=labels.astype(np.float), edgecolor='k') #绘制X中的第3，0，2个维度的特征
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
    # ax.set_xlabel('area') #设置坐标轴名
    # ax.set_ylabel('length')
    # ax.set_zlabel('Sum')
    # ax.set_title('5-3D clusters') #设置图的名字
    # ax.dist = 12
    # fig.show() #绘制整张图
    # plt.show()
    return None

if __name__ == "__main__":
    kmeans()