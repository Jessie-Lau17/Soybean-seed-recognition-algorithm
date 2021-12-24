这里放置深度学习算法设计代码

model.py 放置了GoogLeNet、AlexNet、VGGNet、ResNet、DenseNet的模型

model_corn.py 放置了G_D_Net的模型

train_confusion.py

将数据集按照6:2:2分为训练集、验证集、测试集，并且训练完成后在模型中放入测试集绘制混淆矩阵

train_corn.py

将数据集按照8:2分为训练集、验证集对模型进行训练

train_google.py

针对googlenet网络的两个分支分类器，专门训练googlenet网络
