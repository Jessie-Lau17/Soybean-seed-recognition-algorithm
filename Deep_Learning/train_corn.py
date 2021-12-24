import os
#import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import time
from torchvision import transforms
from PIL import Image
from model import GoogLeNet,ResNet,AlexNet, densenet169,resnet50,resnet152,densenet161
from model_corn import G_D_net
import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
datapath = 'D:\lab_something\Agriculture\Corn\data2_1'
vad = 0.2

def Dataset(path):
    label = -1
    images_train = []
    images_vad = []
    labels_train = []
    labels_vad = []
    list_name = os.listdir(path)
    for index in list_name:
        # print(index)
        frame = 0
        label = label + 1
        list_img = os.listdir(os.path.join(datapath, index))
        list_number = len(list_img)
        split_size = int(list_number * vad)
        list_total = list(range(list_number))
        list_vad = random.sample(list_total, split_size)
        # print(list_vad)
        for i in list_img:
            frame = frame + 1
            img_name = os.path.join(datapath, index, i)
            if frame in list_vad:
                labels_vad.append(label)
                images_vad.append(img_name)
            else:
                labels_train.append(label)
                images_train.append(img_name)
    return images_train, images_vad, labels_train, labels_vad


class set_train():
    def __init__(self,images_train, labels_train, transforms=None):
        super(set_train, self).__init__()
        self.images_train = images_train
        self.labels_train = labels_train
        self.transforms = transforms

    def __getitem__(self, index):
        length = len(self.images_train)
        image = Image.open(self.images_train[index]) # 用PIL.Image读取图像
        label = torch.tensor(self.labels_train[index])
        if transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels_train)


class set_vad():
    def __init__(self, images_vad, labels_vad, transforms=None):
        super(set_vad, self).__init__()
        self.images_vad = images_vad
        self.labels_vad = labels_vad
        self.transforms = transforms

    def __getitem__(self, index):
        length = len(self.images_vad)
        image = Image.open(self.images_vad[index])  # 用PIL.Image读取图像
        label = torch.tensor(self.labels_vad[index])
        if transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels_vad)

# def make_list():
#     label = -1
#     frame = 0
#     list_name = os.listdir(datapath)
#     for index in list_name:
#         print(index)
#         label = label + 1
#         list_img = os.listdir(os.path.join(datapath,index))
#         list_number = len(list_img)
#         split_size = int(list_number * vad)
#         list_total = list(range(list_number))
#         list_vad = random.sample(list_total, split_size)
#         for i in list_img:
#             frame = frame + 1
#             img_name = os.path.join(datapath,index,i)
#             image = Image.open(img_name).convert('RGB')
#             image = transforms.ToTensor()(image)
#             info = [str(label), img_name]
#             if frame in list_vad:
#                 vad_set.append(info)
#             else:
#                 train_set.append(info)
    # datas = [train_set, vad_set]
    # names = ['train', 'vad']
    # for i in range(2):
    #     with open(datapath + '/' + names[i] + '.csv', 'w') as f:
    #         f.write('\n'.join([','.join(line) for line in datas[i]]))

net = densenet169()


def evaluate_accuracy(data_iter, net, device=None):
    loss = torch.nn.CrossEntropyLoss()
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, vad_l_sum, batch_count,n = 0.0, 0.0,0,0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            l = loss(net(X.to(device)),y.to(device))
            vad_l_sum += l.item()
            batch_count += 1
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n, vad_l_sum / batch_count


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    vad_acc = []
    vad_loss = []
    train_acc = []
    train_loss = []
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc , test_loss = evaluate_accuracy(test_iter, net)
        vad_acc.append(test_acc)
        vad_loss.append(test_loss)
        train_acc.append(train_acc_sum/n)
        train_loss.append(train_l_sum/batch_count)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, test loss %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, test_loss, time.time() - start))
    List = [vad_acc, vad_loss, train_acc, train_loss]
    data = pd.DataFrame([List])
    data.to_csv('D:\lab_something\Agriculture\Corn\D169_Net.csv', mode='a', header=False, index=False)
        # print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
    torch.save(net.state_dict(), 'D:\lab_something\Agriculture\Corn\D_Net.pth')


# transform = transforms.Compose([transforms.RandomVerticalFlip(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transform = transforms.Compose([transforms.Resize(224),transforms.RandomRotation(90),transforms.ToTensor()])
image_train, image_vad, label_train, label_vad = Dataset(datapath)
print(len(image_train))
print(len(label_train))
print(len(image_vad))
train_set = set_train(image_train,label_train,transforms=transform)
vad_set = set_vad(image_vad,label_vad,transforms=transform)
train_iter = torch.utils.data.DataLoader(train_set, 16, shuffle=True)
vad_iter = torch.utils.data.DataLoader(vad_set, 16, shuffle=False)

lr, num_epochs = 0.0001, 200
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, vad_iter, 16, optimizer, device, num_epochs)

