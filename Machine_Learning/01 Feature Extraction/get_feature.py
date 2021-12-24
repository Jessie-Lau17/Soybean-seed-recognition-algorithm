import cv2
import math
import numpy as np
import scipy.stats as st
import self_function as sf
from skimage import feature as skft

Variety_Path = 'dataset/data_ori/soybean/Varieties.txt'
DataSet_Path = 'dataset/data_cut/soybean/'
Feature_SavePath = 'dataset/data_feature/soybean/'
LBP_SavePath = 'dataset/data_feature/soybean/lbp_feature/'
Feature_Num = 7

sf.mkdir(Feature_SavePath)
sf.mkdir(LBP_SavePath)

f = open(Variety_Path, 'r')
data = f.read().splitlines()
f.close()

print('The number of varieties is %d. ' % len(data))
num = 1

for var in data:
    open_path = DataSet_Path + var + '/'
    save_path = Feature_SavePath + var + '.csv'
    lbp_save_path = LBP_SavePath + var + '/'
    sf.mkdir(lbp_save_path)

    feature = np.zeros([1, Feature_Num])
    hist = np.zeros([1, 256])
    feature_temp = np.zeros([1, Feature_Num])

    print('Processing the No.%d ' % num + var + '...')
    num += 1

    for i in range(1, 500):
        img = cv2.imread(open_path + '%d.png' % i)
        if img is not None:
            """  特征1：周长  """
            ret, thresh = cv2.threshold(img[:, :, 0], 0, 1, cv2.THRESH_BINARY)
            thresh = sf.FillHole(thresh)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                Attention = '[Attention]: The picture has more than one contours! '
                Note = '[Note]: The picture is ' + open_path + '%d.png' % i
                print('\033[0;33;40m' + Attention + '\033[0m')
                print('\033[0;33;40m' + Note + '\033[0m')
                continue
            else:
                perimeter = cv2.arcLength(contours[0], True)
                x, y, w, h = cv2.boundingRect(contours[0])
            feature_temp[0, 0] = perimeter

            """  特征2：面积  """
            thresh = np.uint8(thresh / 255)
            area = np.sum(thresh)
            feature_temp[0, 1] = area

            """  特征3：轮廓长  """
            feature_temp[0, 2] = h

            """  特征4：轮廓宽  """
            feature_temp[0, 3] = w

            """  特征5：轮廓长宽比  """
            feature_temp[0, 4] = h / w

            """  特征6：轮廓圆形度  """
            roundness = (4 * math.pi * area) / (perimeter * perimeter)
            feature_temp[0, 5] = roundness

            """  特征7：LBP纹理特征  """
            data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = skft.local_binary_pattern(data, 8, 1, 'default')
            cv2.imwrite(lbp_save_path + '%d.png' % i, lbp)
            # cv2.imshow('lbp', lbp)
            # hist size:256
            hist_temp, _ = np.histogram(lbp, bins=256, range=(0, 255))
            hist_temp = hist_temp.reshape(1, -1)

            """  特征8：直方图的三阶矩  """
            skewness = st.skew(hist_temp[0, :])
            feature_temp[0, 6] = skewness

            """  综合：保存特征数据  """
            feature = np.append(feature, feature_temp, axis=0)
            hist = np.append(hist, hist_temp, axis=0)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    feature = np.delete(feature, 0, axis=0)
    hist = np.delete(hist, 0, axis=0)
    feature = np.append(feature, hist, axis=1)
    np.savetxt(save_path, feature, fmt='%.6f', delimiter=",")
    # np.savetxt(save_path, hist, fmt='%.6f', delimiter=",")
