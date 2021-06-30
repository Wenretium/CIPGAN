import cv2
import torch
from PIL import Image
import numpy as np


def getMap(img, map_thresh):
    map_thresh = 1-map_thresh
    # PIL 转 cv2
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 转为灰度图
    src0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 灰度取反
    src = cv2.bitwise_not(src0)
    # 阈值截断
    hist = cv2.calcHist([src], [0], None, [256], [0, 256])
    reversed_hist = reversed(hist)

    n_pix = src.size

    cut_n_pix = int(n_pix*map_thresh)  # 取亮度前map_thresh

    # 计算像素阈值thresh
    temp = 0
    for val, val_n_pix in enumerate(reversed_hist):
        temp += val_n_pix
        if temp >= cut_n_pix:
            thresh = val
            break

    _, map = cv2.threshold(src, thresh, 255, cv2.THRESH_TOZERO)

    # 调试
    # num = np.random.randint(100)
    # cv2.imwrite('map_imgs/map{}.png'.format(num), map)
    # cv2.imwrite('map_imgs/origin{}.png'.format(num), src0)
    # cv2.imwrite('map_imgs/bitwise_not{}.png'.format(num), src)

    # cv2 转 PIL
    map = Image.fromarray(map)


    return map

# img = Image.open('../lake.jpg')
# map = getMap(img, 0.3)
# map.show()
