import sys
import os
import re
import csv
import cv2
import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
from pandas.core import frame
import yaml


script_path = os.path.dirname(__file__)
os.chdir(script_path)


# txt file load
#data = pd.read_csv('파일경로', sep = "\t", engine='python', encoding = "인코딩방식")
data = pd.read_csv(r'mounting_000\mounting_000_det.txt',
                   sep=",", engine='python', encoding="cp949")

# file list 형식으로 콜 하는 방법
# file_list = os.listdir(DIR_IN)
# for file in file_list:          # 코드간결화 작업전
#     if file.endswith(".jpg"):
#         print(file)

# f = open('mounting_000\mounting_000_det.txt', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for line in rdr:
#     print(line)
# f.close()

input_data = data.drop(['no_x', 'no_y', 'ne_x', 'ne_y',
                       'ta_x', 'ta_y', 'theta'], axis=1)
cow_data = data.drop(['frame', 'xc', 'yc', 'width', 'height', 'theta'], axis=1)
frame_data = data['frame']
frame_data = np.array(frame_data, dtype=np.int64)  # .tolist()
theta_data = data['theta']
theta_data = np.array(theta_data, dtype=np.float64).tolist()


input_data['xc'] = input_data['xc'].astype(int)
input_data['yc'] = input_data['yc'].astype(int)
input_data['width'] = (input_data['width']/2).astype(int)
input_data['height'] = (input_data['height']/2).astype(int)
input_data = np.array(input_data, dtype=np.int64).tolist()

print(theta_data[:5])

degrees_theta = []

for i in theta_data:
    math_degree = math.degrees(i)
    degrees_theta.append(math_degree)
print(degrees_theta[:5])

# 4번


def rotate(origin, point, radian):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)
    return round(qx), round(qy)

# 1번


def rotate_box_dot(x_cen, y_cen, width, height, theta):

    # 2번 최솟값 연산
    x_min = x_cen - (width / 2)
    y_min = y_cen - (height / 2)

    # 5번 rotated_x1 --x_4 & y_1 -- y_4 연산
    # 3번 rotate함수 호출
    rotated_x1, rotated_y1 = rotate(
        origin=(x_cen, y_cen), point=(x_min, y_min), radian=theta)
    rotated_x2, rotated_y2 = rotate(
        origin=(x_cen, y_cen), point=(x_min, y_min+height), radian=theta)
    rotated_x3, rotated_y3 = rotate(origin=(x_cen, y_cen), point=(
        x_min+width, y_min+height), radian=theta)
    rotated_x4, rotated_y4 = rotate(
        origin=(x_cen, y_cen), point=(x_min+width, y_min), radian=theta)
    # 6번 최종 출력
    return [rotated_x1, rotated_y1, rotated_x2, rotated_y2, rotated_x3, rotated_y3, rotated_x4, rotated_y4]


if __name__ == "__main__":
    # ==========
    # target image
    # img_files = glob.glob('.\\test\\*.jpg')
    # cv2.imread(argument.img_path) 
    count = 0
    for (data_frame, x_cen, y_cen, width, height), theta in zip(input_data, degrees_theta):

        count += 1

        images = cv2.imread((r"C:\Users\ond\vs_space\intflow\theta_test\frame{}.jpg").format(
            data_frame), cv2.IMREAD_UNCHANGED)
        #image_temp = cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
        img = cv2.ellipse(images, (x_cen, y_cen), (width, height),
                          theta, 0, 360, (0, 256, 0), 2)
        cv2.imwrite(
            r"C:\Users\ond\vs_space\intflow\theta_test\frame{}.jpg".format(data_frame), img)
        print(count)
