import sys
import os
import re
import cv2
import numpy as np
import math
import pandas as pd



script_path = os.path.dirname(__file__)
os.chdir(script_path)


# txt file load

#cow = []
# def showFile(filename):
#    f = open(r'01_202109081705528_2\01_202109081705528_2_det.txt', 'r', encoding='utf-8')
#    lines = f.readlines()
#     for line in lines:
#         cow.append(line)
#     f.close()
# showFile('01_202109081705528_2\01_202109081705528_2_det.txt')
# #print(type(cow))
# col_name = ['frame','xc','yc','width','height','theta','no_x','no_y','ne_x','ne_y','ta_x','ta_y']
# cow_image = pd.DataFrame(data = cow, columns=col_name)
# cow_image.info()
# cow_image.head()


#data = pd.read_csv('파일경로', sep = "\t", engine='python', encoding = "인코딩방식")
data = pd.read_csv(r'01_202109081705528_2\01_202109081705528_2_det.txt',
                   sep=",", engine='python', encoding="cp949")
# data.info()

input_data = data.drop(['no_x', 'no_y', 'ne_x', 'ne_y', 'ta_x', 'ta_y'], axis=1)
cow_data = data.drop(['frame', 'xc', 'yc', 'width', 'height', 'theta'], axis=1)

# input_data.info()

input_data.head()
# cow_data.info()

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
    #return [rotated_x1, rotated_y1, rotated_x2, rotated_y2, rotated_x3, rotated_y3, rotated_x4, rotated_y4]
    print([rotated_x1, rotated_y1, rotated_x2, rotated_y2, rotated_x3, rotated_y3, rotated_x4, rotated_y4])

x = np.array(input_data['xc']).tolist()
y = np.array(input_data['yc']).tolist()
w = np.array(input_data['width']).tolist()
h = np.array(input_data['height']).tolist()
t = np.array(input_data['theta']).tolist()

for x_cen, y_cen, width, height, theta in zip(x,y,w,h,t):

    rotate_box_dot(x_cen, y_cen, width, height, theta)



