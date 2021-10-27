import sys
import os
import re
import cv2
import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt


script_path = os.path.dirname(__file__)
os.chdir(script_path)


# txt file load

# data = pd.read_csv('파일경로', sep = "\t", engine='python', encoding = "인코딩방식")
data = pd.read_csv(r'mounting_000\mounting_000_det.txt',
                   sep=",", engine='python', encoding="cp949")
# data.info()

input_data = data.drop(['no_x', 'no_y', 'ne_x', 'ne_y', 'ta_x', 'ta_y'], axis=1)
cow_data = data.drop(['frame', 'xc', 'yc', 'width', 'height', 'theta'], axis=1)

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
    rotated_x1, rotated_y1 = rotate(origin=(x_cen, y_cen), point=(x_min, y_min), radian=theta)
    rotated_x2, rotated_y2 = rotate(origin=(x_cen, y_cen), point=(x_min, y_min+height), radian=theta)
    rotated_x3, rotated_y3 = rotate(origin=(x_cen, y_cen), point=(x_min+width, y_min+height), radian=theta)
    rotated_x4, rotated_y4 = rotate(origin=(x_cen, y_cen), point=(x_min+width, y_min), radian=theta)

    # xCenter = (rotated_x2 + rotated_x1)/2
    # YCenter = (rotated_y4 + rotated_y2)/2
    # Xwidth = (rotated_x2 - rotated_x1)/2
    # Xheight = (rotated_y3 - rotated_y1)/2
    # 6번 최종 출력
    return [rotated_x1, rotated_y1, rotated_x2, rotated_y2, rotated_x3, rotated_y3, rotated_x4, rotated_y4]
    # print([rotated_x1, rotated_y1, rotated_x2, rotated_y2,
    #      rotated_x3, rotated_y3, rotated_x4, rotated_y4])
    # print(Xwidth,Xheight)

# target image
img_files=glob.glob('.\\test\\*.jpg') 
f = input_data['frame']
x = input_data['xc'].round()
y = input_data['yc'].round()
w = input_data['width'].round()
h = input_data['height'].round()
t = input_data['theta'].round()

f_1= np.array(f).tolist()
x_1= np.array(x).tolist()
y_1= np.array(y).tolist()
w_1= np.array(w).tolist()
h_1= np.array(h).tolist()
t_1= np.array(t).tolist()

image=img_files
# #r'C:\Users\ond\vs_space\intflow\test\frame0.jpg'
# for x_cen, y_cen, width, height, theta, target in zip(x_1, y_1, w_1, h_1, t_1,image):
    
#     image = cv2.imread(target, cv2.IMREAD_UNCHANGED)
#     image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.ellipse(image_temp, (x_cen,y_cen), (width,height), theta, 0, 360, (0,256,0), 2)

print(x_1)        

for target in image :
    image = cv2.imread(target, cv2.IMREAD_UNCHANGED)
    image_temp = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for x_cen, y_cen, width, height, theta, images in zip(x_1, y_1, w_1, h_1, t_1, image_temp):
        img = cv2.ellipse(images, (x_cen,y_cen), (width,height), theta, 0, 360, (0,256,0), 2)

#img = cv2.rectangle(image_temp, (383, 519), (564, 918), (0,255,0), 3)
#plt.imshow(img)
#plt.show()

# 
# for x_cen, y_cen, width, height, theta in zip(x_1, y_1, w_1, h_1, t_1):
    
#     val = rotate_box_dot(x_cen, y_cen, width, height, theta)
#     val_count = [x_cen, y_cen, width, height, theta]

#     image = cv2.imread(r'C:\Users\ond\vs_space\intflow\test\frame0.jpg', cv2.IMREAD_UNCHANGED)
#     image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.ellipse(image_temp, (s,y_1), (w_1,h_1), t_1, 0, 360, (0,256,0), 2)

#img = cv2.rectangle(image_temp, (383, 519), (564, 918), (0,255,0), 3)
plt.imshow(img)
plt.show()

