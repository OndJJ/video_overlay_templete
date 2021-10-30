import cv2
import numpy as np
import glob

# cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval

# 저장하고 싶은 영상의 속성값을 지정해줘야 합니다.


# • filename : 비디오 파일 이름 (e.g. 'video.mp4')

# • fourcc : fourcc (e.g. cv2.VideoWriter_fourcc(*'DIVX'))

# • fps : 초당 프레임 수 (e.g. 30)

# • frameSize : 프레임 크기. (width, height) 튜플.

# • isColor : 컬러 영상이면 True, 그렇지않으면 False. 기본값은 True입니다.

# • retval : cv2.VideoWriter 객체

img_array = []
for filename in glob.glob(r'C:\Users\ond\vs_space\intflow\theta_test\*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter(
    'project_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
