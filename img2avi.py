import cv2
import numpy as np
import glob
import os

# cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval
# 저장하고 싶은 영상의 속성값을 지정해줘야 합니다.
# • filename : 비디오 파일 이름 (e.g. 'video.mp4')
# • fourcc : fourcc (e.g. cv2.VideoWriter_fourcc(*'DIVX'))
# • fps : 초당 프레임 수 (e.g. 30)
# • frameSize : 프레임 크기. (width, height) 튜플.
# • isColor : 컬러 영상이면 True, 그렇지않으면 False. 기본값은 True입니다.
# • retval : cv2.VideoWriter 객체


img_array = []

for filename in sorted(glob.glob(r'.\001\*.jpg'), key=os.path.getctime):
    print(filename)
    img = cv2.imread(filename)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # k = cv2.waitKey(0)
    # if k == 27: #esc
    #     cv2.destroyAllWindows()
    file_path = 'project_11_trk.avi'
    height, width, layers = img.shape
    size = (int(width), int(height))
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    img_array.append(img)


out = cv2.VideoWriter(file_path, fourcc, fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()