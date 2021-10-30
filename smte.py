import cv2
import os
import math

# mp4 파일 frame 추출
folder = 'theta_test'
#os.mkdir(folder)
# use opencv to do the job
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture(
    r'mounting_000\mounting_000.mp4')

tscap = cv2.VideoCapture(
    r'project.avi')
count = 0
while True:
    success, image = vidcap.read()
    if not success:
        break
    # save frame as JPEG file
    cv2.imwrite(os.path.join(folder, "frame{:d}.jpg".format(count)), image)
    count += 1
print("{} images are extacted in {}.".format(count, folder))


print('원본 영상 info : ','mounting_000\mounting_000.mp4')
width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = vidcap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
all = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d, 전체 프레임 수: %d' %(width, height, fps, all))
print('###############################')


print('변환 영상 info : ','project.avi')
width = tscap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = tscap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = tscap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
all = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d, 전체 프레임 수: %d' %(width, height, fps,all))
print('###############################')