import cv2
import os

# mp4 파일 frame 추출

folder = 'test'
os.mkdir(folder)
# use opencv to do the job
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture(
    r'mounting_000\mounting_000.mp4')
count = 0
while True:
    success, image = vidcap.read()
    if not success:
        break
    # save frame as JPEG file
    cv2.imwrite(os.path.join(folder, "frame{:d}.jpg".format(count)), image)
    count += 1
print("{} images are extacted in {}.".format(count, folder))
