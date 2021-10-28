import numpy as np
import cv2
import math
a = [1, 2, 3, 4, 5], [6, 8, 7, 8, 9], [
    10, 11, 12, 13, 14], [15, 16, 17, 18, 19]
b = [5, 6, 7, 8, 'z']
c = [9, 10, 11, 12, 13]
d = [14, 15, 16, 17, 18]
e = ['a', 'b', 'c', 'd', 'e']

for (frame, x_cen, y_cen, width, height), theta in zip(a, e):
    print(frame, x_cen, y_cen, width, height, theta)


def draw_ellipse(img, center, axes, angle):
    image_temp = img
    xc_before, yc_before = center
    xc, yc = round(xc_before), round(yc_before)
    width, height = axes
    width_half = round(width/2)
    height_half = round(height/2)
    rad = angle
    theta = math.degrees(rad)
    overlay_ellipse = cv2.ellipse(img=image_temp, center=(xc, yc), axes=(
        width_half, height_half), angle=theta, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)
    # plt.imshow(overlay_ellipse)
    # plt.show()
    return overlay_ellipse


#draw_ellipse(img=frame, center=(), axes=(195.31213,469.94693), angle=0.063335426)
draw_ellipse(img=frame, center=(490.8695, 715.7377),
             axes=(203.52441, 387.11194), angle=-0.036949597)
draw_ellipse(img=frame, center=(333.08087, 721.1758),
             axes=(165.5154, 400.31036), angle=-0.032474793)

################################################################################

if __name__ == "__main__":

    # input_data = np.array([[0, 1101, 681, 97, 234], [0, 490, 715, 101, 193], [0, 333, 721, 82, 200], [0, 837, 94, 93, 98], [1, 1101, 681, 98, 235], [1, 491, 715, 101, 193], [1, 333, 721, 82, 200], [1, 838, 91, 91, 97], [2, 1101, 681, 98, 235], [2, 491, 715, 101, 193], [2, 333, 721, 82, 200], [2, 838, 91, 91, 97], [3, 1101, 682, 99, 234], [3, 490, 715, 101, 193], [3, 333, 720, 83, 199], [3, 843, 90, 90, 92]])
    # input_data = input_data[np.where(input_data[:, 0] == 0)]

    input_data = [[0, 1101, 681, 97, 234], [0, 490, 715, 101, 193], [0, 333, 721, 82, 200], [0, 837, 94, 93, 98], [1, 1101, 681, 98, 235], [1, 491, 715, 101, 193], [1, 333, 721, 82, 200], [1, 838, 91, 91, 97], [
        2, 1101, 681, 98, 235], [2, 491, 715, 101, 193], [2, 333, 721, 82, 200], [2, 838, 91, 91, 97], [3, 1101, 682, 99, 234], [3, 490, 715, 101, 193], [3, 333, 720, 83, 199], [3, 843, 90, 90, 92]]
    theta_data = [0.063335426, -0.036949597, -0.032474793, 1.5985248, 0.06306146, -0.038128957, -0.031173682,
                  1.6238793, 0.06304373, -0.0382151, -0.03137008, 1.6215281, 0.05942819, -0.03542307, -0.028387114, 1.7034304]
    img_files = ["./frame0.jpg", "./frame1.jpg",
                 "./frame2.jpg", "./frame3.jpg"]

    for (obj_idx, x_center, y_center, width, height), theta in zip(input_data, theta_data):

        img = cv2.imread("./frame{}.jpg".format(obj_idx))
        img = cv2.ellipse(img, (x_center, y_center),
                          (width, height), theta, 0, 360, (0, 256, 0), 2)

        cv2.imwrite("./frame{}.jpg".format(obj_idx), img)

#################################################################################


# 원하는 인덱스만 이미지로 그려주는 메소드
def draw_xywh(index, xywh, thetas):

    img = cv2.imread("./frame{}.jpg".format(index))
    for data, theta in zip(xywh, thetas):
        img = cv2.ellipse(
            img, (data[1], data[2]), (data[3], data[4]), theta, 0, 360, (0, 256, 0), 2)

    cv2.imwrite("./drawed_frame{}.jpg".format(index), img)


if __name__ == "__main__":
    input_data = np.array([[0, 1101, 681, 97, 234], [0, 490, 715, 101, 193], [0, 333, 721, 82, 200], [0, 837, 94, 93, 98], [1, 1101, 681, 98, 235], [1, 491, 715, 101, 193], [1, 333, 721, 82, 200], [1, 838, 91, 91, 97], [
                          2, 1101, 681, 98, 235], [2, 491, 715, 101, 193], [2, 333, 721, 82, 200], [2, 838, 91, 91, 97], [3, 1101, 682, 99, 234], [3, 490, 715, 101, 193], [3, 333, 720, 83, 199], [3, 843, 90, 90, 92]])
    theta_data = np.array([0.063335426, -0.036949597, -0.032474793, 1.5985248, 0.06306146, -0.038128957, -0.031173682,
                          1.6238793, 0.06304373, -0.0382151, -0.03137008, 1.6215281, 0.05942819, -0.03542307, -0.028387114, 1.7034304])

    # input_data에서 이미지파일에 그리고 싶은 인덱스. 여기서는 0번만 뽑아내서 좌표를 그림.
    my_selected_index = 0
    draw_xywh(
        my_selected_index,
        # input_data중에서 my_selected_index로 설정한 녀석들만 뽑아냄
        input_data[np.where(input_data[:, 0] == my_selected_index)],
        # input_data중에서 my_selected_index로 설정한 녀석들과 매칭되는 theta 데이터들을 뽑아냄
        theta_data[np.where(input_data[:, 0] == my_selected_index)]
    )

####################################################################################################
