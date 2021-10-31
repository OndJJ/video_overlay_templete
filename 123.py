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

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math

def draw_circle(img, nose: tuple, neck: tuple, tail: tuple):
    frame=img

    nose_x, nose_y = nose
    neck_x, neck_y = neck
    tail_x, tail_y = tail

    nose_x, nose_y = int(nose_x), int(nose_y)
    neck_x, neck_y = int(neck_x), int(neck_y)
    tail_x, tail_y = int(tail_x), int(tail_y)

    cv2.circle(img= frame, center=(nose_x, nose_y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(img= frame, center=(neck_x, neck_y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(img= frame, center=(tail_x, tail_y), radius=5, color=(0, 0, 255), thickness=-1)

def draw_ellipse(img, center, axes, angle):
    image_temp=img
    xc_before, yc_before=center
    xc, yc=int(xc_before),int(yc_before)
    width,height=axes
    width_half=int(width/2)
    height_half=int(height/2)
    rad=angle
    theta=math.degrees(rad)
    cv2.ellipse(img= image_temp, center= (xc,yc), axes= (width_half,height_half), angle=theta, startAngle= 0, endAngle=360, color=(0, 255, 0), thickness=2)


def main():
    ##텍스트 열기
    text_file = r'drive-download-20211024T024937Z-001\mounting_001\mounting_001_det.txt'
    df = pd.read_csv(text_file)
    
    ##비디오 열기
    mp4_file = 'drive-download-20211024T024937Z-001/mounting_001/mounting_001.mp4'
    cap = cv2.VideoCapture(mp4_file)       #동영상 캡처 객체 생성
    if cap.isOpened():            #잘 열렸는지 확인
        file_path = 'trk_sungmin_1.avi'    #저장할 파일 경로 이름
        fps = cap.get(cv2.CAP_PROP_FPS)               #재생할 파일의 프레임 레이트 얻기
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)     #재생할 파일의 넓이 얻기
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   #재생할 파일의 높이 얻기
        size = (int(width), int(height))              #프레임 크기
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')      #저장할 비디오 코덱 
        out = cv2.VideoWriter(file_path, fourcc, fps, size)   #파일 stream 생성

        
        count = 0
        while(True):
            ret, frame = cap.read()  #파일로 부터 이미지 얻기
            if ret:
                # draw point
                temp_df = df[['frame','no_x','no_y','ne_x','ne_y','ta_x','ta_y']]
                temp_df = temp_df[ temp_df['frame'] == count ]
                for i in range(len(temp_df)):
                    draw_circle(img=frame, nose=(temp_df['no_x'].iloc[i,], temp_df['no_y'].iloc[i,]),
                                           neck=(temp_df['ne_x'].iloc[i,], temp_df['ne_y'].iloc[i,]),
                                           tail=(temp_df['ta_x'].iloc[i,], temp_df['ta_y'].iloc[i,]))

                # draw ellipse 
                temp_df = df[['frame','xc','yc','width','height','theta']]
                temp_df = temp_df[ temp_df['frame'] == count ]
                for i in range(len(temp_df)):
                    draw_ellipse(img=frame, center=(temp_df['xc'].iloc[i,], temp_df['yc'].iloc[i,]),
                                            axes=(temp_df['width'].iloc[i,], temp_df['height'].iloc[i,]),
                                            angle=temp_df['theta'].iloc[i,])

                out.write(frame)
                cv2.imshow('cow_ellipse', frame)        # 다음  프레임 이미지 표시
                
                if cv2.waitKey(1) == 27:    # 1ms 동안 키 입력 대기 ---②
                    break                               # 아무 키라도 입력이 있으면 중지
            else:
                print("no frame!")
                break
            count += 1
        print("{} images are extracted in {}.". format(count, mp4_file))
        out.release()          #저장 파일 종료
    else:
        print("can't open video")
    cap.release()              #재생 파일 종료 /객체 자원 반납
    cv2.destroyAllWindows()    #윈도우 종료

if __name__ == '__main__':
    main()