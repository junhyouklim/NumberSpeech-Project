import tensorflow as tf
import cv2
import numpy as np
import os
import ocr_learn_font
import playsound
import quickstart

#MNIST 학습 데이터 읽어 들이기
mnist = ocr_learn_font.build_model()
#mnist.load_weights('mnist.hdf5')
mnist.load_weights('font_draw.hdf5')

def tts_api(text):
    quickstart.run_quickstart(text=text)
    filename = "num.mp3"
    playsound.playsound(filename)

def process(img_input):
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    (thresh, img_binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h, w = img_binary.shape

    ratio = 100 / h
    new_h = 100
    new_w = w * ratio

    img_empty = np.zeros((110, 110), dtype=img_binary.dtype)
    img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    img_empty[:img_binary.shape[0], :img_binary.shape[1]] = img_binary

    img_binary = img_empty

    cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어의 무게중심 좌표를 구합니다.
    M = cv2.moments(cnts[0][0])
    center_x = (M["m10"] / M["m00"])
    center_y = (M["m01"] / M["m00"])

    # 무게 중심이 이미지 중심으로 오도록 이동시킵니다.
    height, width = img_binary.shape[:2]
    shiftx = width / 2 - center_x
    shifty = height / 2 - center_y

    Translation_Matrix = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_binary = cv2.warpAffine(img_binary, Translation_Matrix, (width, height))

    img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
    flatten = img_binary.flatten() / 255.0

    return flatten

'''
capture.get(속성) : VideoCapture의 속성을 반환합니다.
capture.grab() : Frame의 호출 성공 유/무를 반환합니다.
capture.isOpened() : VideoCapture의 성공 유/무를 반환합니다.
capture.open(카메라 장치 번호 또는 경로) : 카메라나 동영상 파일을 엽니다.
capture.release() : VideoCapture의 장치를 닫고 메모리를 해제합니다.
capture.retrieve() : VideoCapture의 프레임과 플래그를 반환합니다.
capture.set(속성, 값) : VideoCapture의 속성의 값을 설정합니다.

cv2.CAP_PROP_FRAME_WIDTH	프레임의 너비	-
cv2.CAP_PROP_FRAME_HEIGHT	프레임의 높이	-
cv2.CAP_PROP_FRAME_COUNT	프레임의 총 개수	-
cv2.CAP_PROP_FPS	프레임 속도	-
cv2.CAP_PROP_FOURCC	코덱 코드	-
cv2.CAP_PROP_BRIGHTNESS	이미지 밝기	카메라만 해당
cv2.CAP_PROP_CONTRAST	이미지 대비	카메라만 해당
cv2.CAP_PROP_SATURATION	이미지 채도	카메라만 해당
cv2.CAP_PROP_HUE	이미지 색상	카메라만 해당
cv2.CAP_PROP_GAIN	이미지 게인	카메라만 해당
cv2.CAP_PROP_EXPOSURE	이미지 노출	카메라만 해당
cv2.CAP_PROP_POS_MSEC	프레임 재생 시간	ms 반환
cv2.CAP_PROP_POS_FRAMES	현재 프레임	프레임의 총 개수 미만
CAP_PROP_POS_AVI_RATIO	비디오 파일 상대 위치	0 = 시작, 1 = 끝
'''
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while (True):

    ret, img_color = cap.read()

    if ret == False:
        break

    img_input = img_color.copy()
    cv2.rectangle(img_color, (250, 150), (width - 250, height - 150), (0, 0, 255), 3)
    cv2.imshow('CamImage', img_color)
    # 화면 캡처 허용 범위 지정
    img_roi = img_input[150:height - 150, 250:width - 250]

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        flatten = process(img_roi)
        # predict 함수를 사용하여 예측치와 실측치 분석
        predictions = mnist.predict(flatten[np.newaxis, :])
        with tf.compat.v1.Session() as sess:
            '''
            one-hot(원핫)인코딩이란? 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말한다.
            MNIST 코드에서는 one hot 벡터로 표현한 라벨이 의미하는 숫자를 찾기 위해 tf.argmax 함수를 사용됩니다. 
            [ 0 0 1 0 0 0 0 0 0 0]
            라벨이 1차원 벡터인데 실제 코드에서 보면 tf.argmax 함수의 두번째 인자로 1을 사용하고 있습니다. 
            이것은 pred와 y의 shape를 출력해보면 알 수 있습니다. (?, 10) 로 출력됩니다. 
            첫번째 차원은 라벨의 갯수를 표현하기 위해 사용되므로 크기가 정해져 있지 않으며 
            두번째 차원은 0~9까지 10개의 숫자를 위한 라벨로 사용하기 때문에 10입니다.
            두번째 차원을 라벨로 사용하기 때문에 0이 아닌 1을 사용하게 됩니다. 즉 각 행에서 최대값을 찾습니다. 
            eval() 함수는 문자열로 입력된 숫자를 수 형태로 알아서 바꿔주는 함수
            '''
            print(tf.argmax(predictions, 1).eval())
            num = int(tf.argmax(predictions, 1).eval())
            print(num)
            '''
            while (True):
                try:
                    speak(str(num))
                    break
                except:
                    print("error")
            '''
            tts_api(str(num))
            os.remove('num.mp3')
        cv2.imshow('CatchNumber', img_roi)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()