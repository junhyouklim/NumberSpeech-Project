import sys
import numpy as np
import cv2
import ocr_learn_font
import playsound
import quickstart
import os

def tts_api(text):
    quickstart.run_quickstart(text=text)
    filename = "num.mp3"
    playsound.playsound(filename)

def numberProcess(img_roi):
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 블러
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)  # 2진화

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # 추출한 좌표 정렬하기
    rects = []
    im_w = img_roi.shape[1]
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10: continue  # 너무 작으면 생략하기
        if w > im_w / 5: continue  # 너무 크면 생략하기
        y2 = round(y / 10) * 10  # y 좌표 맞추기
        index = y2 * im_w + x
        rects.append((index, x, y, w, h))
    rects = sorted(rects, key=lambda x: x[0])  # 정렬하기

    # 해당 영역의 이미지 데이터 추출하기
    for i, r in enumerate(rects):
        index, x, y, w, h = r
        num = gray[y:y + h, x:x + w]  # 부분 이미지 추출하기
        num = 255 - num  # 반전하기
        # 정사각형 내부에 그림 옮기기
        ww = round((w if w > h else h) * 1.2)
        spc = np.zeros((ww, ww))
        wy = (ww - h) // 2
        wx = (ww - w) // 2
        spc[wy:wy + h, wx:wx + w] = num
        num = cv2.resize(spc, (28, 28))  # MNIST 크기 맞추기
        cv2.imshow("Num", num)  # 자른 문자 저장하기
        cv2.waitKey(0)
        # 데이터 정규화
        num = num.reshape(28 * 28)
        num = num.astype("float32") / 255
        X.append(num)

#MNIST 학습 데이터 읽어 들이기
mnist = ocr_learn_font.build_model()
mnist.load_weights('font_draw.hdf5')

#이미지 읽어 들이기
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
X = []
while(True):
    #윤곽 추출하기
    ret, img_color = cap.read()

    img_input = img_color.copy()
    cv2.rectangle(img_color, (150, 100), (width - 150, height - 100), (0, 0, 255), 3)
    cv2.imshow('CamImage', img_color)
    # 화면 캡처 허용 범위 지정
    img_roi = img_input[100:height - 100, 150:width - 150]

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32:
        numberProcess(img_roi)

        nlist = mnist.predict(np.array(X))
        print('nlist:', nlist)
        for i, n in enumerate(nlist):
            ans = n.argmax()
            print(ans)
            tts_api(str(ans))
            os.remove('num.mp3')
        cv2.imshow('CatchNumber', img_roi)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

