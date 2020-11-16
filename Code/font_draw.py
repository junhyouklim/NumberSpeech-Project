import os, glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2, random

#크기 지정하기
image_size = 28 #MNIST와 같은 크기

# 폰트 설정하기
# glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다
ttf_list = glob.glob("C:/Windows/Fonts/*.ttf")
#ttf_list += glob.glob("C:/Users/user/Desktop/MaruBuri-Regular/*.ttf")
print("font count=", len(ttf_list))

#중앙에 문자 그리기
def draw_text(im, font, text):
    dr = ImageDraw.Draw(im)
    im_sz = np.array(im.size)
    fo_sz = np.array(font.getsize(text))
    xy = (im_sz - fo_sz) / 2
    print(im_sz, fo_sz)
    dr.text(xy, text, font=font, fill=(255))

# 샘플 이미지를 출력할 폴더
if not os.path.exists("./image/num"): os.makedirs("./image/num")
X = []
Y = []
#회전하거나 확대해서 데이터 늘리기
def gen_image(base_im, no, font_name):
    for ang in range(-20, 20, 2):
        sub_im = base_im.rotate(ang)
        data = np.asarray(sub_im)
        X.append(data)
        Y.append(no)
        w = image_size
        # 조금씩 확대하기
        for r in range(8, 15, 3):
            size = round((r/10) * image_size)
            im2 = cv2.resize(data, (size, size), cv2.INTER_AREA)
            data2 = np.asarray(im2)
            if image_size > size:
                x = (image_size - size) //2
                data = np.zeros((image_size,image_size))
                data[x:x+size, x:x+size] = data2
            else:
                x = (size - image_size) // 2
                data = data2[x:x+w, x:x+w]
                X.append(data)
                Y.append(no)
                if random.randint(0, 400) == 0:
                    fname = "image/num/n-{0}-{1}-{2}-{3}.png".format(font_name, no, ang, r)
                    cv2.imwrite(fname, data)

# 이미지 렌더링 하기
for path in ttf_list:
    font_name = os.path.basename(path)
    try:
        fo = ImageFont.truetype(path, size=100)
    except:
        continue
    # 0~9까지 숫자 학습 파일 생성
    for no in range(10):
        # mode: "L" -> 8 비트 픽셀, 흑백
        im = Image.new("L", (200,200))
        draw_text(im, fo, str(no))
        # 폰트 렌더링 범위 추출하기
        ima = np.asarray(im)
        blur = cv2.GaussianBlur(ima, (5,5),0)
        th = cv2.adaptiveThreshold(blur, 255, 1,1, 11, 2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                                  iterations=2)
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: continue
            num = ima[y:y+h, x:x+w] # 부분 이미지 추출하기
            ww = w if w > h else h
            wx = (ww - w) // 2
            wy = (ww - h) // 2
            spc = np.zeros((ww,ww))
            spc[wy:wy+h, wx:wx+w] = num # 중앙에 복사하기
            num = cv2.resize(spc, (image_size, image_size), cv2.INTER_AREA)
            # 표준 상태를 데이터에 추가하기
            X.append(num)
            Y.append(no)
            # 조금씩 회전하기
            base_im = Image.fromarray(np.uint8(num))
            gen_image(base_im, no, font_name)

X = np.array(X)
Y = np.array(Y)
np.savez("./image/font_draw.npz", x=X, y=Y)
print("ok", len(Y))