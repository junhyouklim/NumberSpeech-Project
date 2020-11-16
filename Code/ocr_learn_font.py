from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.metrics import Accuracy,CategoricalAccuracy
from sklearn.model_selection import train_test_split
import numpy as np

img_w = 28
img_h = 28
nb_classes = 10
epochs = 20


def main():
    # 폰트 이미지 데이터 읽기
    xy = np.load("./image/font_draw.npz")
    X = xy["x"]
    Y = xy["y"]
    # 데이터 정규화 하기
    X = X.reshape(X.shape[0], img_w * img_h).astype('float32')
    print("X:", X.shape)
    X /= 255
    Y = np_utils.to_categorical(Y, nb_classes)
    # 학습 전용 데이터와 테스트 전용 데이터 나누기
    x_train, x_test, y_train, y_test = \
        train_test_split(X,Y)
    '''
    # MNIST 데이터 읽어 들이기
    (x_train_mnist, y_trian_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    # 데이터 정규화
    print("x_train_mnist:{0}, x_test_mnist:{1}, y_train_mnist:{2}, y_test_mnist:{3}".format(x_train_mnist.shape,x_test_mnist.shape,y_trian_mnist.shape,y_test_mnist.shape))
    x_train_mnist = x_train_mnist.reshape(x_train_mnist.shape[0], img_w * img_h).astype('float32')
    x_test_mnist = x_test_mnist.reshape(x_test_mnist.shape[0], img_w * img_h).astype('float32')
    x_train_mnist /= 255
    x_test_mnist /= 255
    print("x_train_mnist:{0}, x_test_mnist:{1}".format(x_train_mnist.shape, x_test_mnist.shape))
    y_trian_mnist = np_utils.to_categorical(y_trian_mnist, nb_classes)
    y_test_mnist = np_utils.to_categorical(y_test_mnist, nb_classes)
    print("y_train_mnist:{0}, y_test_mnist:{1}".format(y_trian_mnist.shape, y_test_mnist.shape))
    # 폰트 이미지 MNIST 데이터 통합하기
    x_train[0] += x_train_mnist[0]
    x_test[1] += x_test_mnist[1]
    y_train[1] += y_trian_mnist[1]
    y_test[1] += y_test_mnist[1]
    print("x_train_mnist:{0}, x_test_mnist:{1}, y_train:{2}, y_test:{3}".format(x_train_mnist.shape[0],len(x_test_mnist[1]),len(y_trian_mnist[1]),len(y_test_mnist[1])))
    print("x_train:{0}, x_test:{1}, y_train:{2}, y_test:{3}".format(len(x_train), len(x_test), len(y_train), len(y_test)))
    '''
    # 모델 구축하기
    model = build_model()
    model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    # 모델 저장하기
    model.save_weights('font_draw.hdf5')
    # 모델 평가
    score = model.evaluate(x_test, y_test, verbose=0)
    print('score=', score)

def build_model():
    # MLP 모델 구축
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # loss: 오차 계산, optimizer: 학습 방법, metrics: 평가 기준
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()

