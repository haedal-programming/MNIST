# MNIST

MNIST with Tensorflow and Python  

<hr/>


**MNIST**는 Mixed National Institute of Standards and Technology의 약어로,  
간단한 **컴퓨터 비전 데이터 셋**으로 간단한 이미지들을 포함합니다.  <br>
이번 프로젝트의 경우 **손글씨 이미지** 등을 포함된 데이터 셋을 사용하여 진행합니다. <br>

## 환경 설치
텐서플로우를 설치해 시스템 환경을 구축해야 컴퓨터에서 **MNIST**를 실행합니다.    
필자가 여러번 설치해본 결과  
Window에 그대로 설치하는건 정신건강에 해롭습니다.  
갖고 있는 노트북에 Ubuntu를 설치 후 진행해 환경을 통일하는게 중요합니다.  
문제가 생기면 동일한 에러를 나게해 해결하는 시간을 줄이기 위함입니다.  

#### Ubuntu 설치
컴퓨터마다 환경이 모두 다르므로 여러 블로그를 확인하며 설치하는게 중요합니다.  
구글링을 생활화합시다.아래엔 유명한 블로그들을 정리했습니다.  
[PC에 Ubuntu 설치하기](http://recipes4dev.tistory.com/112)  
[윈도우 10에서 우분투 멀티부팅하기](http://palpit.tistory.com/765)  

#### Tensorflow 설치
구글에서 만들어준 텐서플로우 정식 홈페이지에서 자신의 운영체제와 하드웨어에 적합한 환경으로 설치합니다.  
[Window에 설치](https://www.tensorflow.org/install/install_windows)
[Mac에 설치](https://www.tensorflow.org/install/install_mac)
[Ubuntu에 설치](https://www.tensorflow.org/install/install_linux)
우리의 경우 Ubuntu에 설치하는 문서따라 진행합니다.  
하다가 문제가 생길 수 있습니다. 구글에서 만들어둔 문서라 틀릴리 없습니다.  
문제는 나 자신이니 빼먹거나 놓친 문구나 명령어가 있는지 확인합니다.  

## Tensorflow 연습
[Get Started with Tensorflow](https://www.tensorflow.org/tutorials/)
텐서플로우를 이용한 연습 예제입니다. 위의 링크의 문서를 따라하며 아래의 파이썬 예제를 진행하면
```python3
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
아래처럼 학습을 완료했다는 창이 나오게 됩니다.
```bash
/Users/kangminchoi/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Epoch 1/5
2018-08-27 15:00:29.959079: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
60000/60000 [==============================] - 10s 173us/step - loss: 0.2192 - acc: 0.9361
Epoch 2/5
60000/60000 [==============================] - 10s 170us/step - loss: 0.0955 - acc: 0.9708
Epoch 3/5
60000/60000 [==============================] - 10s 167us/step - loss: 0.0682 - acc: 0.9782
Epoch 4/5
60000/60000 [==============================] - 11s 177us/step - loss: 0.0524 - acc: 0.9834
Epoch 5/5
60000/60000 [==============================] - 10s 167us/step - loss: 0.0437 - acc: 0.9856
10000/10000 [==============================] - 0s 20us/step
```
이제 텐서플로우의 세계에 빠져봅시다

## MNIST 해보자  
2주나 내용이므로 추가하겠습니다
```
```
