# dacon-image-classifier
## 2020 dacon: minist 이미지 분류 대회 (8.23~9.14)

---

- GPU 사용 환경 설정

- 데이터 수집이나 아이디어 보다는 cnn 활용에 초점

    - 같은 데이터로 같은 모델을 사용할 때 정확도가 달라지는 요인은 무엇인지?

----------

### GPU 사용하는 방법

cudnn과 cuda를 버전에 맞게 

```
#사용할 수 있는 장치를 알려주는 코드
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

[name: "/device:CPU:0"

 device_type: "CPU"
 
 memory_limit: 268435456
 
 locality {
 
 }
 
 incarnation: 11896794298049377862,
 
 name: "/device:XLA_CPU:0"
 
 device_type: "XLA_CPU"
 
 memory_limit: 17179869184
 
 locality {
 
 }
 
 incarnation: 1218258424309019341
 
 physical_device_desc: "device: XLA_CPU device",
 
 name: "/device:XLA_GPU:0"
 
 device_type: "XLA_GPU"
 
 memory_limit: 17179869184
 
 locality {
 
 }
 
 incarnation: 5615647587167693483
 
 physical_device_desc: "device: XLA_GPU device"]
 
```
gpus = tf.config.experimental.list_physical_devices('XLA_GPU')
print(gpus)
```

[PhysicalDevice(name='/physical_device:XLA_GPU:0', device_type='XLA_GPU')]

> **Error**
> - tuple has not attribute 'layer'
>     - keras 버전 문제
>     - library import에서 from keras -> from tensorflow.keras
> - tensorflow 2버전에서 GPU 사용하지 않는 문제
>     - cuda 재설치 (10.1) 그에 맞는 cudnn안의 폴더에서 매칭하는 cuda폴더로 파일 넣기 (덮어쓰기 x)
>     - 환경변수 설정
>     - jupyter notebook 로그에서 GPU library를 불러오지 못한다고 함 (cudnn64_7.dll) cudnn 7버전으로 다시 설치했더니 정상적으로 작동


****

### EDA

![image](https://user-images.githubusercontent.com/61912635/106430009-944b9300-64ae-11eb-940e-5dd04a9a492e.png)

알파벳 손글씨에 숫자 손글씨가 겹쳐있는 image에서 머신러닝으로 숫자를 맞추는 문제


### preprocessing

- 이미지의 크기를 3배로 확대 (28x28 -> 84x84) 후 평활화, 스트레칭, 이진화, 모폴로지 적용
 

이미지 확대

![image](https://user-images.githubusercontent.com/61912635/106431649-eab9d100-64b0-11eb-853b-119e3c9dce6f.png)

이미지 스트레칭

![image](https://user-images.githubusercontent.com/61912635/106431810-2359aa80-64b1-11eb-9832-32b983fee0f7.png)

이진화

![image](https://user-images.githubusercontent.com/61912635/106431897-484e1d80-64b1-11eb-8eba-df14ce22d127.png)

모폴로지 - 침식 3회

![image](https://user-images.githubusercontent.com/61912635/106432086-8ea37c80-64b1-11eb-9a7c-e55040d58b64.png)

+ 스트레칭 대신 평활화 했을 때

![image](https://user-images.githubusercontent.com/61912635/106432164-ae3aa500-64b1-11eb-9000-a00e586326eb.png)

### training

전처리 한 데이터와 안한 데이터의 학습 정확도 차이가 큼

stretching까지 전처리한 후 학습 -> 이진화 한 후 한번더 학습하는 방식 사용

정확도 0.98까지 달성 , test에서 최대 정확도는 0.87
