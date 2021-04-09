# Chapter 3. 신경망(Neural Network)
- 퍼셉트론
	- 퍼셉트론으로 복잡한 함수도 표현 가능
	- 원화는 결과 도출을 위한 가중치 값 설정은 사람이 수동으로 해야 함.

=> **신경망**은 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습
## 3.1 퍼셉트론에서 신경망으로
### 3.1 신경망의 예
![신경망의 예](https://user-images.githubusercontent.com/61455647/113971138-24bd9f80-9873-11eb-858b-42327588649b.png)
- 0층이 입력층, 1층이 은닉층. 2층이 출력층인 신경망
- 가중치를 갖는 층이 2개이므로 '2층 신경망'
- 은닉층은 사람 눈에 보이지 않고, 입력층과 출력층만 눈에 보임.
### 3.1.2 퍼셉트론 복습
[chapter02-Summary.md-2-1-퍼셉트론이란](https://github.com/kyurimki/Study-DeepLearningFromScratch/blob/main/chapter02/chapter02-Summary.md#21-%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0%EC%9D%B4%EB%9E%80)
- 편향을 명시한 퍼셉트론

![편향을 명시한 퍼셉트론](https://user-images.githubusercontent.com/61455647/113971739-53884580-9874-11eb-94cf-25e147ef6404.png)

  - 가중치가 b이고 입력이 1인 뉴런 추가
	- x1, x2, 1로 이루어진 3개의 신호가 뉴런에 입력 -> * (가중치) -> 다음 뉴런에 전달 -> 입력된 신호의 값을 더했을 때 합이 0보다 크면 1 출력, 아니면 0 출력
	- 편향의 입력 신호는 항상 1
- 이에 따라 식을 재작성할 수 있음

![기존 식](https://user-images.githubusercontent.com/61455647/113534842-c7c0b000-960c-11eb-851c-af88e2e79a5b.png)

![재작성한 식](https://user-images.githubusercontent.com/61455647/113972185-32742480-9875-11eb-8aee-dc637a928292.png)

  - 입력 신호의 총합이 *h(x)* 를 거쳐, y의 값으로 출력
	- *h(x)* 함수는 입력이 0보다 크면 1을 출력, 아니면 0을 출력
### 3.1.3 활성화 함수의 등장
- 활성화 함수(activation function)
	- 입력 신호의 총합을 출력 신호로 변환하는 함수, *h(x)*
	- 입력 신호의 총합이 활성화를 일으키는가
- *y = h(b + w1x1 + w2x2)* 를 다음과 같이 다시 쓸 수 있다.
    => **a = b + w1x1 + w2x2, y = h(a)**
    - a = 가중치가 달린 입력 신호와 편향의 총합
    - a를 함수 h()에 넣어 y를 출력

![활성화 함수의 처리 과정](https://user-images.githubusercontent.com/61455647/113974141-669d1480-9878-11eb-9e2e-a5ea58c36637.png)

- (참고) 단순 vs. 다층 퍼셉트론
	- 단순 퍼셉트론: 단층 네트워크에서 계단 함수(임계값을 경계로 출력이 바뀌는 함수)를 활성화 함수로 사용한 모델
	- 다층 퍼셉트론: 신경망(여러 층으로 구성되고 매끈한 활성화 함수를 사용하는 네트워크)
## 3.2 활성화 함수
- 계단 함수(step function): 임계값을 경계로 출력이 바뀜
### 3.2.1 시그모이드 함수(sigmoid function)
- 신경망에서 자주 이용하는 활성화 함수

![시그모이드 함수식](https://user-images.githubusercontent.com/61455647/113975198-1e7ef180-987a-11eb-8d00-6a266f7cc2d7.png)

- 신경망에서는 활성화 함수로 신호를 변환해, 변환된 신호를 다음 뉴런에 전달
- 퍼셉트론 vs. 신경망: 활성화 함수의 여부
### 3.2.2 계단 함수 구현하기-3.2.4 시그모이드 함수 구현하기
[3.2.2 계단 함수 구현하기-3.2.4 시그모이드 함수 구현하기](https://github.com/kyurimki/Study-DeepLearningFromScratch/blob/main/chapter03/source-3-2-ActivationFunctions.ipynb)
### 3.2.5 시그모이드 함수와 계단 함수 비교
- 시그모이드 함수
	- 부드러운 곡선
	- 입력에 따라 출력이 연속적으로 변화
	- 매끄러움
	- 실수값을 return
	- => 신경망에서는 연속적인 실수가 흐름
- 계단 함수
	- 0을 경계로 출력에 변화
	- 0과 1 중 하나의 값만 return
	- => 퍼셉트론에서는 뉴런 사이에 0과 1만 흐름
- 시그모이드 함수와 계단 함수의 공통점
	- 입력이 작으면 출력은 0에 가깝고, 입력이 커지면 출력이 1에 가까워짐
	- 출력은 [0, 1] 사이의 값
### 3.2.6 비선형 함수
- 시그모이드 함수와 계단 함수 모두 비선형 함수
- 신경망에서는 활성화 함수로 비선형 함수를 사용해야 함
   ∵ 선형 함수를 이용하면 신경망의 층을 깊게 하는 의미가 없음
- 선형 함수는 **층을 깊게 해도 은닉층이 없는 네트워크로도 같은 기능을 할 수 있다**는 문제
	- ex. if) *h(x) = cx* 가 활성화 함수인 3층 네트워크: y = h(h(h(x))) = c*c*c*x = ax
	- 은닉층이 없는 네트워크로 표현 가능
### 3.2.7 ReLU(Rectified Linear Unit) 함수
- 입력 > 0이면 입력값을 출력하고, 입력 <= 0이면 0을 출력

![ReLU function graph](https://user-images.githubusercontent.com/61455647/114129134-ef7b8500-9938-11eb-9bdf-e8504b6bc0fc.png)
![ReLU function](https://user-images.githubusercontent.com/61455647/114129227-1e91f680-9939-11eb-8964-994c325b17da.png)
```
def relu(x):
    return np.maximum(0, x)
```
- `maximum()`: 두 입력 중 큰 값을 선택해 반환
## 3.3 다차원 배열의 계산
### 3.3.1 다차원 배열
- 다차원 배열: 숫자를 N차원으로 나열
```
>>> import numpy as np

>>> A = np.array([1, 2, 3, 4])
>>> print(A)
[1 2 3 4]
>>> np.ndim(A)
1
>>> A.shape
(4,)
>>> A.shape[0]
4
```
- `np.ndim()`: 배열의 차원수 확인
- `shape`: 배열의 형상, **tuple 반환**(1차원 배열도 다차원 배열일 때와 통일된 형태로 결과 반환하기 위함)
```
>>> B = np.array([[1, 2], [3, 4], [5, 6]])
>>> print(B)
[[1 2]
 [3 4]
 [5 6]]
>>> np.ndim(B)
2
>>> B.shape
(3, 2)
```
- 3*2 배열인 B
	- 0번째 차원에 원소 3개, 1번째 차원에 원소 2개
	- **행렬** = 2차원 배열-**행**: 가로 방향, **열**: 세로 방향

![행렬](https://user-images.githubusercontent.com/61455647/114130322-42eed280-993b-11eb-8d5a-d26a41d416fb.png)
### 3.2.2 행렬의 곱
![image](https://user-images.githubusercontent.com/61455647/114131449-97934d00-993d-11eb-93db-6446cdb3f16f.png)
```
# 2*2 행렬 2개 곱
>>> A = np.array([[1, 2], [3, 4]])
>>> A.shape
(2, 2)

>>> B = np.array([[5, 6], [7, 8]])
>>> B.shape
(2, 2)
>>> np.dot(A,B)
array([[19, 22],
       [43, 50]])
```
- `np.dot()`
	- 1차원 배열이면 벡터를, 2차원 배열이면 행렬 곱 계산
	- `np.dot(A, B)`와 `np.dot(B, A)`는 다른 값일 수 있음
```
# 2*3 행렬과 3*2 행렬의 곱
>>> A = np.array([[1, 2, 3], [4, 5, 6]])
>>> A.shape
(2, 3)

>>> B = np.array([[1, 2], [3, 4], [5, 6]])
>>> B.shape
(3, 2)
>>> np.dot(A, B)
array([[22, 28],
       [49, 64]])
```
- **행렬 A, B의 연산을 하기 위해서는, A의 열 수 = B의 행 수**
```
# 2*3 행렬과 2*2 행렬의 곱 연산 실패
>>> C = np.array([[1, 2], [3, 4]])
>>> C.shape
(2, 2)
>>> A.shape
(2, 3)
>>> np.dot(A, C)
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "<__array_function__ internals>", line 6, in dot
ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
```
![3*2 행렬과 2*4 행렬의 곱](https://user-images.githubusercontent.com/61455647/114131538-c0b3dd80-993d-11eb-9e43-1d086eaef480.png)
- 3\*2 행렬 A와 2\*4 행렬 B를 곱해 3*4 행렬 C를 만들 때
	- 행렬 A의 열 수와 B의 행 수가 같아야 함
	- 결과인 행렬 C의 형상은 A의 행 수와 B의 열 수

![3*2 행렬과 1차원 배열의 곱](https://user-images.githubusercontent.com/61455647/114131833-459ef700-993e-11eb-8d38-27bc16460c66.png)
- A가 2차원 배열이고, B가 1차원 배열일 때도 똑같이 적용됨.
```
>>> A = np.array([[1, 2], [3, 4], [5, 6]])
>>> A.shape
(3, 2)

>>> B = np.array([7, 8])
>>> B.shape
(2,)

>>> np.dot(A, B)
array([23, 53, 83])
```
### 3.3.3 신경망에서의 행렬 곱
- 편향과 활성화 함수 없이 가중치만 갖는 신경망

![image](https://user-images.githubusercontent.com/61455647/114132175-d2e24b80-993e-11eb-8d88-028df01bf5c6.png)
```
>>> X = np.array([1, 2])
>>> X.shape
(2,)

>>> W = np.array([[1, 3, 5], [2, 4, 6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape
(2, 3)

>>> Y = np.dot(X, W)
>>> print(Y)
[5 11 17]
```
## 3.4 3층 신경망 구현하기
[3.4 3층 신경망 구현하기](https://github.com/kyurimki/Study-DeepLearningFromScratch/blob/main/chapter03/source-3-4-3LayerNeuralNetwork.ipynb)
## 3.5 출력층 설계하기
- 신경망은 분류와 회귀 모두 이용 가능
- 활성화 함수의 차이: 회귀=항등 함수, 분류=소프트맥스 함수
- 분류: 데이터가 어느 클래스에 속하느냐
	- ex. 사진 속 인물의 성별 분류
- 회귀: 입력 데이터에서 (연속적인) 수치를 예측
	- ex. 사진 속 인물의 몸무게 예측
### 3.5.1 항등 함수와 소프트맥스 함수 구현하기
- 항등 함수(identity function)
	- 입력을 그대로 출력 = 입력과 출력이 항상 같다
	- 출력층에서 항등 함수를 사용하면 입력 신호가 그대로 출력 신호가 됨
	- 항등 함수에 의한 변환은 은닉층의 활성화 함수와 마찬가지로 화살표로 나타냄

![항등 함수](https://user-images.githubusercontent.com/61455647/114146746-ed262480-9952-11eb-85a1-6fa63b85a97e.png)

- 소프트맥스 함수(softmax function)
	- exp(x): e^x의 지수함수
	- n: 출력층의 뉴런 수
	- yk: k번째 출력
	- 분자: 입력 신호 ak의 지수함수
	- 분모: 모든 입력 신호의 지수함수의 합
	- 소프트맥스의 출력은 모든 입력 신호로부터 화살표를 받음 ∵ 출력층의 각 뉴런이 모든 입력 신호에서 영향을 받기 때문

![소프트맥스 함수 수식](https://user-images.githubusercontent.com/61455647/114146846-08912f80-9953-11eb-844c-3e359dedbdfa.png)



![소프트맥스 함수](https://user-images.githubusercontent.com/61455647/114146993-31192980-9953-11eb-8f0b-fe7341d6f7a6.png)



```
>>> a = np.array([0.3, 2.9, 4.0])
>>> exp_a = np.exp(a)  # exponential function
>>> print(exp_a)
[ 1.34985881 18.17414537 54.59815003]

>>> sum_exp_a = np.sum(exp_a)  # sum of exponential functions
>>> print(sum_exp_a)
74.1221542101633

>>> y = exp_a / sum_exp_a
>>> print(y)
[0.01821127 0.24519181 0.73659691]
```
```
def softmax(a):
	exp_a = np.exp(a)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y
```
### 3.5.2 소프트맥스 함수 구현 시 주의점
- 소프트맥스 함수가 지수 함수이기 때문에 오버플로가 발생할 수 있음
- -> 소프트맥스 함수를 개선해 구현해야 함
