# Chapter 1. 헬로 파이썬
## 1.1 파이썬이란?
- 파이썬은 간단하고 배우기 쉬운 프로그래밍 언어
- 코드가 읽기 쉽고, 성능이 뛰어남 -> ML, Data Science에서 널리 쓰임
- 파이썬 자체 성능 + 수치 계산&통계 처리 라이브러리(ex. NumPy, SciPy, etc.)를 통해 Data Science에서 널리 사용
- 딥러닝 프레임워크(ex. Caffe, TensorFlow, Chainer, Theano, etc.)에서도 파이썬용 API 제공
## 1.2 파이썬 설치하기
### 1.2.1 파이썬 버전
- 파이썬 2.x와 3.x 버전이 존재
- 100% 호환되지 않음(하위 호환성이 없음)
### 1.2.2 사용하는 외부 라이브러리
- 효율적인 딥러닝을 구현하기 위해 사용
> NumPy
- 수치 계산용 라이브러리
- 고도의 수학 알고리즘과 행렬을 조작에 용이
> matplotlib
- 그래프를 그려주는 라이브러리
- 결과, 딥러닝 실행 과정의 중간 데이터를 시각화
 ### 1.2.3 아나콘다 배포판
 - 데이터 분석에 중점을 둔 배포판
 ## 1.3. 파이썬 인터프리터
 - `python --version`: Python 3.7.6
 - `python`으로  파이썬 인터프리터 수행
 - 파이썬 인터프리터를 통해 대화식으로 프로그래밍할 수 있음
### 1.3.1 Arithmetic Operations
```
 >>> 7 / 5
 1.4
 >>> 3 ** 2
 9
```
**/**: 나눗셈, 파이썬3에서 (정수)÷(정수)=(실수)
**\*\***: 거듭제곱 (ex. 3 ** 2 = 3 * 3 = 9)
### 1.3.2 자료형
- `type()` 함수로 데이터의 자료형을 알 수 있음
```
>>> type(10)
<class 'int'>
```
- `<class 'int'>`처럼 자료형과 class가 같은 의미로 사용되기도 함
### 1.3.3 Variable
- 변수를 정의할 수 있고, 변수를 사용해 계산하거나 다른 값을 대입할 수 있음
```
>>> x = 10    # 초기화
>>> print(x)  # x값 출력
10
>>> x = 100   # x에 100 대입해 저장
>>> print(x)
100
>>> y = 3.14
>>> x * y
314.0
>>> type(x * y)
<class 'float'>
```
- 파이썬은 **동적 언어**: 변수의 자료형을 상황에 맞게 자동으로 결정
	- `x`를 10으로 초기화할 때, `x`의 형태가 int임을 자동으로 판단
	- `x`가 100이고 `y`가 3.14일 때, 	`x * y`는 (정수)*(실수)로 (실수): **자동 형변환**
- `#`은 주석의 시작
### 1.3.4 List
```
>>> a = [1, 2, 3, 4, 5]  # 리스트 생성
>>> print(a)   # 리스트 내용 출력
[1, 2, 3, 4, 5]
>>> len(a)     # 리스트 길이 출력
5
>>> a[0]       # 리스트 첫 번째 원소
1
>>> a[4]       # 리스트 5번째 원소(길이가 5이므로 마지막 원소)
5
>>> a[4] = 99  # 리스트 5번째 원소에 99 대입
>>> print(a)
[1, 2, 3, 4, 99]
>>> a[0:2]     # 인덱스 0~(2-1)까지 원소(2는 포함X)
[1, 2]
>>> a[1:]      # 인덱스 1~끝까지 원소
[2, 3, 4, 99]
>>> a[:3]      # 인덱스 처음~(3-1)까지 원소
[1, 2, 3]
>>> a[:-1]     # 인덱스 처음~(끝-1)까지 원소
[1, 2, 3, 4]
>>> a[:-2]     # 인덱스 처음~(끝-2)까지 원소
[1, 2, 3]
```
- **a[인덱스]**로 리스트 원소에 접근
- 인덱스는 0부터 시작
- **slicing**: 범위를 지정해 원하는 부분 리스트 얻을 수 있음
### 1.3.5 Dictionary
- **key**와 **value**를 한 쌍으로 저장
```
>>> me = {'height':180}  # 딕셔너리 생성
>>> me['height']         # 원소에 접근
180
>>> me['weight'] = 70    # 새 원소 추가
>>> print(me)
{'weight': 70, 'height': 180}
```
### 1.3.6 Bool
- `bool`: True/False 중에 값을 가짐
- **and, or, not** 연산자를 사용할 수 있음
```
>>> hungry = True
>>> sleepy = False
>>> type(hungry)
<class 'bool'>
>>> not hungry
False
>>> hungry and sleepy
False
>>> hungry or sleepy
True
```
### 1.3.7 If-clause
- 조건에 따른 처리방식으로 if/else 사용
```
>>> hungry = True
>>> if hungry:
...     print("I'm hungry")
...
I'm hungry
>>> hungry = False
>>> if hungry:
...     print("I'm hungry")
... else:
...     print("I'm not hungry")
...     print("I'm sleepy")
...
I'm not hungry
I'm sleepy
```
- **파이썬에서는 들여쓰기 중요**
### 1.3.8 For-loop
- 반복(루프) 처리에 for문 사용
```
>>> for i in [1, 2, 3]:
...      print(i)
...
1
2
3
```
- `for ... in ...:` 구문을 통해 데이터 set의 각 원소에 접근할 수 있음
### 1.3.9 Function
- **function**: 특정 기능을 수행하는 명령의 집합
```
>>> def hello():
...     print("Hello World!")
...
>>> hello()
Hello World!
```
- 함수는 인수를 가질 수 있음
- **+**를 이용해 문자열을 이어붙일 수 있음
```
>>> def hello(object):
...     print("Hello " + object + "!")
...
>>> hello("cat")
Hello cat!
```
## 1.4. 파이썬 스크립트 파일
- 인터프리터는 간단한 수행에 적절
- 스크립트는 파일을 저장하고, 실행하는 방법
### 1.4.1 파일로 저장하기
```
**hungry.py**
print("I'm hungry!")
```
```
$ python hungry.py
I'm hungry
```
### 1.4.2 Class
- class를 정의해 독자적인 자료형을 만들 수 있음
- class만의 method와 속성을 정의할 수 있음
```
class ClassName:
    def __init__(self, 인수, ...):
        ...
    def method1(self, 인수, ...):
        ...
    def method2(self, 인수, ...):
        ...
```
- `__init__`
	- constructor
	- 클래스를 초기화하는 방법 정의
	- 클래스 인스턴스가 만들어질 때 한 번만 불림
- method의 **첫 번째 인수로 self**를 명시적으로 표기
```
**man.py**
class Man:  
    def __init__(self, name):  # Constructor
        self.name = name  
        print("Initialized!")  
  
    def hello(self):           # Method #1
        print("Hello " + self.name + "!")  
  
    def goodbye(self):         # Method #2
        print("Good-bye " + self.name + "!")  
  
m = Man("David")  
m.hello()  
m.goodbye()
```
```
$ python man.py
Initialized!
Hello David!
Good-bye David!
```
- `Man`이라는 class 정의
- `Man`에서 `m`이라는 인스턴스(객체) 생성
- `Man의 constructor`는 `name` 인수를 받고, 인스턴스 변수인 `self.name` 초기화
## 1.5 NumPy
### 1.5.1 NumPy 가져오기
- NumPy는 외부 라이브러리이므로, `import` 필요
`>>> import numpy as np`: numpy를 np라는 이름으로 가져오기
### 1.5.2 NumPy 배열 생성
- `np.array()`로 NumPy 배열 생성
- input: List
- output: numpy.ndarray
```
>>> x = np.array([1.0, 2.0, 3.0])
>>> print(x)
[1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'>
```
### 1.5.3 NumPy의 산술 연산
```
>>> x = np.array([1.0, 2.0, 3.0])
>>> y = np.array([2.0, 4.0, 6.0])
>>> x + y  # element-wise sum
array([3., 6., 9.])
>>> x - y
array([-1., -2., -3.])
>>> x * y  # element-wise product
array([ 2.,  8., 18.])
>>> x / y
array([0.5, 0.5, 0.5])
```
- **배열 x와 y의 원소 수가 같음**
- **broadcast**: NumPy 배열-스칼라값의 산술 연산
```
>>> x = np.array([1.0, 2.0, 3.0])
>>> x / 2.0
array([0.5, 1. , 1.5])
```
### 1.5.4 NumPy의 N차원 배열
```
>>> A = np.array([[1, 2], [3, 4]])
>>> print(A)
[[1 2]
 [3 4]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int64')
```
- `shape`: 행렬의 각 차원의 크기
- `dtype`: 행렬에 담긴 원소의 자료형
```
>>> B = np.array([[3, 0], [0, 6]])
>>> A + B
array([[ 4,  2],
       [ 3, 10]])
>>> A * B
array([[ 3,  0],
       [ 0, 24]])
```
```
>>> print(A)
[[1 2]
 [3 4]]
>>> A * 10
array([[10, 20],
       [30, 40]])
```
### 1.5.5 Broadcast
> The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.
[Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
```
>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([10, 20])
>>> A * B
array([[10, 40],
       [30, 80]])
```
### 1.5.6 원소 접근
```
>>> X = np.array([[51, 55], [14, 19], [0, 4]])
>>> print(X)
[[51 55]
 [14 19]
 [ 0  4]]
>>> X[0]     # 0행 원소
array([51, 55])
>>> X[0][1]  # (0, 1) 위치의 원소
55
>>> for row in X:
...     print(row)
...
[51 55]
[14 19]
[0 4]
```
```
>>> X = X.flatten()
>>> print(X)
[51 55 14 19  0  4]
>>> X[np.array([0, 2, 4])]  # 인덱스가 0, 2, 4인 원소
array([51, 14,  0])
```
> `ndarray.flatten(order='C')`: Return a copy of the array collapsed into one dimension.
[numpy.ndarray.flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

```
>>> X > 15
array([ True,  True, False,  True, False, False])
>>> X[X>15]
array([51, 55, 19])
```
- `X > 15`의 결과는 bool 배열이 생성되어 출력
