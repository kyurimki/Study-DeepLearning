# Chapter 2. Perceptron

- Frank Rosenblatt가 1957년 고안한 알고리즘

- 신경망(딥러닝)의 기원이 되는 알고리즘

## 2.1 퍼셉트론이란?

- 다수의 신호를 입력으로 받아 하나의 신호 출력

- 신호로 흐름을 만들고 정보 전달

- 신호는 흐른다/안 흐른다(1/0)만 존재:

- 1 = 신호가 흐른다, 0 = 신호가 흐르지 않는다

- ex. 입력이 2개인 퍼셉트론
![입력이 2개인 퍼셉트론](https://user-images.githubusercontent.com/61455647/113532469-61389380-9606-11eb-92d7-769cf65a8b07.png)
	- x1, x2: 입력신호
	- y: 출력신호
	- w1, w2: 가중치
	- x1, x2, y: 뉴런/노드
	- 입력 신호가 뉴런에 보내질 때 고유한 가중치가 곱해진다.(x1*w1, x2*w2)
	- 뉴런에서 보내온 신호의 총합이 일정 값(임계값)을 넘을 때만 1을 출력 -> θ = 임계값
	- 수식으로 나타내면 다음과 같다
	![입력이 2개인 퍼셉트론 동작 원리 수식](https://user-images.githubusercontent.com/61455647/113532824-49adda80-9607-11eb-9a11-b9344c678e8e.png)
- 복수의 입력 신호에 고유한 가중치를 부여함
- 가중치는 신호가 결과에 영향을 주는 정도를 조절함 -> 가중치가 클수록 신호의 중요성 ↑
## 2.2 단순한 논리 회로
### 2.2.1 AND 게이트
- 입력 2개, 출력 1개
- 진리표: 입력 신호와 출력 신호의 대응 표
- 두 입력이 모두 1일 때만 1을 출력하고, 그 외에는 0을 출력
    |x1|x2|y|
    |--|--|--|
    |0|0|0|
    |1|0|0|
    |0|1|0|
    |1|1|1|

- 퍼셉트론으로 표현하기(w1, w2, θ):
	- 무수히 많음
	- ex. (0.5, 0.5, 0.7), (0.5, 0.5, 0.8), (1.0, 1.0, 1.0), ...
### 2.2.2 NAND 게이트와 OR 게이트
- NAND 게이트
	- Not AND
	- AND 게이트의 출력을 뒤집은 것
	- x1, x2가 모두 1일 때만 0을 출력하고, 그 외에는 1을 출력
	- AND 게이트를 구현하는 매개변수의 부호를 모두 반전
        |x1|x2|y|
        |--|--|--|
        |0|0|1|
        |1|0|1|
        |0|1|1|
        |1|1|0|

- OR 게이트: 입력 신호 중 하나 이상이 1이면 1이 출력
    |x1|x2|y|
    |--|--|--|
    |0|0|0|
    |1|0|1|  
    |0|1|1|
    |1|1|1|
=> **퍼셉트론의 구조는 AND, NAND, OR 게이트 모두 똑같음. 다른 것은 매개변수(가중치와 임계값)**
## 2.3 퍼셉트론 구현하기
```
**andGate.py**
def AND(x1, x2):  
    w1, w2, theta = 0.5, 0.5, 0.7  
  tmp = x1*w1 + x2*w2  
    if tmp <= theta:  
        return 0  
  else:  
        return 1  
  
  
def printResult(x1, x2, y):  
    print("AND("+str(x1)+", "+str(x2)+") = "+str(y))  
  
  
printResult(0, 0, AND(0, 0))  
printResult(1, 0, AND(1, 0))  
printResult(0, 1, AND(0, 1))  
printResult(1, 1, AND(1, 1))
```
```
**RESULT**
AND(0, 0) = 0
AND(1, 0) = 0
AND(0, 1) = 0
AND(1, 1) = 1
```
### 2.3.2 가중치와 편향 도입
- θ를 *-b*로 치환하면 다음과 같이 나타낼 수 있다.
![image](https://user-images.githubusercontent.com/61455647/113534842-c7c0b000-960c-11eb-851c-af88e2e79a5b.png)
- 이때 *b*를 **편향 bias**라고 한다.
- => 퍼셉트론은 ((입력신호)*(가중치)의 합)+(편향)이 0보다 크면 1을 출력, 0보다 작으면 0을 출력한다.
```
>>> import numpy as np
>>> x = np.array([0, 1])  # 입력
>>> w = np.array([0.5, 0.5])  # 가중치
>>> b = -0.7  # 편향
>>> w * x
array([0. , 0.5])
>>> np.sum(w*x)
0.5
>>> np.sum(w*x) + b
-0.19999999999999996  # 대략 -0.2(연산 오차)
```
### 2.2.3 가중치와 편향 구하기
```
**gatesWithBias.py**
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def printResult(x1, x2, y1, y2, y3):
    print("AND("+str(x1)+", "+str(x2)+") = "+str(AND(x1, x2))+ ", NAND("+str(x1)+", "+str(x2)+") = "+str(NAND(x1, x2))+", OR("+str(x1)+", "+str(x2)+") = "+str(OR(x1, x2)))

printResult(0, 0, AND(0, 0), NAND(0, 0), OR(0, 0))
printResult(1, 0, AND(1, 0), NAND(1, 0), OR(0, 0))
printResult(0, 1, AND(0, 1), NAND(0, 1), OR(0, 1))
printResult(1, 1, AND(1, 1), NAND(1, 1), OR(1, 1))
```
```
**RESULT**
AND(0, 0) = 0, NAND(0, 0) = 1, OR(0, 0) = 0
AND(1, 0) = 0, NAND(1, 0) = 1, OR(1, 0) = 1
AND(0, 1) = 0, NAND(0, 1) = 1, OR(0, 1) = 1
AND(1, 1) = 1, NAND(1, 1) = 0, OR(1, 1) = 1
```
