Chapter 3. 신경망(Neural Network)
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
### 3.2.2 계단 함수 구현하기
```
>>> import numpy as np

>>> x = np.array([-1.0, 1.0, 2.0])
>>> x
array([-1.,  1.,  2.])

>>> y = x > 0
>>> y
array([False,  True,  True])
```
- 넘파이 배열에 부등호 연산 수행 -> 배열의 원소 각각에 부등호 연산 수행한 bool 배열 생성
- y는 bool 배열, 원하는 결과는 int형이므로, y의 원소를 bool에서 int로 바꿔준다.
```
>>> y = y.astype(np.int)
>>> y
array([0, 1, 1])
```
- 넘파이 배열의 자료형을 변환할 때 `astype()` 이용
