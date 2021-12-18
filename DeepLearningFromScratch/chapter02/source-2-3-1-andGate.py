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