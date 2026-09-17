import numpy as np

def add(num1, num2):
    sum = num1 + num2
    return sum

def inverse(arr):
    A = np.array(arr)
    return np.linalg.inv(A)

print(add(2, 3))
print(inverse([[1,2,3],[4,5,6],[7,8,10]]))