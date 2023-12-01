# coding:utf8
import numpy as np

def res():
    for i in rande()
def ravno(result1,result2,result3):
    c=0
    if result1.all()==result2.all()==result.all():
        c=1
    return c
def matrixmult (array1, array2):
    C = [[0 for row in range(len(arary1))] for col in range(len(arary2[0]))]
    for i in range(len(array1)):
        for j in range(len(array2[0])):
            for k in range(len(array2)):
                C[i][j] += array1[i][k]*array2[k][j]
    return C
if __name__=='__main__':
    array1=[[2,-3],[4,5]]
    array2=[[1,-3,-2],[2,4,-1]]
    row1 = len(array1[0])
    col1 = len(array1)
    row2 = len(array2)
    col2 = len(array2[0])
    result1=np.matmul(array1,array2)
    result2=np.dot(array1,array2)
    result3=matrixmult(array1,arary2)
    print(ravno(result1,result2,result3))

