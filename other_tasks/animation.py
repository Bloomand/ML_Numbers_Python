import numpy as np
import string
import argparse
def raw():
    return np.arange(35).reshape(5,7)
def zmeyka():
    return [[ i + 1 + 7 * j for i in range(7)][::pow(-1, j)] for j in range(5)]
def col():
    return np.arange(35).reshape(5,7).T

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, nargs='?', const='raw', help='vvedite vid matrix')
    args = parser.parse_args()
    a = args.name
    if a=='raw':
        print(raw())
    elif a=='zmeyka':
        print(zmeyka())
    else:
        print(col())

