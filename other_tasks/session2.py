#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math

def cylinder(radius,height):
   vol=(math.pi)*(radius **2)*(height)
   return vol


if __name__ == '__main__':
    print(cylinder(2,4))