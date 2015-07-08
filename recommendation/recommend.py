#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from util.hosvd import HOSVD, unfold, reconstruct, frobenius_norm
from util.mobility import trans, init_data
import numpy as np
                    
# 第一步：构建转移概率tensor
# tensor = trans(count(init_db()))

# tensor = trans(count([0,1,0,1,0,1,1],2))
from util.tensor import hosvd
from util.tensor_old import hosvd2
from util.util import sparse

users = (1,)
train = 0.8
axis_poi, data_map, predicts, recommends = init_data(users, 0.8)
print "data_map: ", data_map
print "predicts: ", predicts
print "recommends: ", recommends
dimension = len(axis_poi)

for key in data_map:
    data = data_map[key]
    # print data
    # tensor为列表
    tensor = trans(data, dimension, 3)
    # print tensor



# afunc(tensor)
# 第二步：HOSVD，重构tensor

# threshold = 0.8

# 将列表转化为高维数组
tensor = np.array(tensor)

print "tensor:"
print tensor

# sparse(tensor)

threshold = 1.0
U, S, D = HOSVD(tensor, 0.8)

# new_T, T, Z, Un, Sn, Vn = hosvd(tensor)
# new_T2, Z2, Un2, Sn2, Vn2 = hosvd2(tensor)

print "the mode-1 unfold of core tensor:"
print unfold(S, 1)

print "The n-mode singular values:"
print D

A = reconstruct(S, U)
print "reconstruct tensor: ", A


print frobenius_norm(tensor-A)

# sparse(A)