#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from operator import itemgetter
from tensorlib.tensor import tensor
import numpy as np
import ctypes as ct


def sparse(data):
    total = 1
    # print data.shape[0]
    shape = data.shape
    for i in range(len(shape)):
        total *= shape[i]

    # print total
    # print type(data), data
    ts = tensor(data, data.shape)

    # print 'a'
    ts2 = ts.tosptensor()

    print "非零元素个数: ", ts2.nnz()

    print "稀疏度: ", ts2.nnz()/total


class float_bits(ct.Structure):
    _fields_ = [('M', ct.c_uint, 23),
                ('E', ct.c_uint, 8),
                ('S', ct.c_uint, 1)
                ]

''' 针对IEEE754标准的32位浮点数的Python版的eps函数 '''
def epsf(innum):
    f1 = ct.c_float(innum)
    f2 = ct.c_float(innum)
    # p1 = (float_bits*)&f1;
    p1 = ct.cast(ct.byref(f1), ct.POINTER(float_bits))
    # p2 = (float_bits*)&f2;
    p2 = ct.cast(ct.byref(f2), ct.POINTER(float_bits))
    # p1->M = 1;
    p1.contents.M = 1
    # p2->M = 0;
    p2.contents.M = 0
    p1.contents.S = p2.contents.S = 0
    return f1.value - f2.value

# def get_top_n(A, user, slot, top_N, predicts, recommends):
#     data = A[user][slot]
#     sort_data = []
#     for item in range(0, len(data)):
#         meta_data = (item, data[item])
#         sort_data.append(meta_data)
#
#     sort_data.sort(key=lambda x: x[1], reverse=True)
#
#     # result = []
#     # for item in range(0, top_N):
#     #     result.append(sort_data[item][0])
#     #
#     # return result
#
#     return sort_data



if __name__ == '__main__':
    print 89/1764
    sparse(np.array([[[0,1],[1,2]],[[2,3],[3,4]]]))