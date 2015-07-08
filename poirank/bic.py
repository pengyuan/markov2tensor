#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from poirank.transition import init_data
import pylab
import numpy as np

'''
贝叶斯信息准则
在实际中不太好操作，对于状态数很大的马尔科夫链，极大似然值小而惩罚值过大，容易过拟合，从而阶数始终为1。
'''

poi_axis, axis_poi, data = init_data((6,))
#poi_axis, axis_poi, data = init_data(tuple(range(0, 182)))
data_length = len(data)
state_num = len(axis_poi)
print "key: ", axis_poi.keys()
print "state_num: ", state_num
# tensor = []
# tensor[1] = [[0 for i in range(state_num)] for j in range(state_num)]
# tensor[2] = [[[0 for i in range(state_num)] for j in range(state_num)] for k in range(state_num)]
# tensor[3] = [[[[0 for i in range(state_num)] for j in range(state_num)] for k in range(state_num)] for l in range(state_num)]
# tensor[4] = [[[[[0 for i in range(state_num)] for j in range(state_num)] for k in range(state_num)] for l in range(state_num)] for m in range(state_num)]

def bic_1(data):
    tensor1 = [0 for i in range(state_num)]
    tensor2 = [[0 for l in range(state_num)] for m in range(state_num)]
    for index in range(0, data_length-1):
        check_list = data[index:index+1]
        check_list_plus_one = data[index:index+2]
        # print check_list, check_list_plus_one
        tensor1[check_list[0]] += 1
        tensor2[check_list_plus_one[0]][check_list_plus_one[1]] += 1

    sum = 0.0
    for i in range(0, state_num):
        for j in range(0, state_num):
                if tensor1[i] != 0 and tensor2[i][j] != 0:
                    # print "a: ", tensor2[i][j][k]
                    # print "b: ", tensor1[i][j]
                    sum += tensor2[i][j] * np.log(tensor2[i][j]/tensor1[i])

    #print "sum: ", sum
    #return -2 * sum + 2*pow(state_num, 1) * (state_num - 1) #* np.log(data_length)
    return sum - pow(state_num, 1) #* np.log(data_length)


def bic_2(data):
    tensor1 = [[0 for i in range(state_num)] for j in range(state_num)]
    tensor2 = [[[0 for l in range(state_num)] for m in range(state_num)] for n in range(state_num)]
    for index in range(0, data_length-2):
        check_list = data[index:index+2]
        check_list_plus_one = data[index:index+3]
        # print check_list, check_list_plus_one
        tensor1[check_list[0]][check_list[1]] += 1
        tensor2[check_list_plus_one[0]][check_list_plus_one[1]][check_list_plus_one[2]] += 1

    sum = 0.0
    res = [[0 for o in range(state_num)] for p in range(state_num)]
    for i in range(0, state_num):
        for j in range(0, state_num):
            for k in range(0, state_num):
                if tensor1[i][j] != 0 and tensor2[i][j][k] != 0:
                    # print "a: ", tensor2[i][j][k]
                    # print "b: ", tensor1[i][j]
                    sum += tensor2[i][j][k] * np.log(tensor2[i][j][k]/tensor1[i][j])

    #print "sum: ", sum
    #return -2 * sum + 2*pow(state_num, 2) * (state_num - 1) #* np.log(data_length)
    return sum - pow(state_num, 2) #* np.log(data_length)


def bic_3(data):
    tensor2 = [[[0 for l in range(state_num)] for m in range(state_num)] for n in range(state_num)]
    tensor3 = [[[[0 for i in range(state_num)] for j in range(state_num)] for k in range(state_num)] for l in range(state_num)]
    for index in range(0, data_length-3):
        check_list = data[index:index+3]
        check_list_plus_one = data[index:index+4]
        # print check_list, check_list_plus_one
        tensor2[check_list_plus_one[0]][check_list_plus_one[1]][check_list_plus_one[2]] += 1
        tensor3[check_list_plus_one[0]][check_list_plus_one[1]][check_list_plus_one[2]][check_list_plus_one[3]] += 1
    sum = 0.0
    res = [[0 for o in range(state_num)] for p in range(state_num)]
    for i in range(0, state_num):
        for j in range(0, state_num):
            for k in range(0, state_num):
                for l in range(0, state_num):
                    if tensor2[i][j][k] != 0 and tensor3[i][j][k][l] != 0:
                        # print "a: ", tensor2[i][j][k]
                        # print "b: ", tensor1[i][j]
                        sum += tensor3[i][j][k][l] * np.log(tensor3[i][j][k][l]/tensor2[i][j][k])

    #print "mle: ", sum
    #return -2 * sum + 2*pow(state_num, 3) * (state_num - 1) #* np.log(data_length)
    return sum - pow(state_num, 3) #* np.log(data_length)


print "一阶bic值: ", bic_1(data)
print "二阶bic值: ", bic_2(data)
print "三阶bic值: ", bic_3(data)
#
#
#     esum = 0.0
#     for i in range(data_length):
#         esum = esum + data[i]
#     expectation = esum / data_length
#     print "expectation: ", expectation
#
#     vsum = 0.0
#     for j in range(data_length):
#         vsum += pow((data[j] - expectation), 2)
#     variance = vsum / data_length
#
#     tsum = 0.0
#     for k in range(data_length - order):
#         tsum += (data[k] - expectation) * (data[k + order] - expectation)
#     ar = tsum / (data_length) * variance
#
#     print "ar: ", ar
#     return ar
#
#
#
# order = 1
# y_values = []
# x_values = []
# while order <= 20:
#     ar = bic(order)
#     y_values.append(ar)
#     x_values.append(order)
#     order += 1
#
# pylab.plot(x_values, y_values, 'r.')
# pylab.show()