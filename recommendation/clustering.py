#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import numpy as np
from numpy.linalg import inv
from util import pykov
from util.mobility import trans
from util.preference import init_data, param, vec


def city_computing():
    users = range(182)
    axis_prefs, datas = init_data(users)
    dimen = len(axis_prefs)

    # print dimen
    # print axis_prefs
    # print datas
    print datas.has_key(50)
    for key in datas:
        # print key
        data = datas[key]
        if len(data) == 1:
            continue
        tensor = trans(data, dimen, 2)
        # print tensor

        para = param(tensor)
        T = pykov.Chain(para)
        balance = T.steady()
        # print balance
        # print balance.sort(False)
        # print balance.sort(True)

        res = vec(balance.sort(True), dimen)
        print 'user'+str(key)+": "+str(res)


def get_reverse_poll(poll):
    shape = poll.shape
    reverse_poll = np.zeros((shape[1], shape[0]))

    for i in range(0, shape[1]):
        sum = 0
        for j in range(0, shape[0]):
            sum += poll[j][i]
        for j in range(0, shape[0]):
            reverse_poll[i][j] = poll[j][i] / sum

    return reverse_poll


if __name__ == '__main__':
    poll = [[1/2, 1/2], [1/3, 2/3], [1/4, 3/4]]
    matrix = np.array(poll)
    print "matrix: ", matrix
    rev_poll = get_reverse_poll(matrix)
    print "reverse matrix: ", rev_poll
    poi_list = [0, 0]
    user_list = [1, 1, 1]
    poi_length = len(poi_list)
    user_length = len(user_list)
    tol = 1e-8
    niter = 1e2

    for i in range(1, int(niter)):
        for poi in range(0, poi_length):
            poi_list[poi] = 0

        for poi in range(0, poi_length):
            for user in range(0, user_length):
                poi_list[poi] += user_list[user] * poll[user][poi]
        print "poi_list: ", poi_list

        for user in range(0, user_length):
            user_list[user] = 0
        for user in range(0, user_length):
            for poi in range(0, poi_length):
                # user_list[user] += poi_list[poi] * rev_poll[poi][user]
                user_list[user] += poi_list[poi] / 3

        print "user_list: ", user_list

    print "poi_list: ", poi_list
    print "user_list: ", user_list


    print 1.0833333333333333 * 0.46153846 + 1.9166666666666665 * 0.26086957


    # if i == niter and curres > tol:
    #     print 'failure: did not converge after %i iterations to %e tolerance', niter, tol
    #     raise ValueError
    #     flag = 0
    # else:
    #     flag = 1