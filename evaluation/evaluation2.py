#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from matplotlib.pyplot import gca
import numpy
import pylab
from pmpt.hosvd import frobenius_norm, reconstruct, HOSVD
from pmpt.mobility import init_data, preprocess, trans
from pmpt.util import get_length_height

# 1. 正确率 = 提取出的正确信息条数 /  提取出的信息条数
# 2. 召回率 = 提取出的正确信息条数 /  样本中的信息条数
# 两者取值在0和1之间，数值越接近1，查准率或查全率就越高。
# 3. F值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F 值即为正确率和召回率的调和平均值）


def recommend(A, recommends, time_slice, top_k, order):
    user_num = len(A)
    total = user_num * time_slice
    availablity = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0

    for user in range(user_num):
        for time in range(time_slice):
            data = A[user][time]
            recom = recommends[user][time]
            if recom is None or len(recom) == 0:
                continue

            check_set = set()
            recom_set = set()
            recom_list = []
            for item in recom:
                if order == 2:
                    check_set.add(item[1])
                else:
                    check_set.add(item[2])

            availablity += 1
            for item in recom:
                if order == 2:
                    sort_data = []
                    pre_data = data[item[0]]
                    for index in range(0, len(pre_data)):
                        if index in check_set:
                            sort_data.append((index, pre_data[index]))
                    sort_data.sort(key=lambda x: x[1], reverse=True)
                    re_sort_data = sort_data[:top_k]
                    for item in re_sort_data:
                        recom_list.append(item)

                else:
                    sort_data = []
                    pre_data = data[item[0]][item[1]]
                    for index in range(0, len(pre_data)):
                        if index in check_set:
                            sort_data.append((index, pre_data[index]))
                    sort_data.sort(key=lambda x: x[1], reverse=True)
                    re_sort_data = sort_data[:top_k]
                    for item in re_sort_data:
                        recom_list.add(item)

            recom_list.sort(key=lambda x: x[1], reverse=True)
            i = 0
            while len(recom_set) < top_k:
                recom_set.add(recom_list[i][0])
                i += 1

            count_hit = len(recom_set & check_set)
            precision = count_hit / len(recom_set)
            print "precision: ", precision
            recall = count_hit / len(check_set)
            sum_precision += precision
            sum_recall += recall
            f1_score = (2 * precision * recall) / (precision + recall)
            sum_f1_score += f1_score

    return sum_precision / total, sum_recall / total, sum_f1_score / total, availablity / total

# 三个方向优化：
# 1.计算方法的优化： 设计了基于poi集合的转移概率计算
# 2.训练数据的优化： 针对用户访问不同poi数目很少或者某poi被不同的用户访问很少
# 3.hosvd计算的优化： 降低阀值和迭代次数

if __name__ == '__main__':
    time_slice = 1
    train = 0.8
    # beijing = (39.433333, 41.05, 115.416667, 117.5)
    # haidian = (39.883333, 40.15, 116.05, 116.383333)
    # region = (39.88, 40.03, 116.05, 116.25)
    # region = (39.88, 40.05, 116.05, 116.26)
    region = (39.88, 40.05, 116.05, 116.26)
    cluster_radius = 1
    filter_count = 30
    order = 2
    top_k = 1

    length, height, top_left = get_length_height(region)
    print "区域（长度，宽度）：", length, height


    time_slice_range = (1, 2, 4, 6, 12)
    index = 0
    # avg_precision, avg_recall, avg_f1_score, availability = recommend(A, recommends, time_slice, top_k, order)
    # print "avg_precision: ", avg_precision
    # print "avg_recall: ", avg_recall
    # print "avg_f1_score: ", avg_f1_score
    # print "availability: ", availability

    x_values = []
    y_values1 = []
    y_values2 = []
    y_values3 = []
    y_values4 = []
    for time_slice in time_slice_range:
        temp_data, time_slice, train, cluster_radius = init_data(time_slice, train, region, cluster_radius, filter_count)

        axis_pois, axis_users, train_structure_data, poi_adjacent_list, recommends = preprocess(temp_data, time_slice, train, cluster_radius, order)
        print "train_structure_data: ", train_structure_data
        print "poi_adjacent_list: ", poi_adjacent_list
        print "recommends: ", recommends

        tensor = trans(train_structure_data, poi_adjacent_list, order, len(axis_pois), len(axis_users), time_slice)
        # print "transition tensor: ", tensor


        U, S, D = HOSVD(numpy.array(tensor), 0.7)

        A = reconstruct(S, U)
        print "reconstruct tensor: ", A
        print frobenius_norm(tensor-A)
        avg_precision, avg_recall, avg_f1_score, availability = recommend(A, recommends, time_slice, top_k, order)
        print "avg_precision: ", avg_precision
        print "avg_recall: ", avg_recall
        print "avg_f1_score: ", avg_f1_score
        print "availability: ", availability

        y_values1.append(avg_precision)
        y_values2.append(avg_recall)
        y_values3.append(avg_f1_score)
        y_values4.append(availability)
        x_values.append(time_slice)
        # time_slice += 1

    pylab.plot(x_values, y_values1, 'r',  linewidth=1, linestyle="-", label=u"精确率")
    pylab.plot(x_values, y_values2, 'g',  linewidth=1, linestyle="-", label=u"召回率")
    pylab.plot(x_values, y_values3, 'b',  linewidth=1, linestyle="-", label=u"f1值")
    pylab.plot(x_values, y_values4, 'k',  linewidth=1, linestyle="-", label=u"可用率")
    pylab.xlabel(u"训练集占比")
    pylab.ylabel(u"精确率-召回率")
    pylab.title(u"训练集占比与精确率-召回率之间的关系")
    pylab.legend(loc='upper right')
    # pylab.xlim(0.0, )
    pylab.show()