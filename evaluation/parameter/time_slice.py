#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import numpy
import pylab
from pmpt.build import recommend
from pmpt.hosvd import frobenius_norm, reconstruct, HOSVD
from pmpt.mobility import init_data, preprocess, trans
from pmpt.util import get_length_height

# 1. 正确率 = 提取出的正确信息条数 /  提取出的信息条数
# 2. 召回率 = 提取出的正确信息条数 /  样本中的信息条数
# 两者取值在0和1之间，数值越接近1，查准率或查全率就越高。
# 3. F值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F 值即为正确率和召回率的调和平均值）

if __name__ == '__main__':
    # time_slice = 1
    train = 0.5
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

    x_values = []
    y_values1 = []
    y_values2 = []
    y_values3 = []
    y_values4 = []
    while index < len(time_slice_range):
        time_slice = time_slice_range[index]
        temp_data, time_slice, train, cluster_radius = init_data(time_slice, train, region, cluster_radius, filter_count)

        axis_pois, axis_users, train_structure_data, poi_adjacent_list, recommends, unknow_poi_set = preprocess(temp_data, time_slice, train, cluster_radius, order)
        print "train_structure_data: ", train_structure_data
        print "poi_adjacent_list: ", poi_adjacent_list
        print "recommends: ", recommends

        tensor = trans(train_structure_data, poi_adjacent_list, order, len(axis_pois), len(axis_users), time_slice)
        # print "transition tensor: ", tensor


        U, S, D = HOSVD(numpy.array(tensor), 0.7)

        A = reconstruct(S, U)
        print "reconstruct tensor: ", A
        print frobenius_norm(tensor-A)

        avg_precision, avg_recall, avg_f1_score, availability = recommend(A, recommends, unknow_poi_set, time_slice, top_k, order)
        print "avg_precision: ", avg_precision
        print "avg_recall: ", avg_recall
        print "avg_f1_score: ", avg_f1_score
        print "availability: ", availability

        y_values1.append(avg_precision)
        y_values2.append(avg_recall)
        y_values3.append(avg_f1_score)
        y_values4.append(availability)
        x_values.append(time_slice)
        index += 1

    pylab.plot(x_values, y_values1, 'rs', linewidth=1, linestyle="-", label=u"准确率")
    pylab.plot(x_values, y_values2, 'gs', linewidth=1, linestyle="-", label=u"召回率")
    pylab.plot(x_values, y_values3, 'bs', linewidth=1, linestyle="-", label=u"f1值")
    pylab.plot(x_values, y_values4, 'ks', linewidth=1, linestyle="-", label=u"可用率")
    pylab.xlabel(u"时间维度（一天均分为x片）")
    pylab.ylabel(u"准确率-召回率")
    pylab.title(u"时间维度与准确率-召回率之间的关系")
    pylab.legend(loc='upper right')
    # pylab.xlim(1, 10)
    # pylab.ylim(0, 1.)
    pylab.show()