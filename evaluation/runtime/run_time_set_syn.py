#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import datetime
import numpy
import pylab
from pmpt.build import recommend
from pmpt.hosvd import frobenius_norm, reconstruct, HOSVD
from pmpt.mobility import init_data, preprocess, trans
from pmpt.util import get_length_height


if __name__ == '__main__':
    time_slice = 4
    train = 0.5
    # beijing = (39.433333, 41.05, 115.416667, 117.5)
    # haidian = (39.883333, 40.15, 116.05, 116.383333)
    # region = (39.88, 40.03, 116.05, 116.25)
    # region = (39.88, 40.05, 116.05, 116.26)
    region = (39.88, 40.05, 116.05, 116.26)
    cluster_radius = 0.1
    filter_count = 20   # 30
    order = 2
    top_k = 1

    length, height, top_left = get_length_height(region)
    print "区域（长度，宽度）：", length, height

    x_values = []
    y_values1 = []
    y_values2 = []
    y_values3 = []
    while cluster_radius <= 4:
        print "filter_count: ", filter_count
        start_time_total = datetime.datetime.now()
        temp_data, time_slice, train, cluster_radius = init_data(time_slice, train, region, cluster_radius, filter_count)

        axis_pois, axis_users, train_structure_data, poi_adjacent_list, recommends, unknow_poi_set, poi_num = preprocess(temp_data, time_slice, train, cluster_radius, order, return_poi_num=True)
        # print "train_structure_data: ", train_structure_data
        # print "poi_adjacent_list: ", poi_adjacent_list
        # print "recommends: ", recommends
        print "poi数目: ", poi_num

        tensor = trans(train_structure_data, poi_adjacent_list, order, len(axis_pois), len(axis_users), time_slice)
        # print "transition tensor: ", tensor

        start_time_svd = datetime.datetime.now()
        U, S, D = HOSVD(numpy.array(tensor), 0.7)
        end_time_svd = datetime.datetime.now()
        A = reconstruct(S, U)
        print "reconstruct tensor: ", A
        # print frobenius_norm(tensor-A)

        avg_precision, avg_recall, avg_f1_score, availability = recommend(A, recommends, unknow_poi_set, time_slice, top_k, order)
        print "avg_precision: ", avg_precision
        end_time_total = datetime.datetime.now()
        interval_total = (end_time_total - start_time_total).seconds
        interval_svd = (end_time_svd - start_time_svd).microseconds
        print "interval_svd: ", interval_svd
        # y_values1.append(interval_total)
        y_values2.append(interval_svd)
        # y_values3.append(avg_precision)
        x_values.append(cluster_radius)
        cluster_radius += 0.1

    # pylab.plot(x_values, y_values1, 'rs', linewidth=1, linestyle="-", label=u"总运行时间")
    pylab.plot(x_values, y_values2, 'gs', linewidth=1, linestyle="-", label=u"hosvd运行时间")
    # pylab.plot(x_values, y_values3, 'bs', linewidth=1, linestyle="-", label=u"准确率")
    pylab.xlabel(u"poi近邻半径（千米）")
    pylab.ylabel(u"hosvd运行时间（微秒）")
    pylab.title(u"poi近邻半径与运行时间的关系")
    pylab.legend(loc='upper right')
    # pylab.xlim(1, 10)
    # pylab.ylim(0, 1.)
    pylab.show()