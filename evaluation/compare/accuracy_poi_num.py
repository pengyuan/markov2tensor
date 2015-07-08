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
from factorized_markov_chain.mobility import init_data as init_data2, preprocess as preprocess2, trans as trans2
from tensor_factorization.mobility import init_data as init_data3, preprocess as preprocess3, trans as trans3
from factorized_markov_chain.build import recommend as recommend2
from tensor_factorization.build import recommend as recommend3
# 1. 正确率 = 提取出的正确信息条数 /  提取出的信息条数
# 2. 召回率 = 提取出的正确信息条数 /  样本中的信息条数
# 两者取值在0和1之间，数值越接近1，查准率或查全率就越高。
# 3. F值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F 值即为正确率和召回率的调和平均值）

# 与state of the art方法进行对比：
# 1.Factorized Markov Chain（FMC）：二阶马尔科夫链进行hosvd分解，没有用户之间和时间之间的信息协同
# 2.Tersor Factorization（TF）：基于用户-时间-地点的频数张量进行hosvd分解，没有用户的转移信息（地点对地点的评分）

if __name__ == '__main__':
    time_slice = 1
    train = 0.9
    # beijing = (39.433333, 41.05, 115.416667, 117.5)
    # haidian = (39.883333, 40.15, 116.05, 116.383333)
    # region = (39.88, 40.03, 116.05, 116.25)
    # region = (39.88, 40.05, 116.05, 116.26)
    region = (39.88, 40.05, 116.05, 116.26)
    cluster_radius = 0.5
    filter_count = 30
    order = 2
    top_k = 1

    length, height, top_left = get_length_height(region)
    print "区域（长度，宽度）：", length, height

    x_values = []
    y_values1 = []
    y_values2 = []
    y_values3 = []
    y_values4 = []
    last_count = 0
    while filter_count >= 10:
        start_time = datetime.datetime.now()
        # pmpt
        temp_data, time_slice, train, cluster_radius = init_data(time_slice, train, region, cluster_radius, filter_count)
        axis_pois, axis_users, train_structure_data, poi_adjacent_list, recommends, unknow_poi_set, poi_num = preprocess(temp_data, time_slice, train, cluster_radius, order, return_poi_num=True)
        print "poi数目: ", poi_num
        new_count = poi_num
        if new_count == last_count:
            filter_count -= 5
            continue
        last_count = new_count
        tensor = trans(train_structure_data, poi_adjacent_list, order, len(axis_pois), len(axis_users), time_slice)
        U, S, D = HOSVD(numpy.array(tensor), 0.7)
        A = reconstruct(S, U)
        avg_precision, avg_recall, avg_f1_score, availability = recommend(A, recommends, unknow_poi_set, time_slice, top_k, order)
        print "avg_precision(pmpt): ", avg_precision
        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds

        # fmc
        start_time2 = datetime.datetime.now()
        temp_data2, time_slice, train2 = init_data2(region, train, time_slice, filter_count)
        axis_pois2, axis_users2, train_structure_data2, recommends2, unknow_poi_set2 = preprocess2(temp_data2, time_slice, train2, order)
        A2 = trans2(train_structure_data2, order, len(axis_pois2), time_slice, 0.7)
        avg_precision2, avg_recall2, avg_f1_score2, availability2 = recommend2(A2, recommends2, unknow_poi_set2, time_slice, top_k, order)
        print "avg_precision(fmc): ", avg_precision2
        end_time2 = datetime.datetime.now()
        interval2 = (end_time2 - start_time2).seconds

        # tf
        start_time3 = datetime.datetime.now()
        temp_data3, time_slice, train3 = init_data3(time_slice, train, region, filter_count)
        axis_pois3, axis_users3, train_structure_data3, recommends3, unknow_poi_set3 = preprocess3(temp_data3, time_slice, train3, order)
        tensor3 = trans3(train_structure_data3, order, len(axis_pois3), len(axis_users3), time_slice)
        U3, S3, D3 = HOSVD(numpy.array(tensor3), 0.7)
        A3 = reconstruct(S3, U3)
        avg_precision3, avg_recall3, avg_f1_score3, availability3 = recommend3(A3, recommends3, unknow_poi_set3, time_slice, top_k, order)
        print "avg_precision(tf): ", avg_precision3
        end_time3 = datetime.datetime.now()
        interval3 = (end_time3 - start_time3).seconds

        print "interval ", interval
        print "interval2 ", interval2
        print "interval3 ", interval3
        y_values1.append(interval)
        y_values2.append(interval2)
        y_values3.append(interval3)
        # y_values4.append(availability)
        x_values.append(poi_num)
        filter_count -= 5

    pylab.plot(x_values, y_values1, 'rs', linewidth=1, linestyle="-", label=u"PMPT")
    pylab.plot(x_values, y_values2, 'gs', linewidth=1, linestyle="-", label=u"FMC")
    pylab.plot(x_values, y_values3, 'bs', linewidth=1, linestyle="-", label=u"TF")
    # pylab.plot(x_values, y_values4, 'ks', linewidth=1, linestyle="-", label=u"可用率")
    pylab.xlabel(u"poi维度")
    pylab.ylabel(u"准确率")
    pylab.title(u"poi维度与准确率之间的关系的几种方法对比")
    pylab.legend(loc='upper right')
    # pylab.xlim(1, 10)
    # pylab.ylim(0, 1.)
    pylab.show()