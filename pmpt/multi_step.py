#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import numpy
import pylab
from pmpt.decomposition import cp
from pmpt.hosvd import frobenius_norm, reconstruct, HOSVD
from pmpt.mobility import init_data, preprocess2, trans, is_stochastic
from pmpt.util import get_length_height, sparsity_tensor

# 1. 正确率 = 提取出的正确信息条数 /  提取出的信息条数
# 2. 召回率 = 提取出的正确信息条数 /  样本中的信息条数
# 两者取值在0和1之间，数值越接近1，查准率或查全率就越高。
# 3. F值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F 值即为正确率和召回率的调和平均值）


def recommend(A, recommends, unknow_poi_set, time_slice, top_k, order):
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

            availablity += 1
            start_set = set()

            for item in recom:
                if order == 2:
                    start_set.add(item[0])
                else:
                    start_set.add(item[1])

            check_list = {}
            for item in start_set:
                check_list[item] = set()

            for item in recom:
                if order == 2:
                    check_list[item[0]].add(item[1])
                else:
                    check_list[item[1]].add(item[2])

            s_precision = 0
            s_recall = 0
            s_f1_score = 0
            for item in start_set:
                recommend_set = set()
                pre_data = data[item]
                sort_data = []
                for index in range(0, len(pre_data)):
                    if index in unknow_poi_set[user][time]:
                        sort_data.append((index, pre_data[index]))
                sort_data.sort(key=lambda x: x[1], reverse=True)
                re_sort_data = sort_data[:top_k]
                for sort_item in re_sort_data:
                    recommend_set.add(sort_item[0])

                count_hit = len(recommend_set & check_list[item])

                if count_hit == 0:
                    precision = 0
                    recall = 0
                    f1_score = 0
                else:
                    precision = count_hit / len(recommend_set)
                    recall = count_hit / len(check_list[item])
                    f1_score = (2 * precision * recall) / (precision + recall)

                s_precision += precision
                s_recall += recall
                s_f1_score += f1_score

            # print "precision", s_precision / len(start_set)
            sum_precision += s_precision / len(start_set)
            sum_recall += s_recall / len(start_set)
            sum_f1_score += s_f1_score / len(start_set)

    return sum_precision / total, sum_recall / total, sum_f1_score / total, availablity / total


def n_step_tensor(tensor, n):
    # ten = numpy.array(tensor)
    # print len(tensor[0][0])
    if n == 1:
        return tensor
    num_user = len(tensor)
    num_time = len(tensor[0])
    num_poi = len(tensor[0][0])
    res = [[[[0 for i in range(num_poi)] for j in range(num_poi)] for l in range(num_time)] for m in range(num_user)]
    for user in range(num_user):
        for time in range(num_time):
            matrix = [[0 for i in range(num_poi)] for j in range(num_poi)]
            for index_i in range(num_poi):
                for index_j in range(num_poi):
                    # is_stochastic(tensor[user][time][index_i][index_j])
                    matrix[index_i][index_j] = tensor[user][time][index_i][index_j]
            p_tensor = numpy.matrix(matrix)
            # print "p_tensor: ", is_stochastic(p_tensor)
            if n == 2:
                p_tensor_res = numpy.dot(p_tensor, p_tensor)
            elif n == 3:
                p_tensor_res = numpy.dot(numpy.dot(p_tensor, p_tensor), p_tensor)
            elif n == 4:
                p_tensor_res = numpy.dot(numpy.dot(numpy.dot(p_tensor, p_tensor), p_tensor), p_tensor)
            elif n == 5:
                p_tensor_res = numpy.dot(numpy.dot(numpy.dot(numpy.dot(p_tensor, p_tensor), p_tensor), p_tensor), p_tensor)
            # print "p_tensor_res: ", p_tensor_res
            res[user][time] = numpy.ndarray.tolist(p_tensor_res)
    # print "res: ", res
    return res


# 三个方向优化：
# 1.计算方法的优化： 设计了基于poi集合的转移概率计算
# 2.训练数据的优化： 针对用户访问不同poi数目很少或者某poi被不同的用户访问很少
# 3.hosvd计算的优化： 降低阀值和迭代次数

if __name__ == '__main__':
    time_slice = 2
    train = 0.6
    # beijing = (39.433333, 41.05, 115.416667, 117.5)
    # haidian = (39.883333, 40.15, 116.05, 116.383333)
    # region = (39.88, 40.03, 116.05, 116.25)
    # region = (39.88, 40.05, 116.05, 116.26)
    region = (39.88, 40.05, 116.05, 116.26)
    cluster_radius = 1
    filter_count = 30
    order = 2
    top_k = 1
    step = 1
    length, height, top_left = get_length_height(region)
    print "区域（长度，宽度）：", length, height

    temp_data, time_slice, train, cluster_radius = init_data(time_slice, train, region, cluster_radius, filter_count)

    axis_pois, axis_users, train_structure_data, poi_adjacent_list, recommends, unknow_poi_set = preprocess2(temp_data, time_slice, train, cluster_radius, order)
    print "train_structure_data: ", train_structure_data
    print "poi_adjacent_list: ", poi_adjacent_list
    print "recommends: ", recommends

    tensor = trans(train_structure_data, poi_adjacent_list, order, len(axis_pois), len(axis_users), time_slice)
    # print "transition tensor: ", tensor

    # print "tensor: ", tensor, numpy.array(tensor).shape
    # print "tensor: ", tensor[0][0]
    # print "sparsity: ", sparsity_tensor(tensor[0][0])
    # print "tensor.shape: ", numpy.array(tensor[0][1]).shape
    #
    #
    # print cp(numpy.array(tensor[0][0]), n_components=1)

    # a = numpy.matrix([[0.5, 0.5], [0.2, 0.8]])
    # print is_stochastic(numpy.dot(a, a))

    # print "n_step_tensor3: ", n_step_tensor(tensor, 3)
    # print "n_step_tensor4: ", n_step_tensor(tensor, 4)
    # print "n_step_tensor2: ", n_step_tensor(tensor, 2)
    #
    # U, S, D = HOSVD(numpy.array(tensor), 0.7)
    #
    # A = reconstruct(S, U)
    # print "reconstruct tensor: ", A
    # print frobenius_norm(tensor-A)

    # time_slice_range = (1, 2, 4, 6, 12)
    # index = 0

    x_values = []
    y_values1 = []
    y_values2 = []
    y_values3 = []
    y_values4 = []
    while step <= 4:
        res_tensor = n_step_tensor(tensor, step)

        U, S, D = HOSVD(numpy.array(res_tensor), 0.7)

        A = reconstruct(S, U)

        avg_precision, avg_recall, avg_f1_score, availability = recommend(A, recommends[step], unknow_poi_set, time_slice, top_k, order)
        print "avg_precision: ", avg_precision
        print "avg_recall: ", avg_recall
        print "avg_f1_score: ", avg_f1_score
        print "availability: ", availability

        y_values1.append(avg_precision)
        y_values2.append(avg_recall)
        y_values3.append(avg_f1_score)
        y_values4.append(availability)
        x_values.append(step)
        step += 1

    pylab.plot(x_values, y_values1, 'rs', linewidth=1, linestyle="-", label=u"准确率")
    pylab.plot(x_values, y_values2, 'gs', linewidth=1, linestyle="-", label=u"召回率")
    pylab.plot(x_values, y_values3, 'bs', linewidth=1, linestyle="-", label=u"f1值")
    pylab.plot(x_values, y_values4, 'ks', linewidth=1, linestyle="-", label=u"可用率")
    pylab.xlabel(u"N步兴趣点推荐")
    pylab.ylabel(u"准确率-召回率")
    pylab.title(u"N步兴趣点推荐与准确率-召回率之间的关系")
    pylab.legend(loc='lower right')
    pylab.show()