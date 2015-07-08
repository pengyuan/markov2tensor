#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import numpy as np
import pylab
from poirank import find_lcs, find_lcs_len
from poirank.locality import calculate_distance
from poirank.poi_rank import POIRank, rank
from poirank.transition import init_data, trans
from util import pykov


def printSerilize(foo):
     print "[",
     for f in foo:
          print "\'" + f + "\',",
     print "]",


def merge_result(result, distribution, axis_poi):
    res = []

    for key in distribution.keys():
        if result.has_key(key):
            data = result[key]
            block_size = len(data)
            total_size = len(axis_poi)
            for item in data:
                res.append((item[0], axis_poi[item[1]], item[2]*distribution[key]*(block_size/total_size)))

    return res


def similarity(total_list, hierarchical_list):
    print "hierarchical_list长度: ", len(hierarchical_list)
    print "total_list长度: ", len(total_list)
    a = []
    b = []
    pure_a = []
    pure_b = []
    hierarchical_ = {}
    for item in hierarchical_list:
        hierarchical_[item[1]] = item[2]
    for item in total_list:
        if item[1] in hierarchical_:
            a.append((item[1], item[2]))
            b.append((item[1], hierarchical_[item[1]]))
    # print len(b)
    b.sort(key=lambda x: x[1], reverse=True)
    for item in a:
        pure_a.append(item[0])
    for item in b:
        pure_b.append(item[0])
    # res = []
    # for item in b:
    #     res.append(item[0])
    # print "poi_order: ", printSerilize(a)
    # print "total_order: ", printSerilize(res)

    printSerilize(find_lcs(pure_a, pure_b))

    common_length = find_lcs_len(pure_a, pure_b)

    print "公共子序列长度：", common_length
    print "参与比较的序列长度：", len(a)

    return common_length / len(a)
    # index = 1
    # file_object = open('整体POI排序.txt', 'w')
    # for item in a:
    #     file_object.write(str(index)+" "+str(item[0])+" "+str(item[1])+"\n")
    #     index += 1
    # file_object.close()
    #
    # index2 = 1
    # file_object2 = open('分层POI排序.txt', 'w')
    # for item in b:
    #     file_object2.write(str(index2)+" "+str(item[0])+" "+str(item[1])+"\n")
    #     index2 += 1
    # file_object2.close()


def get_block_num(stay_point, left_top):
    length_distance = calculate_distance(left_top[1], stay_point[1], left_top[1], left_top[0])
    height_distance = calculate_distance(stay_point[2], left_top[0], left_top[1], left_top[0])
    # print "(length_distance, height_distance):", (length_distance, height_distance)
    i = (int)(height_distance // block_width)
    j = (int)(length_distance // block_width)
    # print "(i,j): ",(i,j)
    return block_matrix[i][j]


# def get_axis(block_num):
#     return block_num // block_width - 1, block_num % block_width - 1


if __name__ == '__main__':
    block_width = 1
    y_values = []
    x_values = []
    while(block_width < 3):
        block_num = int(30//block_width)
        block_matrix = [[i+1 for i in range(j*block_num, (j+1)*block_num)] for j in range(0, block_num)]
        block_transition = [[0 for i in range(0, block_num*block_num)] for j in range(0, block_num*block_num)]
        print block_matrix
        print block_transition

        block_array_list = {}
        for i in range(1, block_num*block_num+1):
            block_array_list[i] = []


        poi_axis, axis_poi, data = init_data(tuple(range(0, 182)), return_gps=True)   # (0, 182)
        max_latitude = 0
        min_latitude = data[0][2]
        max_longtitude = 0
        min_longtitude = data[0][1]

        for i in range(len(data)):
            if data[i][2] > max_latitude:
                max_latitude = data[i][2]
            if data[i][2] < min_latitude:
                min_latitude = data[i][2]
            if data[i][1] > max_longtitude:
                max_longtitude = data[i][1]
            if data[i][1] < min_longtitude:
                min_longtitude = data[i][1]

        # print "max_latitude: ", max_latitude
        # print "min_latitude: ", min_latitude
        # print "max_longtitude: ", max_longtitude
        # print "min_longtitude: ", min_longtitude

        # calculate_distance(lat1, lng1, lat2, lng2)
        length = calculate_distance(min_latitude, min_longtitude, min_latitude, max_longtitude)
        length2 = calculate_distance(max_latitude, min_longtitude, max_latitude, max_longtitude)
        height = calculate_distance(min_latitude, min_longtitude, max_latitude, min_longtitude)
        height2 = calculate_distance(min_latitude, max_longtitude, max_latitude, max_longtitude)
        # print "length: ", length, length2
        # print "height: ", height, height2

        # block为5km，则划分为30/5=6，6*6=36个block
        block_list = []
        for item in data:
            block_list.append(get_block_num(item, (min_longtitude, max_latitude)))

        print block_list
        for i in range(0, len(block_list)-1):
            current = block_list[i]
            next = block_list[i+1]
            if current == next:
                block_array_list[current].append(data[i])
                block_array_list[current].append(data[i+1])
            else:
                block_transition[current-1][next-1] += 1

        print "block_array_list: ", block_array_list

        for i in range(0, block_num*block_num):
            sum = 0
            for j in range(0, block_num*block_num):
                sum += block_transition[i][j]
            if sum != 0:
                for j in range(0, block_num*block_num):
                    block_transition[i][j] /= sum

        print "block_transition: ", block_transition

        # x, hist, flag, ihist = POIRank(np.array(block_transition), 0.95, v).solve()
        # print flag, x

        state_param = {}
        for i in range(0, block_num*block_num):
            for j in range(0, block_num*block_num):
                state_param[(i, j)] = block_transition[i][j]

        # d = {('R','R'):1./2, ('R','N'):1./4, ('R','S'):1./4,
        #      ('N','R'):1./2, ('N','N'):0., ('N','S'):1./2,
        #      ('S','R'):1./4, ('S','N'):1./4, ('S','S'):1./2}
        T = pykov.Chain(state_param)
        distribution = T.steady()
        print "block distribution: ", distribution
        sum = 0
        for item in distribution.keys():
            sum += distribution[item]

        print "distribution sum: ", sum

        result = {}
        for key in block_array_list.keys():
            data = block_array_list[key]
            # print "data: ", data
            if len(data) > 0:
                data_list = []
                for item in data:
                    data_list.append(item[0])

                data_set = set(data_list)
                data_set_list = []
                for item in data_set:
                    data_set_list.append(int(item))

                pois_axis2 = {}
                axis_pois2 = {}
                index = 0
                for item in data_set_list:
                    pois_axis2[item] = index
                    axis_pois2[index] = item
                    index += 1

                dimen = len(data_set)
                print "dimen: ", dimen
                data_input = [pois_axis2[item] for item in data_list]
                # print "data_input: ", data_input
                tensor = trans(data_input, dimen)
                alpha = 0.95
                v = []
                for i in range(0, dimen):
                    v.append(1/dimen)
                # print "v: ", v
                x, hist, flag, ihist = POIRank(np.array(tensor), alpha, v).solve()
                print "key: ", key
                res = rank(x, axis_pois2)
                print "res: ", res
                result[key] = res


        print "result: ", result
        res = merge_result(result, distribution, axis_poi)
        total_poi_list = sorted(res, key=lambda x: x[2], reverse=True)
        # print res.sort(key=lambda x: x[2], reverse=True)

        poi_axis, axis_poi, data = init_data(tuple((0, 2, 7, 19, 78, 101, 181)))   # (0, 182)

        tensor = trans(data, len(axis_poi))
        alpha = 0.95
        v = []
        for i in range(0, len(poi_axis)):
            v.append(1/len(poi_axis))
        x, hist, flag, ihist = POIRank(np.array(tensor), alpha, v).solve()
        # print "hist: ", hist
        # print "ihist: ", ihist
        # print "flag: ", flag
        if flag == 1:
            print "收敛，x: ", x
        else:
            print "无法收敛"

        poi_list = rank(x, axis_poi)
        simi = similarity(poi_list, total_poi_list)
        x_values.append(block_width)
        y_values.append(simi)
        block_width += 0.2

    pylab.plot(x_values, y_values, 'r.')
    pylab.show()




    # poi_axis, axis_poi, data = init_data(tuple(range(0, 10)))   # (0, 182)
    # tensor = trans(data, len(axis_poi))
    # alpha = 0.95
    # v = []
    # for i in range(0, len(poi_axis)):
    #     v.append(1/len(poi_axis))
    # x, hist, flag, ihist = BlockPOIRank(np.array(tensor), alpha, v).solve()
    # print "hist: ", hist
    # print "ihist: ", ihist
    # print "flag: ", flag
    # if flag == 1:
    #     print "收敛，x: ", x
    # else:
    #     print "无法收敛"
    #
    # result = rank(x, axis_poi)
    # sum = 0.0
    # for item in result:
    #     print item[0], item[1], item[2]
    #     sum += item[2]
    #
    # print "POIRank值的和： ", sum
