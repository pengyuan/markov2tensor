#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from util import pykov
import MySQLdb
import numpy as np
from poirank.poi_rank import POIRank
from poirank.preference import init_poi_type_data, param, vec
from poirank.transition import init_data, trans
from util.mobility import trans as trans2
from preprocess import settings
from numpy import *


# 计算两个经纬度之间的距离，单位千米
def calculate_distance(lat1, lng1, lat2, lng2):
    earth_radius = 6378.137
    rad_lat1 = rad(lat1)
    rad_lat2 = rad(lat2)
    a = rad_lat1 - rad_lat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(rad_lat1) * math.cos(rad_lat2) * math.pow(math.sin(b / 2), 2)))
    s *= earth_radius
    if s < 0:
        return round(-s, 2)
    else:
        return round(s, 2)


def rad(flo):
    return flo * math.pi / 180.0


# def init_data(users):
#     conn = MySQLdb.connect(host=settings.HOST, user=settings.USER, passwd=settings.PASSWORD, db=settings.DB)
#     cursor = conn.cursor()
#     result = 0
#
#     try:
#         sql = "select distinct(poi_name) from staypoint"
#         result = cursor.execute(sql)
#         result = cursor.fetchall()
#         conn.commit()
#     except Exception, e:
#         print e
#         conn.rollback()
#
#     print "POI数目: ", len(result)
#
#     poi_axis = {}
#     axis_poi = {}
#     index = 0
#     for item in result:
#         poi_axis[item[0]] = index
#         axis_poi[index] = item[0]
#         index += 1
#
#     data = {}
#     try:
#         for user in range(0, 182):
#             sql = "select poi_name, mean_coordinate_latitude, mean_coordinate_longtitude from staypoint where user_id = "+ str(user) +" order by id"
#             result = cursor.execute(sql)
#             result = cursor.fetchall()
#             conn.commit()
#             data[user] = []
#             for item in result:
#                 data[user].append((poi_axis[item[0]], item[1], item[2]))
#     except Exception, e:
#         print "exception: ", e.message
#         conn.rollback()
#
#     # print len(result), type(result), len(data), data
#
#     cursor.close()
#     conn.close()
#
#     return poi_axis, axis_poi, data

def rank_poi(x, axis_poi, axis_poi2):
    sort_data = []
    for index in range(0, len(x)):
        meta_data = (index, x[index])
        sort_data.append(meta_data)
    sort_data.sort(key=lambda x: x[1], reverse=True)
    res = []
    for i in range(0, len(sort_data)):
        res.append((i+1, axis_poi[axis_poi2[sort_data[i][0]]], sort_data[i][1]))

    return res


if __name__ == '__main__':
    i_distance = 2
    current_position = (116.361684, 39.951409)  # poi为1231
    poi_axis, axis_poi, data, name2type = init_data(tuple(range(0, 182)), return_gps=True, return_poi_type=True)   # (0, 182)
    print "当前位置: ", axis_poi[1231]
    print "name2type: ", name2type
    count_list = []
    count_set = set()

    for i in range(0, len(data)):
        distance = calculate_distance(data[i][2], data[i][1], current_position[1], current_position[0])
        if distance <= i_distance:
            count_list.append(data[i])
            count_set.add(data[i][0])

    print "count_list: ", len(count_list)
    print count_set
    print len(count_set)
    dimensionality = len(count_set)

    pois_axis2 = {}
    axis_pois2 = {}
    index = 0
    for item in count_set:
        pois_axis2[item] = index
        axis_pois2[index] = item
        index += 1

    count_list2 = []
    for item in count_list:
        count_list2.append(pois_axis2[item[0]])

    tensor = trans(count_list2, dimensionality)
    alpha = 0.95
    v = []
    for i in range(0, dimensionality):
        v.append(1/dimensionality)
    x, hist, flag, ihist = POIRank(np.array(tensor), alpha, v).solve()
    print "hist: ", hist
    print "ihist: ", ihist
    print "flag: ", flag
    if flag == 1:
        print "收敛，x: ", x
    else:
        print "无法收敛"

    rank_result = rank_poi(x, axis_poi, axis_pois2)
    sum = 0.0
    for item in rank_result:
        print item[0], item[1], item[2]
        sum += item[2]

    print "POIRank值的和： ", sum

    prefs = {}
    axis_prefs, prefs_axis, datas = init_poi_type_data(tuple(range(0, 182)))
    print "axis_prefs: ", axis_prefs
    for key in axis_prefs.keys():
        print key, " ", axis_prefs[key]
    print "datas: ", datas
    for key in datas.keys():
        data = datas[key]
        # print "data_length: ", len(data)
        if len(data) > 1:
            tensor = trans2(data, len(axis_prefs), 2)
            result = param(tensor)
            T = pykov.Chain(result)
            res = T.steady()
            print "user"+str(key)+": ", vec(res, len(axis_prefs), len(pois_axis2))
            prefs[key] = vec(res, len(axis_prefs), len(pois_axis2))

    # 综合推荐
    user = 1

    rank_result2 = []
    for item in rank_result:
        res = item[2] * prefs[user][prefs_axis[name2type[item[1]]]]
        rank_result2.append((item[1], res))


    rank_result2.sort(key=lambda x: x[1], reverse=True)

    index = 1
    for item in rank_result2:
        print index, item[0], item[1]
        index += 1






    # pois_axis2 = {}
    # axis_pois2 = {}
    # index = 0
    # for item in count_set:
    #     pois_axis2[item] = index
    #     axis_pois2[index] = item
    #     index += 1
    #
    # count_list2 = []
    # for item in count_list:
    #     count_list2.append(pois_axis2[item[0]])
    # print "count_list2: ", len(count_list2)
    #
    # tensor = [[[0 for i in range(dimensionality)] for j in range(dimensionality)] for k in range(dimensionality)]
    # for i in range(0, len(count_list2)-2):
    #     tensor[count_list2[i]][count_list2[i+1]][count_list2[i+2]] += 1
    #
    # print "tensor: ", tensor



    # for i in range(0, len(data)-2):
    #     pre_distance = calculate_distance(data[i][2], data[i][1], current_position[1], current_position[0])
    #     now_distance = calculate_distance(data[i+1][2], data[i+1][1], current_position[1], current_position[0])
    #     next_distance = calculate_distance(data[i+2][2], data[i+2][1], current_position[1], current_position[0])
    #     # print distance
    #     if pre_distance > i_distance:
    #         pre_flag = False
    #     else:
    #         pre_flag = True
    #
    #     if now_distance > i_distance:
    #         now_flag = False
    #     else:
    #         now_flag = True
    #
    #     if next_distance > i_distance:
    #         next_flag = False
    #     else:
    #         next_flag = True
    #
    #     if pre_flag and now_flag and next_flag:
    #         count_list.append((data[i][0], data[i+1][0], data[i+2][0]))
    #         count_set.add(data[i][0])
    #         count_set.add(data[i+1][0])
    #         count_set.add(data[i+2][0])
    #
    # print "count_list: ", count_list
    # # print count_set
    # print len(count_set)
    # dimensionality = len(count_set)
    #
    # pois_axis2 = {}
    # axis_pois2 = {}
    # index = 0
    # for item in count_set:
    #     pois_axis2[item] = index
    #     axis_pois2[index] = item
    #     index += 1
    #
    # count_list2 = []
    # for item in count_list:
    #     count_list2.append((pois_axis2[item[0]], pois_axis2[item[1]], pois_axis2[item[2]]))
    #
    # print "count_list2: ", count_list2
    #
    # tensor = [[[0 for i in range(dimensionality)] for j in range(dimensionality)] for k in range(dimensionality)]
    #
    # for item in count_list2:
    #     print item[0], item[1], item[2]
    #     tensor[item[0]][item[1]][item[2]] += 1
    #
    # print "count_tensor: ", tensor

    # for item in range(0, dimensionality):
    #     for item2 in range(0, dimensionality):
    #         count_sum = 0
    #         for item3 in range(0, dimensionality):
    #             count_sum += tensor[item][item2][item3]
    #         if 0 == count_sum:
    #             continue
    #         else:
    #             for item4 in range(0, dimensionality):
    #                 tensor[item][item2][item4] = tensor[item][item2][item4] / count_sum

    # print "transition tensor: ", tensor









    # poi_axis, axis_poi, data = init_data(tuple(range(0, 182)), return_gps=True)
    # data_length = len(data)
    # print "data_length: ", data_length
    # print data
    # y_data = []
    # for i in range(data_length-1):
    #     y_data.append(calculate_distance(data[i][2], data[i][1], data[i+1][2], data[i+1][1]))
    #
    # print min(y_data), max(y_data)
    # w = 1.0
    # y_values = []
    # x_values = []
    # res = 0
    # max_y_value = 0
    # for index, x in enumerate(linspace(min(y_data), max(y_data), 1000)):
    #     y = kde(x, w, y_data)
    #     y_values.append(y)
    #     x_values.append(x)
    #
    #     # flag += 1
    #     if y >= max_y_value:
    #         max_y_value = y
    #         res = index
    #
    # print res
    # print x_values[res]
    #
    #
    # pylab.plot(x_values, y_values, 'r.')
    # pylab.show()