#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from factorized_markov_chain.hosvd import HOSVD, reconstruct
from factorized_markov_chain.util import *
from preprocess import settings

__author__ = 'Peng Yuan <pengyuan.org@gmail.com>'
__copyright__ = 'Copyright (c) 2014 Peng Yuan'
__license__ = 'Public domain'


# 连接数据库
def init_data(region, train, time_slice, filter_count):
    conn = MySQLdb.connect(host=settings.HOST, user=settings.USER, passwd=settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

    # 得到区域用户所有位置移动信息，按时间排序
    try:
        sql = "select user_id from staypoint where mean_coordinate_latitude between "+str(region[0])+" and "+str(region[1])+" and mean_coordinate_longtitude between "+str(region[2])+" and "+str(region[3])+" group by user_id having count(*) > "+str(filter_count)
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    user_available = []
    for item in result:
        user_available.append(int(item[0]))

    if len(user_available) == 0:
        raise "没有足够数据"
    try:
        sql = "select user_id, poi_name,arrival_timestamp, mean_coordinate_latitude, mean_coordinate_longtitude, poi_distance from staypoint where user_id in "+tuple(user_available).__str__()+" and mean_coordinate_latitude between "+str(region[0])+" and "+str(region[1])+" and mean_coordinate_longtitude between "+str(region[2])+" and "+str(region[3])+" order by id"
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    temp_data = []
    for item in result:
        temp_data.append((item[0], item[1], item[2], item[3], item[4], item[5]))

    cursor.close()
    conn.close()
    return temp_data, time_slice, train


def preprocess(temp_data, time_slice, train, order):
    length = int(len(temp_data) * train)
    recommends = {}
    time_slot = range(0, time_slice)

    poi_set = set()
    user_set = set()
    for item in temp_data:
        poi_set.add(item[1])
        user_set.add(item[0])

    poi_num = len(poi_set)
    user_num = len(user_set)
    print "poi数目：", poi_num
    print "用户数目：", user_num

    pois_axis = {}
    axis_pois = {}
    index = 0
    for item in poi_set:
        pois_axis[item] = index
        axis_pois[index] = item
        index += 1

    users_axis = {}
    axis_users = {}
    index = 0
    for item in user_set:
        users_axis[item] = index
        axis_users[index] = item
        index += 1

    full_data = []
    for item in temp_data:
        time = int(item[2] % 86400 // (3600 * (24 // time_slice)))
        full_data.append((users_axis[item[0]], pois_axis[item[1]], time))

    train_data = full_data[:length]
    test_data = full_data[length:]

    train_structure_data = {}
    test_structure_data = {}
    know_poi_set = {}
    unknow_poi_set = {}
    for user in range(user_num):
        train_structure_data[user] = {}
        test_structure_data[user] = {}
        know_poi_set[user] = {}
        unknow_poi_set[user] = {}
        recommends[user] = {}
        for time in time_slot:
            train_structure_data[user][time] = []
            test_structure_data[user][time] = []
            know_poi_set[user][time] = set()
            unknow_poi_set[user][time] = set()
            recommends[user][time] = []

    for item in train_data:
        train_structure_data[item[0]][item[2]].append(item[1])
        know_poi_set[item[0]][item[2]].add(item[1])

    for item in test_data:
        test_structure_data[item[0]][item[2]].append(item[1])
        unknow_poi_set[item[0]][item[2]].add(item[1])

    for user in range(user_num):
        for time in time_slot:
            data = test_structure_data[user][time]
            data_length = len(data)
            if data_length == 0:
                recommends[user][time] = None
            else:
                for index in range(data_length):
                    if data[index] not in unknow_poi_set[user][time]:
                        continue
                    else:
                        if order == 3:
                            if index == 0:
                                if len(train_structure_data[user][time]) < 2:
                                    continue
                                past = train_structure_data[user][time][-2]
                                now = train_structure_data[user][time][-1]
                                future = train_structure_data[user][time][0]
                            elif index == 1:
                                if len(train_structure_data[user][time]) < 1:
                                    continue
                                past = train_structure_data[user][time][-1]
                                now = data[0]
                                future = data[1]
                            else:
                                past = data[index-2]
                                now = data[index-1]
                                future = data[index]
                            recommends[user][time].append((past, now, future))

                        else:
                            if index == 0:
                                if len(train_structure_data[user][time]) < 1:
                                    continue
                                now = train_structure_data[user][time][-1]
                                future = data[0]
                            else:
                                now = data[index-1]
                                future = data[index]
                            recommends[user][time].append((now, future))

    return axis_pois, axis_users, train_structure_data, recommends, unknow_poi_set


def trans(train_structure_data, order, poi_num, time_slice, threshold):
    time_slot = range(0, time_slice)
    result = {}
    for user in train_structure_data.keys():
        result[user] = {}
        for time in time_slot:
            result[user][time] = None

    for key in train_structure_data.keys():
        for time in time_slot:
            if order == 2:
                tensor = [[0 for i in range(poi_num)] for j in range(poi_num)]
            else:
                tensor = [[[0 for i in range(poi_num)] for j in range(poi_num)] for k in range(poi_num)]
            data = train_structure_data[key][time]
            if order == 3:
                if len(data) < 3:
                    A = None
                    continue
                else:
                    for index in range(len(data)-2):
                        check_list = data[index:index+3]
                        tensor[check_list[0]][check_list[1]][check_list[2]] += 1

                    for item in range(poi_num):
                        for item2 in range(poi_num):
                            count_sum = 0
                            for item3 in range(poi_num):
                                count_sum += tensor[item][item2][item3]
                            if 0 == count_sum:
                                continue
                            else:
                                for item4 in range(poi_num):
                                    tensor[item][item2][item4] = tensor[item][item2][item4] / count_sum

                    U, S, D = HOSVD(numpy.array(tensor), threshold)
                    A = reconstruct(S, U)

                    # U, S, V = linalg.svd(numpy.array(tensor), 0.7)

            if order == 2:
                if len(data) < 2:
                    A = None
                    continue
                else:
                    for index in range(len(data)-1):
                        check_list = data[index:index+2]
                        tensor[check_list[0]][check_list[1]] += 1

                    for item in range(poi_num):
                        count_sum = 0
                        for item2 in range(poi_num):
                            count_sum += tensor[item][item2]
                        if 0 == count_sum:
                            continue
                        else:
                            for item3 in range(poi_num):
                                tensor[item][item3] = tensor[item][item3] / count_sum

                    U, S, D = HOSVD(numpy.array(tensor), threshold)
                    A = reconstruct(S, U)

            result[key][time] = A

    return result

if __name__ == '__main__':
    print "here"