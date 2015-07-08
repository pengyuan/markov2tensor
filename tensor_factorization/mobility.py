#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from tensor_factorization.util import *
from preprocess import settings

__author__ = 'Peng Yuan <pengyuan.org@gmail.com>'
__copyright__ = 'Copyright (c) 2014 Peng Yuan'
__license__ = 'Public domain'


# 一个用户必须访问10个poi，一个poi必须被10个用户访问，大部分情况poi数目>>用户数目，后者更重要
def init_data(time_slice, train, region, filter_count):
    conn = MySQLdb.connect(host=settings.HOST, user=settings.USER, passwd=settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

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
        full_data.append((users_axis[item[0]], pois_axis[item[1]], time, item[3], item[4], item[5]))

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
        if item[1] not in know_poi_set[item[0]][item[2]]:
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
                                future = data[0]
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


def trans(train_structure_data, order, poi_num, user_num, time_slice):
    time_slot = range(0, time_slice)
    if order == 2:
        tensor = [[[[0 for i in range(poi_num)] for j in range(poi_num)] for k in range(time_slice)] for l in range(user_num)]
    else:
        tensor = [[[[[0 for i in range(poi_num)] for j in range(poi_num)] for k in range(poi_num)] for l in range(time_slice)] for m in range(user_num)]

    for key in train_structure_data.keys():
        for time in time_slot:
            data = train_structure_data[key][time]
            if order == 3:
                if len(data) < 3:
                    continue
                else:
                    for index in range(len(data)-2):
                        past = data[index]
                        now = data[index+1]
                        future = data[index+2]
                        tensor[key][time][past][now][future] += 1
            if order == 2:
                if len(data) < 2:
                    continue
                else:
                    for index in range(len(data)-1):
                        now = data[index]
                        future = data[index+1]
                        tensor[key][time][now][future] += 1

    return tensor

if __name__ == '__main__':
    print "here"