#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import MySQLdb
from preprocess import settings


def init_data(users, return_gps=False, return_poi_type=False):
    conn = MySQLdb.connect(host = settings.HOST, user = settings.USER, passwd = settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

    try:
        if len(users) == 1:
            sql = "select distinct(poi_name) from staypoint where user_id = "+ str(users[0]) +" and province = '北京市' and district = '海淀区' order by id"
        else:
            sql = "select distinct(poi_name) from staypoint where user_id in "+ users.__str__() +" and province = '北京市' and district = '海淀区' order by id"
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    print len(result)

    poi_axis = {}
    axis_poi = {}
    index = 0
    for item in result:
        poi_axis[item[0]] = index
        axis_poi[index] = item[0]
        index += 1

    data = []
    try:
        if len(users) == 1:
            sql = "select poi_name, mean_coordinate_latitude, mean_coordinate_longtitude, poi_type from staypoint where user_id = "+ str(users[0]) +" and province = '北京市' and district = '海淀区' order by id"
        else:
            sql = "select poi_name, mean_coordinate_latitude, mean_coordinate_longtitude, poi_type from staypoint where user_id in "+ users.__str__() +" and province = '北京市' and district = '海淀区' order by id"
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    if return_gps:
        for item in result:
            data.append((poi_axis[item[0]], item[1], item[2]))
    else:
        for item in result:
            data.append(poi_axis[item[0]])

    if return_poi_type:
        name2type = {}
        for item in result:
            name2type[item[0]] = item[3]

    # print len(result), type(result), len(data), data

    cursor.close()
    conn.close()

    return poi_axis, axis_poi, data #name2type


def init_data2(users, return_gps=False):
    conn = MySQLdb.connect(host = settings.HOST, user = settings.USER, passwd = settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

    try:
        if len(users) == 1:
            sql = "select distinct(poi_name) from staypoint where user_id = "+ str(users[0]) +" and province = '北京市' and district = '海淀区' order by id"
        else:
            sql = "select distinct(poi_name) from staypoint where user_id in "+ users.__str__() +" and province = '北京市' and district = '海淀区' order by id"
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    print len(result)

    poi_axis = {}
    axis_poi = {}
    index = 0
    for item in result:
        poi_axis[item[0]] = index
        axis_poi[index] = item[0]
        index += 1

    data = {}
    for user in users:
        data[user] = []
    try:
        if len(users) == 1:
            sql = "select user_id, poi_name, mean_coordinate_latitude, mean_coordinate_longtitude, poi_type from staypoint where user_id = "+ str(users[0]) +" and province = '北京市' and district = '海淀区' order by id"
        else:
            sql = "select user_id, poi_name, mean_coordinate_latitude, mean_coordinate_longtitude, poi_type from staypoint where user_id in "+ users.__str__() +" and province = '北京市' and district = '海淀区' order by id"
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    if return_gps:
        for item in result:
            data[item[0]].append((poi_axis[item[1]], item[2], item[3]))
    else:
        for item in result:
            data[item[0]].append(poi_axis[item[1]])

    cursor.close()
    conn.close()

    return poi_axis, axis_poi, data


# 从线性停留点序列计算马儿可夫转移矩阵或转移张量
def trans(data, dimensionality):
    data_length = len(data)
    tensor = [[[0 for i in range(dimensionality)] for j in range(dimensionality)] for k in range(dimensionality)]

    for index in range(0, data_length-2):
        check_list = data[index:index+3]
        tensor[check_list[0]][check_list[1]][check_list[2]] += 1

    # sparse(np.array(tensor))
    for item in range(0, dimensionality):
        for item2 in range(0, dimensionality):
            count_sum = 0
            for item3 in range(0, dimensionality):
                count_sum += tensor[item][item2][item3]
            if 0 == count_sum:
                continue
            else:
                for item4 in range(0, dimensionality):
                    tensor[item][item2][item4] = tensor[item][item2][item4] / count_sum

    return tensor


if __name__ == '__main__':
    # print init_data((0, 3, 4, 5, 30))
    poi_axis, axis_poi, data = init_data((3,))
    tensor = trans(data, len(axis_poi))
    print "tensor: ", tensor