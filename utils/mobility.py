#!/usr/bin/env python
# coding: UTF-8
"""
gowalla_filter:
SELECT user,COUNT(unkown) as loc,COUNT(DISTINCT unkown) as distinct_loc,COUNT(unkown)/COUNT(DISTINCT unkown) as ratio FROM raw a GROUP BY `user` ORDER BY ratio desc;
找出那些比率（所有地点/不同地点）合适的用户
所有地点决定了tensor的稀疏度；不同地点决定了tensor的dimensionality

eg：找到了用户id为147986的所有记录，并将unknow一栏替换为字母（为了方便分析）
"""

from __future__ import division
import MySQLdb
from scipy import linalg
import numpy as np
from numpy.matlib import eye, identity
from preprocess import settings

__author__ = 'Peng Yuan <pengyuan.org@gmail.com>'
__copyright__ = 'Copyright (c) 2014 Peng Yuan'
__license__ = 'Public domain'

#连接数据库
def init_data(users, train = 1):
    conn = MySQLdb.connect(host = settings.HOST, user = settings.USER, passwd = settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

    #得到用户所有位置移动信息，按时间排序
    #select distinct poi_name  from staypoint where user_id in (0,3,4,5,30) and province = '北京市' and district = "海淀区";

    try:
        if len(users) == 1:
            sql = "select distinct(poi_name) from staypoint where user_id = "+ str(users[0]) +" and  province = '北京市' and district = '海淀区' order by id"
        else:
            sql = "select distinct(poi_name) from staypoint where user_id in "+ users.__str__() +" and province = '北京市' and district = '海淀区' order by id"
        print sql
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    #print len(result)
    
    pois_axis = {}
    axis_pois = {}
    index = 0
    for item in result:
        pois_axis[item] = index
        axis_pois[index] = item
        index += 1
    
    datas = {}
    predicts = {}
    recommends = {}
    for user in users:     
        try:
            sql = "select poi_name from staypoint where user_id = "+ str(user) +" and province = '北京市' and district = '海淀区' order by id"
            result = cursor.execute(sql)
            result = cursor.fetchall()
            conn.commit()
        except Exception, e:
            print e
            conn.rollback()
            
        data = []
        length = int(len(result) * train)
        train_data = result[:length]
        remain_data = result[length:]
        for item in train_data:
            data.append(pois_axis[item])
        train_set = set(train_data)
        predict = []
        recommend = []
        for item in remain_data:
            if item in train_set:
                predict.append(pois_axis[item])
            else:
                recommend.append(pois_axis[item])

        datas[user] = data
        predicts[user] = predict
        recommends[user] = recommend

    cursor.close()
    conn.close()    
#     print pois_axis
#     print axis_pois
#     print datas
    
    return axis_pois, datas, predicts, recommends


# 连接数据库
'''strategy 1: arrival_slot; 2: existance'''
def init_data2(users, train, time_slice):
    conn = MySQLdb.connect(host = settings.HOST, user = settings.USER, passwd = settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

    #得到用户所有位置移动信息，按时间排序
    #select distinct poi_name  from staypoint where user_id in (0,3,4,5,30) and province = '北京市' and district = "海淀区";

    try:
        if len(users) == 1:
            sql = "select distinct(poi_name) from staypoint where user_id = "+ str(users[0]) +" and  province = '北京市' and district = '海淀区' order by id"
        else:
            sql = "select distinct(poi_name) from staypoint where user_id in "+ users.__str__() +" and province = '北京市' and district = '海淀区' order by id"
        print sql
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()

    # print len(result)

    pois_axis = {}
    axis_pois = {}
    index = 0
    for item in result:
        pois_axis[item[0]] = index
        axis_pois[index] = item[0]
        index += 1

    datas = {}
    predicts = {}
    recommends = {}
    # trains = {}
    time_slot = range(0, time_slice)
    for user in users:
        try:
            # sql = "select poi_name from staypoint where user_id = "+ str(user) +" and province = '北京市' and district = '海淀区' and arrival_timestamp % 86400 div 3600 = "+str(slot)
            sql = "select poi_name, arrival_timestamp from staypoint where user_id = "+ str(user) +" and province = '北京市' and district = '海淀区' order by id"
            result = cursor.execute(sql)
            result = cursor.fetchall()
            conn.commit()
        except Exception, e:
            print e
            conn.rollback()

        data = {}
        for slot in time_slot:
            data[slot] = []
        length = int(len(result) * train)
        train_data = result[:length]
        remain_data = result[length:]
        # train_data_list = []

        for item in train_data:
            # print data.keys()
            index = item[1] % 86400 // (3600 * (24 // time_slice))
            # print type(index)
            # print data.has_key(index)
            data[index].append(pois_axis[item[0]])
            # train_data_list.append(pois_axis[item[0]])

        datas[user] = data

        # train_set = set(train_data_list)
        # print "trainset: ", train_set
        predict = {}
        recommend = {}
        for slot in time_slot:
            recommend[slot] = set()
            predict[slot] = set()
        for item in remain_data:
            axis = pois_axis[item[0]]
            index = item[1] % 86400 // (3600 * (24 // time_slice))
            if axis in set(data[index]):
                predict[index].add(pois_axis[item[0]])
            else:
                recommend[index].add(pois_axis[item[0]])

        predicts[user] = predict
        recommends[user] = recommend
        # trains[user] = train_set

    cursor.close()
    conn.close()
#     print pois_axis
#     print axis_pois
#     print datas

    return axis_pois, datas, predicts, recommends


# 从线性停留点序列计算马儿可夫转移矩阵或转移张量
def trans(data, dimensionality, order):
    # 得到停留点序列长度
    data_length = len(data)
    
    if order == 2:
        tensor = [[0 for i in range(dimensionality)] for j in range(dimensionality)]
        for index in range(data_length-1):
            check_list = data[index:index+2]
            tensor[check_list[0]][check_list[1]] += 1
            
        for item in range(dimensionality):
            count_sum = 0
            for item2 in range(dimensionality):
                count_sum += tensor[item][item2]
            if 0 == count_sum:
                continue
            else:
                for item3 in range(dimensionality):
                    tensor[item][item3] = tensor[item][item3] / count_sum
    
    elif order == 3:
        # 三维数组，元素初始化为零
        tensor = [[[0 for i in range(dimensionality)] for j in range(dimensionality)] for k in range(dimensionality)]
        
        for index in range(data_length-2):
            check_list = data[index:index+3]
            tensor[check_list[0]][check_list[1]][check_list[2]] += 1
        
        for item in range(dimensionality):
            for item2 in range(dimensionality):
                count_sum = 0
                for item3 in range(dimensionality):
                    count_sum += tensor[item][item2][item3]
                if 0 == count_sum:
                    continue
                else:        
                    for item4 in range(dimensionality):    
                        tensor[item][item2][item4] = tensor[item][item2][item4] / count_sum

    return tensor


# 从线性停留点序列统计用户-时间-频数
def trans2(data_map, poi_dimension, users, time_slice):
    user_dimension = len(users)

    # 三维数组，元素初始化为零
    tensor = [[[0 for poi in range(poi_dimension)] for time in range(0, time_slice)] for user in range(user_dimension)]

    print np.array(tensor).shape
    for key in data_map.keys():
        data = data_map[key]
        for slot in range(0, time_slice):
            poi_list = data[slot]
            for poi in poi_list:
                tensor[users.index(key)][slot][poi] += 1

    # for item in range(dimensionality):
    #     for item2 in range(dimensionality):
    #         count_sum = 0
    #         for item3 in range(dimensionality):
    #             count_sum += tensor[item][item2][item3]
    #         if 0 == count_sum:
    #             continue
    #         else:
    #             for item4 in range(dimensionality):
    #                 tensor[item][item2][item4] = tensor[item][item2][item4] / count_sum

    return tensor


def is_contain_zero(vector):
    length = len(vector)

    while(True):
        if vector[length-1] == 0:
            length -= 1
        else:
            break

    return vector.any(0), length-1


def matrix_sn_nn(res):
    # # print tensor[:-1]
    # x = np.array(matrix)
    #
    # # sum(1) 按行求和
    # print "sum: ", x.sum(1)
    #
    # U, s, Vh = linalg.svd(matrix, full_matrices=True)
    # # print type(s)
    #
    # # print U
    # U2 = U[:, :]
    # # print U2
    #
    # V2 = Vh[:, :]
    #
    # s = s[:]
    # S = np.diag(s)
    # # print S
    #
    # # S = linalg.diagsvd(s, 6, 6)
    # # print np.allclose(tensor, np.dot(U, np.dot(S, Vh)))
    #
    # print np.allclose(matrix, np.dot(U2, np.dot(S, V2)))
    #
    # temp = U2.transpose().sum(1)
    # print "temp1: ", temp.shape
    # temp = np.array([temp]).transpose()
    # print "temp2: ", temp.shape
    #
    # # print type(temp)
    # # print identity(4)
    # #
    # #
    # # print type(eye(4))
    # # print eye(4).shape[1]
    #
    # flag, num = is_contain_zero(temp)
    # nr = U2.shape[1]
    #
    # print "is_contains_zero: ", flag, num
    #
    # if flag:
    #     print nr, num, type(np.zeros((nr, num-1))), type(temp)
    #     print np.zeros((nr, num-1)).shape
    #     print temp.shape, np.ones((nr, 1)).shape
    #
    #     print np.sum([[0, 1], [0, 5]], axis=1)
    #
    #
    #     temp_matrix = np.concatenate((np.zeros((nr, num-1)), temp-np.ones((nr, 1))), 1)
    #     sigma = identity(nr) + np.concatenate((temp_matrix, np.zeros((nr, nr-num))), 1)
    # else:
    #     sigma = np.diag(temp)
    #
    # res = U2.dot(sigma)

    print "res1: ", res
    res = np.array(res)
    nc = res.shape[1]
    res_min = res.min()
    # print np.transpose(S)

    if res_min >= -1:
        param = 1
    else:
        param = 1/abs(res_min)

    param_matrix = (1/(nc+param))*(np.ones((nc, nc)) + param * eye(nc))
    result = res.dot(param_matrix)

    print result.sum(1)

    return np.array(result)#, sigma, param_matrix


if __name__ == '__main__':
    # init_data((0, 3, 4, 5, 30))

    #res, sigma, param = matrix_sn_nn([[2**0.5/2, -2**0.5/2], [2**0.5/2, 2**0.5/2]])
    #res, sigma, param = matrix_sn_nn([[0.2, 0.8], [0.3, 0.7]])
    # print "res2: ", res,res[0][0],res[0][1]
    # print "sigma: ", sigma
    # print "param: ", param.dot(np.linalg.inv(param))
    # res = matrix_sn_nn([[0.1, 0.2, 0.3, 0.4], [0.3, 0.6, 0.05, 0.05]])
    res = matrix_sn_nn([[-0.1, 0.2, 0.5, 0.5], [0.3, 0.6, 0.1, 0]])
    print "res2:", res