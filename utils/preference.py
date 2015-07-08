#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import MySQLdb
from eigentensor import pykov
from preprocess import settings

# 连接数据库
from utils import pykov
from utils import mobility
from mobility import trans

def init_data(users):
    conn = MySQLdb.connect(host=settings.HOST, user=settings.USER, passwd=settings.PASSWORD, db=settings.DB)
    cursor = conn.cursor()
    result = 0

    prefs_axis = {}
    axis_prefs = {}

    try:
        sql = "select distinct(poi_type) from staypoint where poi_type is not NULL"
        result = cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception, e:
        print e
        conn.rollback()
    
    index = 0
    for item in result:
        prefs_axis[item[0]] = index
        axis_prefs[index] = item[0]
        index += 1
        
    #print prefs_axis
    #print axis_prefs
    
    
    datas = {}
    for user in users:
        data = []
        try:
            sql = "select poi_type from staypoint where user_id = "+ str(user) +" and poi_type is not NULL"
            result = cursor.execute(sql)
            result = cursor.fetchall()
            conn.commit()
        except Exception, e:
            print e
            conn.rollback()   
        if len(result) > 0:
            for item in result:
                data.append(prefs_axis[item[0]])
        
            datas[user] = data
        else:
            print 'no data: ', str(user)
        
    cursor.close()
    conn.close()    
#     print pois_axis
#     print axis_pois
#     print datas
    
    return axis_prefs, datas


def param(tensor):
    result = {}
    #{('A','B'): .3, ('A','A'): .7, ('B','A'): 1.}
    dimen = len(tensor)
    #print "dimen", dimen
    for i in range(dimen):
        for j in range(dimen):
            if tensor[i][j] != 0:
                result[(i,j)] = tensor[i][j]
    
    return result


def vec(param, dimen):
    # param[(0, 0.3333333333333332), (2, 0.25000000000000006), (9, 0.24999999999999994), (11, 0.16666666666666671)]
    parameter = {}
    for item in param:
        #print "item:",item
        parameter[item[0]] = item[1]
    # print parameter
     
    preference = []
    for i in range(dimen):
        # print i
        if parameter.has_key(i):
            preference.append(parameter[i])
        else:
            preference.append(0.0)
    
    return preference


if __name__ == '__main__':
    axis_prefs, datas = init_data(tuple(range(0, 182)))
    print "axis_prefs: ", axis_prefs
    for key in axis_prefs.keys():
        print key, " ", axis_prefs[key]
    print "datas: ", datas
    for key in datas.keys():
        data = datas[key]
        tensor = trans(data, len(axis_prefs), 2)
        result = param(tensor)
        T = pykov.Chain(result)
        res = T.steady()
        print "res: ", res, type(res)

        print "user"+str(key)+": ", res
