#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import MySQLdb
import string
"""
gowalla_filter:
SELECT user,COUNT(unkown) as loc,COUNT(DISTINCT unkown) as distinct_loc,COUNT(unkown)/COUNT(DISTINCT unkown) as ratio FROM raw a GROUP BY `user` ORDER BY ratio desc;
找出那些比率（所有地点/不同地点）合适的用户
所有地点决定了tensor的稀疏度；不同地点决定了tensor的dimensionality

eg：找到了用户id为147986的所有记录，并将unknow一栏替换为字母（为了方便分析）
"""

__author__ = 'Peng Yuan <pengyuan.org@gmail.com>'
__copyright__ = 'Copyright (c) 2014 Peng Yuan'
__license__ = 'Public domain'

#连接数据库
conn = MySQLdb.connect(host = 'localhost', user = 'pythonic', passwd = 'pythonic', db='test')
cursor = conn.cursor()
result = 0

#得到用户所有位置移动信息，按时间排序
try:
    sql = "select unknow from aaaaa order by time"
    result = cursor.execute(sql)
    result = cursor.fetchall()
    conn.commit()
except Exception, e:
    print e
    conn.rollback()
finally:
    cursor.close()
    conn.close()
    
result_str = ''
for item in result:
    result_str = result_str + item[0]
    
print '访问序列：'+result_str
#HIHHHIHHHJHIHHIHIHHHHGHHHGHGHHIHIHHHBHHGHHHHHHHJGHHHHCHIDFHHHHHHEIHHAEAHGHIHHAHHIHHHHJIHIHHHHHHHHHHG

length = len(result_str)
print '长度：'+ str(length)
#100

range_char = string.uppercase[:10]
#[A,B,C,D,E,F,G,H,I,J]

#三维数组，元素初始化为零
tensor = [[[0 for a in range(10)] for col in range(10)] for row in range(10)]
print tensor

#在tensor中将地点与坐标关联，物理世界与数学世界的映射
func = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

def count():
    for item in range_char:
        for item2 in range_char:
            for item3 in range_char:
                check_str = item + item2 + item3
                counter = 0
                while counter < length-3:
                    if result_str[counter:counter+3] == check_str:
                        tensor[func[item]][func[item2]][func[item3]] += 1
                    counter += 1
                    
def trans():
    for item in range(10):
        for item2 in range(10):
            sum = 0
            for item3 in range(10):
                sum += tensor[item][item2][item3]
            print sum
            if 0 == sum:
                continue
            else:        
                for item4 in range(10):    
                    tensor[item][item2][item4] = tensor[item][item2][item4] / sum 
                    
#通过计数，构建tensor               
count()
print tensor

#建立转移概率张量
trans()
print tensor

