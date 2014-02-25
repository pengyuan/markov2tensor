#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import MySQLdb
import string
conn = MySQLdb.connect(host = 'localhost', user = 'pythonic', passwd = 'pythonic', db='test')
cursor = conn.cursor()
result = 0
try:
    sql = "select unkown from aaaaa"
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
print '长度：'+str(len(result_str))

length = len(result_str)
range_char = string.uppercase[:10]

count_num = [[[0 for a in range(10)] for col in range(10)] for row in range(10)]
prob_tensor = [[[0 for a in range(10)] for col in range(10)] for row in range(10)]
#print count_num

func = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

def count():
    for item in range_char:
        for item2 in range_char:
            for item3 in range_char:
                check_str = item+item2+item3
                #print 'check_str:',check_str
                counter = 0
                while counter<length-3:
                    #print result_str[counter:counter+3]
                    if result_str[counter:counter+3] == check_str:
                        #print 'yesssssssssssssssssssssssssssssssssssssssssssssssssssssss'
                        #print result_str[counter:counter+3],' ',check_str
                        count_num[func[item]][func[item2]][func[item3]] += 1
                    counter += 1
                 
count()
#print count_num[7][7][7]

def transform():
    for item in range(10):
        for item2 in range(10):
            sum = 0
            for item3 in range(10):
                sum += count_num[item][item2][item3]
            print sum,'  ',item,item2,item3
            for item3 in range(10):
                if sum == 0:
                    prob_tensor[item][item2][item3] = float(0)
                else:
                    prob_tensor[item][item2][item3] = count_num[item][item2][item3]/float(sum)
                
            
transform()
print count_num[7][7][9]
print prob_tensor[7][7][9]

print count_num[7][8][9]
print prob_tensor[7][8][9]

print prob_tensor
