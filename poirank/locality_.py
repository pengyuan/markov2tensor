#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import math
import numpy as np
from poirank.transition import init_data, trans
import pylab
from numpy import *

# 以紫荆园为中心
z_latitude = 40.010428
z_longtitude = 116.322341


poi_axis, axis_poi, data = init_data(tuple(range(0, 182)), return_gps=True)
data_length = len(data)
print "data_length: ", data_length
#print data


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


# Calculating kernel density estimates
# z: position, w: bandwidth, xv: vector of points
def kde(z, w, xv):
    return sum(exp(-0.5*((z-xv)/w)**2)/sqrt(2*pi*w**2))

dis_range = []
y_values = []
x_values = []
m_distance = 0.1
# for m_distance in range(1, 200, 1):
while m_distance <= 20:
    in_count = 0
    out_count = 0
    for i in range(data_length-1):
        # print data[i][2], data[i][1]
        distance1 = calculate_distance(data[i][1], data[i][2], z_latitude, z_longtitude)
        distance2 = calculate_distance(data[i+1][1], data[i+1][2], z_latitude, z_longtitude)
        print m_distance, i, distance1, distance2
        if (distance1 < m_distance) and (distance2 < m_distance):
            in_count += 1
        elif (distance1 < m_distance) and (distance2 > m_distance):
            out_count += 1
        elif (distance1 > m_distance) and (distance2 < m_distance):
            out_count += 1
        else:
            continue
    # print in_count, out_count
    if in_count + out_count == 0:
        print "................................................................."
        continue
    ratio = in_count / (in_count + out_count)
    x_values.append(m_distance)
    y_values.append(ratio)

    if ratio > 0.5 and ratio < 0.68:
        dis_range.append(m_distance)

    m_distance += 0.1

print (min(dis_range), max(dis_range))
pylab.plot(x_values, y_values, 'r.')
pylab.xlabel(u"区域的半径（千米）")
pylab.ylabel(u"聚合度")
pylab.title(u"区域半径与聚合度之间的关系")
pylab.legend(loc='lower right')
pylab.show()

# for i in range(data_length):
#     distance = calculate_distance(data[i][2], data[i][1], z_latitude, z_longtitude)
#
#
#
#
#
#
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