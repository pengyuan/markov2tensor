#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from poirank.transition import init_data, init_data2
import pylab
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


# Calculating kernel density estimates
# z: position, w: bandwidth, xv: vector of points
def kde(z, w, xv):
    return sum(exp(-0.5*((z-xv)/w)**2)/sqrt(2*pi*w**2))


if __name__ == '__main__':
    poi_axis, axis_poi, data = init_data2(tuple(range(0, 182)), return_gps=True)
    data_length = len(data)
    print "data_length: ", data_length
    print data
    y_data = []

    for key in data.keys():
        data_key = data[key]
        for i in range(len(data_key)-1):
            y_data.append(calculate_distance(data_key[i][2], data_key[i][1], data_key[i+1][2], data_key[i+1][1]))

    print min(y_data), max(y_data)
    w = 1.0
    y_values = []
    x_values = []
    res = 0
    max_y_value = 0
    for index, x in enumerate(linspace(min(y_data), max(y_data), 1000)):
        y = kde(x, w, y_data)
        y_values.append(y)
        x_values.append(x)

        # flag += 1
        if y >= max_y_value:
            max_y_value = y
            res = index

    print res
    print x_values[res]

    pylab.plot(x_values, y_values, 'r.')
    pylab.xlabel(u"兴趣点间距（千米）")
    pylab.ylabel(u"核密度分布值")
    pylab.title(u"兴趣点间距的核密度分布")
    pylab.legend(loc='upper right')
    pylab.show()