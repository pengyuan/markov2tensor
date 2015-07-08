#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
from poirank.transition import init_data, trans
import pylab

poi_axis, axis_poi, data = init_data(tuple(range(2,30)))
#tensor = trans(data, len(axis_poi))
# print "data: ", data
data_length = len(data)
print "data_length: ", data_length


def poi_coorelation(order):
    esum = 0.0
    for i in range(data_length):
        esum = esum + data[i]
    expectation = esum / data_length
    print "expectation: ", expectation

    vsum = 0.0
    for j in range(data_length):
        vsum += pow((data[j] - expectation), 2)
    print "vsum: ", vsum
    variance = vsum / data_length

    tsum = 0.0
    for k in range(data_length - order):
        tsum += (data[k] - expectation) * (data[k + order] - expectation)
    print "tsum: ", tsum
    ar = tsum / vsum

    print "ar: ", ar
    return ar



order = 1
y_values = []
x_values = []
while order <= 40:
    ar = poi_coorelation(order)
    y_values.append(ar)
    x_values.append(order)
    order += 1

pylab.plot(x_values, y_values, 'rs',linewidth=1, linestyle="-")
pylab.xlabel(u"马尔科夫链阶数")
pylab.ylabel(u"自相关系数")
pylab.title(u"马尔科夫链阶数与自相关系数的关系（兴趣点序列长度为3868）")
pylab.legend(loc='center right')
pylab.show()

#(2,3)   26,29
#(2,10)  494,2600
#(2,20)  853,3868

# from scipy import stats
# def measure(n):
#  "Measurement model, return two coupled measurements."
#  m1 = np.random.normal(size=n)
#  m2 = np.random.normal(scale=0.5, size=n)
#  return m1+m2, m1-m2
# m1, m2 = measure(2000)
# xmin = m1.min()
# xmax = m1.max()
# ymin = m2.min()
# ymax = m2.max()
# 对数据执行内核密度估计：
# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# values = np.vstack([m1, m2])
# kernel = stats.gaussian_kde(values)
# Z = np.reshape(kernel(positions).T, X.shape)
# 绘制的结果：
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#    extent=[xmin, xmax, ymin, ymax])
# ax.plot(m1, m2, 'k.', markersize=2)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])
# plt.show()