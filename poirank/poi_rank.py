#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import numpy as np
from numpy.core.umath import spacing
from numpy.linalg import inv
from numpy.matlib import eye
from poirank.transition import init_data, trans
from util.hosvd import unfold
from util.util import epsf


class POIRank:
    def __init__(self, R, alpha, v):
        if R.ndim == 3:
            self.R = unfold(R, 3)
        else:
            self.R = R
        self.alpha = alpha
        self.v = np.array(v)

    def parse_param(self, n, maxiter=1e4, tol=1e-8, xtrue=[], randinit=False, x0=[]):
        niter = maxiter
        xcur = np.zeros(n)     # this allows us to keep v = 1/n
        xcur = xcur + self.v
        if randinit:
            xcur = np.rand(n, 1)
            xcur = xcur / sum(xcur)
        if x0:
            xcur = np.zeros(n) + x0

        hist = np.zeros((niter, 1))
        ihist = np.zeros((n, niter))

        return niter, xcur, hist, ihist

    def jacobian(self, x, gamma=1):
        n = self.R.shape[0]
        I = eye(n)
        return self.alpha * gamma * self.R * (np.kron(x, I) + np.kron(I, x)) + (1-gamma) * I

    def residual(self, x):
        return self.alpha * (self.R.dot(np.kron(x, x))) + (1 - self.alpha) * self.v - x

    def solve(self):
        if self.alpha < 1/2:
            return self.shifted()
        else:
            return self.inverseiter()

    # Run the power method (no shift) on a tensor PageRank problem
    def power(self):
        return self.shifted(0)

    def shifted(self, gamma=1, maxiter=1e4, tol=1e-8, xtrue=[], randinit=False, x0=[]):
        n = self.R.shape[0]
        Gamma = 1 / (1+gamma)
        niter, xcur, hist, ihist = self.parse_param(n, maxiter, tol, xtrue, randinit, x0)

        for i in range(1, int(niter+1)):
            y = self.alpha * (self.R.dot(np.kron(xcur, xcur)))
            # print y * Gamma
            # print Gamma * (1-sum(y)) * self.v
            # print Gamma * (1-sum(y)) * np.array(v)
            #print type(v)
            z = y * Gamma + Gamma * (1 - sum(y)) * self.v
            xn = z + (1 - sum(z)) * xcur

            ihist[:, i] = xn
            curdiff = np.linalg.norm(xcur - xn, 1)
            curres = np.linalg.norm(self.residual(xn), 1)
            hist[i] = curres

            if xtrue:
                hist[i] = np.linalg.norm(xn - self.xtrue, float("inf"))

            xcur = xn
            # print "curres: ", curres
            # print "tol: ", tol
            # print "curdiff: ", curdiff

            if curres <= tol or curdiff <= tol/10:
                print "breck................................."
                break

        hist = hist[1:i, :]
        ihist = ihist[:, 1:i]

        if i == niter and curres > tol:
            print 'tensorpr3:notConverged, did not converge after %i iterations to %e tolerance', niter, tol
            raise ValueError
            flag = 0
        else:
            flag = 1

        x = xcur / sum(xcur)

        return x, hist, flag, ihist

    def inverseiter(self, maxiter=1e3, tol=1e-8, xtrue=[], randinit=False, x0=[]):
        n = self.R.shape[0]
        niter, xcur, hist, ihist = self.parse_param(n, maxiter, tol, xtrue, randinit, x0)
        I = np.eye(n)

        for i in range(1, int(niter+1)):
            A = np.kron(xcur, I) + np.kron(I, xcur)
            A = I - self.alpha / 2 * self.R.dot(A.transpose())
            b = (1 - self.alpha) * self.v

            xn = inv(A).dot(b)
            # print xn
            xn = xn / np.linalg.norm(xn, 1)
            # print xn
            ihist[:, i] = xn
            curdiff = np.linalg.norm(xcur - xn, 1)
            curres = np.linalg.norm(self.residual(xn), 1)
            hist[i] = curres

            if xtrue:
                hist[i] = np.linalg.norm(xn - self.xtrue, float("inf"))
            xcur = xn
            if curres <= tol or curdiff <= tol/10:
                print "breck................................."
                break
        hist = hist[1:i, :]
        ihist = ihist[:, 1:i]

        if i == niter and curres > tol:
            print 'tensorpr3:notConverged, did not converge after %i iterations to %e tolerance', niter, tol
            raise ValueError
            flag = 0
        else:
            flag = 1

        x = xcur

        return x, hist, flag, ihist

    def newton(self, maxiter=1e3, tol=1e-8, xtrue=[], randinit=False, x0=[]):
        n = self.R.shape[0]
        niter, xcur, hist, ihist = self.parse_param(n, maxiter, tol, xtrue, randinit, x0)
        I = np.eye(n)

        for i in range(1, int(niter+1)):
            A = self.alpha * self.R.dot((np.kron(xcur, I) + np.kron(I, xcur)).transpose()) - I
            b = self.alpha * self.R.dot(np.kron(xcur, xcur)) - (1 - self.alpha) * self.v
            xn = inv(A).dot(b)
            print xn
            xn = xn / sum(xn)
            print xn
            print ihist.shape
            ihist[:, i] = xn
            curdiff = np.linalg.norm(xcur - xn, 1)
            curres = np.linalg.norm(self.residual(xn), 1)
            hist[i] = curres

            if xtrue:
                hist[i] = np.linalg.norm(xn - self.xtrue, float("inf"))
            xcur = xn
            if curres <= tol or curdiff <= tol/10:
                print "breck................................."
                break

        hist = hist[1:i, :]
        ihist = ihist[:, 1:i]

        if i == niter and curres > tol:
            print 'tensorpr3:notConverged, did not converge after %i iterations to %e tolerance', niter, tol
            raise ValueError
            flag = 0
        else:
            flag = 1

        x = xcur

        return x, hist, flag, ihist

    def innout(self, maxiter=1e3, tol=1e-8, xtrue=[], randinit=False, x0=[]):
        n = self.R.shape[0]
        niter, xcur, hist, ihist = self.parse_param(n, maxiter, tol, xtrue, randinit, x0)
        print ihist.shape
        # print (self.alpha * self.R).shape
        # print (np.ones((1, n**2))).shape
        # print self.v
        v2 = np.zeros((self.v.shape[0], 1))
        v2[:, 0] = self.v
        # print self.v
        # print v2
        # print (np.ones((1, n**2))).shape

        # print type(np.array(np.asmatrix(self.v)))
        # print np.array(np.asmatrix(self.v))
        # print np.array(np.asmatrix(self.v)).dot(np.ones(n**2))

        Rt = self.alpha * self.R + (1 - self.alpha) * v2.dot(np.ones((1, n**2)))
        at = self.alpha / 2
        x = xcur
        for i in range(1, int(niter+1)):
            Tr = POIRank(Rt, at, x)
            #xn = Tr.solve('tol', max(tol/10, spacing(1)))
            xn, hist_no_use, flag_no_use, ihist_no_use = Tr.solve()
            xn = xn / sum(xn)
            # print "xn: ", xn
            ihist[:, i] = xn
            curdiff = np.linalg.norm(xcur - xn, 1)
            curres = np.linalg.norm(self.residual(xn), 1)
            hist[i] = curres

            if xtrue:
                hist[i] = np.linalg.norm(xn - self.xtrue, float("inf"))
            x = xn
            if curres <= tol or curdiff <= tol/10:
                print "breck................................."
                break

        hist = hist[1:i, :]
        ihist = ihist[:, 1:i]

        if i == niter and curres > tol:
            print 'tensorpr3:notConverged, did not converge after %i iterations to %e tolerance', niter, tol
            raise ValueError
            flag = 0
        else:
            flag = 1

        return x, hist, flag, ihist


def rank(x, axis_poi):
    sort_data = []
    for index in range(0, len(x)):
        meta_data = (index, x[index])
        sort_data.append(meta_data)
    sort_data.sort(key=lambda x: x[1], reverse=True)
    res = []
    for i in range(0, len(sort_data)):
        res.append((i+1, axis_poi[sort_data[i][0]], sort_data[i][1]))

    return res


if __name__ == '__main__':
    matrix = [[[1/2, 1/2], [1/3, 2/3]], [[1/4, 3/4], [2/5, 3/5]]]
    alpha = 0.7
    v = [0.5, 0.5]
    x, hist, flag, ihist = POIRank(np.array(matrix), alpha, v).solve()
    print "hist: ", hist
    print "ihist: ", ihist
    print "flag: ", flag
    if flag == 1:
        print "收敛，x: ", x
    else:
        print "无法收敛"

    # poi_axis, axis_poi, data = init_data(tuple(range(0, 2)))   # (0, 182)
    #
    # tensor = trans(data, len(axis_poi))
    # alpha = 0.95
    # v = []
    # for i in range(0, len(poi_axis)):
    #     v.append(1/len(poi_axis))
    # x, hist, flag, ihist = POIRank(np.array(tensor), alpha, v).solve()
    # print "hist: ", hist
    # print "ihist: ", ihist
    # print "flag: ", flag
    # if flag == 1:
    #     print "收敛，x: ", x
    # else:
    #     print "无法收敛"
    #
    # result = rank(x, axis_poi)
    # sum = 0.0
    # for item in result:
    #     print item[0], item[1], item[2]
    #     sum += item[2]
    #
    # print "POIRank值的和： ", sum
    # print "data: ", data