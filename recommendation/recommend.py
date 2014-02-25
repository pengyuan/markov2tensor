#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
from numpy.core.numeric import array
from util.hosvd import unfold, HOSVD
from util.tensor import trans, count, init_db
                    
#第一步：构建转移概率tensor               
tensor = trans(count(init_db()))
print "构建转移概率张量：",tensor

#第二步：HOSVD，重构tensor
threshold = 0.8
tensor = array(tensor)
#print unfold(tensor,1)

U, S, D = HOSVD(tensor)




print U[0]
print U[1]
print U[2]




print unfold(S, 1)

print "The n-mode singular values are:"
print D[0]
print D[1]
print D[2]