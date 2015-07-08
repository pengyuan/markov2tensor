#!/usr/bin/env python
# coding: UTF-8

# 与state of the art方法进行对比：
# 1.Factorized Markov Chain（FMC）：二阶马尔科夫链进行hosvd分解，没有用户之间和时间之间的信息协同
# 2.Tersor Factorization（TF）：基于用户-时间-地点的频数张量进行hosvd分解，没有用户的转移信息（地点对地点的评分）