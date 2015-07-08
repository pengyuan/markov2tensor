#!/usr/bin/env python
# coding: UTF-8

from pymatbridge import Matlab
mlab = Matlab(executable='/Applications/MATLAB_R2014a.app/bin/matlab')

mlab.start()

results = mlab.run_code('a=1;')

var = mlab.get_variable('a')

print var
mlab.stop()