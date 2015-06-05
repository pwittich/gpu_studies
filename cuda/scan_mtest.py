#! /usr/bin/env python

import shutil
import subprocess 
#import matplotlib as plt


nmat = 1

values = []

def convert(val):
    lookup = { 'ns' : 1e-3, 'us' : 1, 'ms' : 1000, 's' : 1000000 }
    unit = val[-2:]
    try:
        number = float(val[:-2])
    except ValueError:
        print 'exception!', val[:-2]
        return val
    print unit
    if unit in lookup:
        return lookup[unit]*number
    return int(val)
        




while ( nmat < 1e9 ) :
    print "nmat = ", nmat
    ret = subprocess.check_output(["nvprof", "./mtest", str(nmat)], stderr=subprocess.STDOUT)
    ret = ret.splitlines()
    matches = filter(lambda x:'smallMatrix' in x, ret)
    print "matches >> ", matches
    #print len(ret)
    elem  = matches[0].split()
    #values.append([float(elem[4]),float(elem[5])])
    values.append([nmat, convert(elem[3]),convert(elem[4])])
    convert(elem[3])
    # shutil.copyfile('final.dat','final_%s_%s.dat' % (nthreads, nblocks))
    # try:
    #     shutil.move('cuda_profile_0.log', 'profile_%s_%s.log' % ( nthreads, nblocks))
    # except IOError as err:
    #     print 'copy failed: ', err
    nmat = nmat * 10


print '\n\n\n\n'
print values
print '\n\n\n\n'

for v in values:
    for fv in v:
        print fv, '\t',
    print 

