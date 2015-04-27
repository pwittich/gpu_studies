#! /usr/bin/env python

import shutil
import subprocess 
#import matplotlib as plt

nthreads = 1024
nblocks = 16

values = []

print "nparticles = ", nthreads*nblocks

while ( nthreads >= 32 ) :
    print "threads, blocks = ", nthreads, nblocks, nthreads*nblocks
    ret = subprocess.check_output(["./nbody_nothrust", str(nthreads), str(nblocks)])
    ret = ret.splitlines()
    matches = filter(lambda x:'Average' in x, ret)
    print "matches >> ", matches
    #print len(ret)
    elem  = matches[0].split()
    values.append([float(elem[-4]),float(elem[-2])])
    shutil.copyfile('final.dat','final_%s_%s.dat' % (nthreads, nblocks))
    try:
        shutil.move('cuda_profile_0.log', 'profile_%s_%s.log' % ( nthreads, nblocks))
    except IOError as err:
        print 'copy failed: ', err
    nthreads = nthreads/2
    nblocks = nblocks * 2

print values
