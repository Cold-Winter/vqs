#coding:utf-8
import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2 
import math
inputfile = open('testFeaNorm.txt')
imagelist = []

for line in open('testList.txt'):
    lines = line.split()[0].split('/')
    imagename = lines[-1][:-4]
    imageid=int(imagename.split('_')[-1])
    imagelist.append(imageid)
imageCount = 0
imagefeamap = {}
for line in open('feaTestPool5.txt'):
    if imageCount == len(imagelist):
        break
    lines = line.split()
    imagefeamap[imagelist[imageCount]] = lines
    imageCount += 1
    #print len(lines)
print len(imagelist)
print len(imagefeamap)
regionshotcount = 0

imagefeaarray = np.asarray(imagefeamap.values(),dtype=np.float32)
lmdb_file = 'vqaTestNormLmdb'
map_size = imagefeaarray.nbytes * 2000
print map_size

env = lmdb.open(lmdb_file, map_size=map_size)
count = 0


with env.begin(write=True) as txn:
    # txn is a Transaction object
    for line in inputfile:
        if count % 100000 == 0:
            print str(count) + ' data processed'
        veclist = line.split('\t')
        qlist = veclist[0].split()
        alist = veclist[1].split()
        ilist = imagefeamap[int(veclist[3])]
        feaSum = 0
        for fea in ilist:
            feaSum += float(fea)*float(fea)
        feaSum = math.sqrt(feaSum)
        for i in range(len(ilist)):
            ilist[i] = float(ilist[i])/feaSum
        fealist = qlist+ilist+alist
        #fealist = alist
        #print len(fealist)
        feaarray = np.asarray(fealist,dtype=np.float)
        x = np.reshape(feaarray,(2648,1,1))
        #print x
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = x.shape[0]
        datum.height = x.shape[1]
        datum.width = x.shape[2]
        datum.float_data.extend(x.flat)  # or .tostring() if numpy < 1.9
        # if veclist[2] == '0':
        #     if count % 2 == 0:
        #         continue
        datum.label = int(veclist[2])
        str_id = '{:08}'.format(count)
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        count += 1
