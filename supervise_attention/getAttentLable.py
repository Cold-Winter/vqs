# coding: utf-8
import os
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
import cv2
import lmdb
import caffe
import shutil

inputfile= open('regionTrainList.txt')

qMat = np.zeros((14, 14), dtype='float')
print qMat.shape
lmdb_file = 'vqsTrainLableLmdb'
if os.path.exists(lmdb_file):
    shutil.rmtree(lmdb_file)
map_size = qMat.nbytes * 70000000
print map_size
errorquestion = 0
errorword = 0
env = lmdb.open(lmdb_file, map_size=map_size)

count = 0
env = lmdb.open(lmdb_file, map_size=map_size)
count = 0
with env.begin(write=True) as txn:
    for line in inputfile:
        lines = line.split()[0].split('/')
        imagename = lines[-1][:-6]+'.jpg'+'_'+lines[-1][-5]+'.mat'
        if count % 10000==0:
            print count
        count += 1  
        ground_mask = sio.loadmat('./train_ground/'+imagename)['ground_mask']
        resizeMask = cv2.resize(ground_mask,(14,14))
        # countPixel = 0
        # for h in range(14):
        #     for w in range(14):
        #         if resizeMask[h,w] == 1:
        #             countPixel+=1
        #print countPixel

        qMat = np.asarray(resizeMask, dtype='float')


        #qMat = np.zeros((14,14), dtype='float')
        #qMatSum=0
        # for i in range(14):
        #     for j in range(14):
        #         countReg=0
        #         for h in range(i*32,(i+1)*32):
        #             for w in range(j*32,(j+1)*32):
        #                 if resizeMask[h,w]==1:
        #                     countReg+=1
        #         qMat[i,j]=float(countReg)/countPixel
        #         qMatSum+=qMat[i,j]
        #print qMatSum


        qMat = np.reshape(qMat,(196,1,1))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = qMat.shape[0]
        datum.height = qMat.shape[1]
        datum.width = qMat.shape[2]
        datum.float_data.extend(qMat.flat)  # or .tostring() if numpy < 1.9
        # if veclist[2] == '0':
        #     if count % 2 == 0:
        #         continue
        datum.label = 0
        str_id = '{:08}'.format(count)
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

        
        
