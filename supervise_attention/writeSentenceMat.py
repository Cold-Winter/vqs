import json
import os
import numpy as np
import cv2
import lmdb
import caffe
from caffe.proto import caffe_pb2 
import math
from nltk.tokenize import TweetTokenizer
from gensim.models import word2vec
import shutil
tokenzer = TweetTokenizer()

model = word2vec.Word2Vec.load_word2vec_format('model.bin', binary=True)


feaFile = open('regionValOuestionList.txt')
count = 0
qMat = np.zeros((20, 300), dtype='float')
print qMat.shape
lmdb_file = 'regionQuestionValLmdb'
if os.path.exists(lmdb_file):
    shutil.rmtree(lmdb_file)
map_size = qMat.nbytes * 700000
print map_size
errorquestion = 0
errorword = 0
env = lmdb.open(lmdb_file, map_size=map_size)
count = 0
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for questionEn in feaFile:
        qMat = np.zeros((20, 300), dtype='float')
        if count % 10000 == 0:
            print 'process questions count '+ str(count)
        # answerlist[question['question_id']] = question['multiple_choices']
        #print questionEn
        question = questionEn[:-1].lower()
        qwordlist = tokenzer.tokenize(question)
        qFea = []
        qwCount=0
        for word in qwordlist:
            try:
                qMat[qwCount] = model[word]
            except:
                errorword += 1
            else:
                qwCount += 1
        if qwCount == 0:
            errorquestion += 1
        #print qMat
        #print x
        qMat = np.reshape(qMat,(1,20,300))
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
        count += 1
        #print x
print errorquestion
print count

feaFile.close()

