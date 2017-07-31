#!/usr/bin/env sh
set -e
# set GLOG_logtostderr=0
# set GLOG_log_dir=./vqaScripts
/data/liyandong/caffe/caffe/build/tools/caffe train --solver=mlpVqa_solver.prototxt  2>&1 | tee log/mlpVqalr6S20G02w9r05M9ConcatMix.log
