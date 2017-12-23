#!/usr/bin/env python

# note this is system version 4 --
# points of improvement..
# - uses bottleneck features from imagenet wining networks
# - prep data will create outputs for all known clases
#  -- this reduces the amount of disk waste and cpu time to generate inputs
# - training will work to compare all in one model vs individual class models

import numpy as np
import pandas as pd
import cv2
import os
import sys

import matplotlib.pyplot as plt
import sklearn.metrics

import test_models
import libs.data_flow
import libs.patch_generator
import libs.balancer
import libs.timing
import bottlenecks

import hashlib
import subprocess
import datetime
import glob

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

script_name = "./" + os.path.basename(sys.argv[0])

def apply_all(transform, data, patch_size, channels):
    out = transform.allocate(data.shape[0],patch_size,channels)

    for i in range(data.shape[0]):
        out[i] = transform(data[i])

    return out

        
class Trainer:
    def __init__(self, file, 
        train_necks,train_roi,
        valid_necks,valid_roi):

        self.file          = file
        self.train_necks = train_necks
        self.train_roi     = train_roi
        self.valid_necks = valid_necks
        self.valid_roi     = valid_roi
        
    def train(self, model):
        metrics = libs.metrics.Metrics()

        stopper = EarlyStopping(monitor='loss', patience=10, verbose=0)
        model_checkpoint = ModelCheckpoint(self.file, 
            monitor='loss', 
            save_best_only=True)

        hist = model.fit(self.train_necks, self.train_roi, 
                batch_size=64, 
                epochs=300, verbose=1, shuffle=True,
                callbacks=[model_checkpoint, stopper, metrics],
                validation_data=(self.valid_necks, self.valid_roi))

        print "final.. "
        print " confusion:"
        print metrics.confusion
        print " precision:", metrics.precision
        print " recall   :", metrics.recall
        print " f1s      :", metrics.f1s
        print " kappa    :", metrics.kappa
        print " auc      :", metrics.auc

def load_data(patch_size):
    mark = libs.timing.sync()

    print "loading data set.."

    train_necks   = np.load("prep/train_necks_size%d.npy" % (patch_size))
    train_roi     = np.load("prep/train_roi_size%d.npy" % (patch_size))
    valid_necks   = np.load("prep/valid_necks_size%d.npy" % (patch_size))
    valid_roi     = np.load("prep/valid_roi_size%d.npy" % (patch_size))

    mark = libs.timing.timing(mark)

    return train_necks,train_roi,valid_necks,valid_roi

def setup_data(patch_size, max_samples, validation_samples):
    if os.path.isfile("prep/train_necks_patch%d.npy" % (patch_size)):
        print "data exists... bye"
        return

    if not os.path.exists("prep"):
        os.makedirs("prep")

    # ok setup data flow
    mark = libs.timing.sync()

    idents   = np.load("raw/idents.npy") # meh.. duplicated data (in filenames/wkts file)
    channels = np.arange(10) + 1
    
    data_flow = libs.data_flow.DataFlowFromDisk(idents, channels,
                                                lambda i  : "raw/%s_3band_in.npz"     %  i,
                                                lambda i,c: "raw/%s_3band_chan%d.npz" % (i,c))

    gen = libs.patch_generator.DataGenerator(3,channels.shape[0],
                                             patchs_per_augmentation = 100, 
                                             patch_size              = patch_size)

    for tag,samples in {"train": max_samples, "valid": validation_samples}.items():
        # fulll blown mode... generate raw patches and we can inspect  them later
        print "generating data set for " + tag + "..."

        # capture the raw data..
        ipatch,opatch = gen(data_flow, epochs=1, samples=samples).next()

        mark = libs.timing.timing(mark)
        print "saving patches data..."
    
        
        np.save("prep/%s_in_size%d.npy"  % (tag, patch_size), ipatch)
        np.save("prep/%s_out_size%d.npy" % (tag, patch_size), opatch)

        mark = libs.timing.timing(mark)
        print "doing roi transforms..."
            
        transform_out = libs.patch_generator.RoiTransform()

        roi = apply_all(transform_out, opatch, patch_size, channels.shape[0])            
        np.save("prep/%s_roi_size%d.npy" % (tag, patch_size), roi)            
        del opatch
        del roi
        del transform_out 

        mark = libs.timing.timing(mark)
        print "doing necks transforms..."

        transform_in  = bottlenecks.BottlenecksTransform()

        necks = apply_all(transform_in, ipatch, patch_size, channels.shape[0])
        np.save("prep/%s_necks_size%d.npy" % (tag, patch_size), necks)
        del ipatch
        del necks
        del transform_in

        mark = libs.timing.timing(mark)


    # free some mem
    del gen
    del data_flow

    print "done..."

    #else:
    #    gen = libs.patch_generator.DataGenerator(3,channels.shape[0],
    #                                             patchs_per_augmentation = 100, 
    #                                             patch_size              = patch_size,
    #                                             transform_in  = bottlenecks.BottlenecksTransform(),
    #                                             transform_out = libs.patch_generator.RoiTransform())
    #    
    #    train_necks,train_roi = gen(data_flow, epochs=1, samples=max_samples       ).next()
    #    valid_necks,valid_roi = gen(data_flow, epochs=1, samples=validation_samples).next()
    #
    #    mark = libs.timing.timing(mark)
    # 
    #    # free some mem
    #    del gen
    #    del data_flow
    #
    #    # save the setup..
    #    save_data(patch_size,
    #              train_necks,train_roi,valid_necks,valid_roi)


def take_sample_point(hash_tag,
        ident, patch_size, samples):
    # setup dirs

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if not os.path.exists("weights"):
        os.makedirs("weights")

    # load data
    bestfile = "weights/%s_best.hdf5" % (hash_tag)
    print "trial run:", bestfile, " for samples:", samples 

    mark = libs.timing.sync()
    print "loading data..."

    train_necks,train_roi,valid_necks,valid_roi = load_data(patch_size)

    train_ratio = float(np.sum(train_roi)) / train_roi.shape[0]
    valid_ratio = float(np.sum(valid_roi)) / valid_roi.shape[0]

    print np.sum(train_roi), "/", train_roi.shape[0], " (", train_ratio, ")"
    print np.sum(valid_roi), "/", valid_roi.shape[0], " (", valid_ratio, ")"

    mark = libs.timing.sync()
    print "data rebalance and selection..."

    select_necks,select_roi = libs.balancer.multi_rebalance(train_necks, train_roi, samples)

    mark = libs.timing.sync()
    print "model build..."
    
    model_builder = test_models.models()[ident]
    model = model_builder(train_necks.shape[1])

    mark = libs.timing.timing(mark)
    print "training..."

    trainer = Trainer(bestfile,
        select_necks,select_roi,
        valid_necks,valid_roi)

    trainer.train(model)

    mark = libs.timing.timing(mark)
    print "predict..."

    predict_train_roi = model.predict(select_necks)
    predict_valid_roi = model.predict(valid_necks)

    mark = libs.timing.timing(mark)
    print "cleanup..."

    del trainer
    del model

    mark = libs.timing.timing(mark)
    print "measure..."

    train_loss = sklearn.metrics.log_loss(select_roi, predict_train_roi)
    valid_loss = sklearn.metrics.log_loss(valid_roi,  predict_valid_roi)

    train_mse = sklearn.metrics.mean_squared_error(select_roi, predict_train_roi)
    valid_mse = sklearn.metrics.mean_squared_error(valid_roi,  predict_valid_roi)

    # and some less accurate
    train_acc = sklearn.metrics.accuracy_score(select_roi, predict_train_roi > 0.5)
    valid_acc = sklearn.metrics.accuracy_score(valid_roi,  predict_valid_roi > 0.5)

    train_prec, train_recall, train_fscore, train_support = sklearn.metrics.precision_recall_fscore_support(select_roi, predict_train_roi > 0.5)
    valid_prec, valid_recall, valid_fscore, valid_support = sklearn.metrics.precision_recall_fscore_support(valid_roi,  predict_valid_roi > 0.5)

    # maybe this was silly. i just want to gather into into a pandas dataframe in the end
    curves = {
        "samples": samples,
        "tag": hash_tag,
        "model": {
            "patch_size": patch_size,
            "ident": ident,
        },
        "train": {
            "loss":    train_loss,
            "mse":     train_mse,
            "acc":     train_acc,
            "prec":    train_prec,
            "recall":  train_recall,
            "fscore":  train_fscore,
            "support": train_support
            },
        "valid": {
            "loss": valid_loss,
            "mse": valid_mse,
            "acc": valid_acc,
            "prec": valid_prec,
            "recall": valid_recall,
            "fscore": valid_fscore,
            "support": valid_support
        }     
    }

    # save results each step in case it dies mid point
    np.save("outputs/%s_metrics" % hash_tag, curves)

    print  curves
    mark = libs.timing.timing(mark)


def cmd_prep(patch_size, samples):
    if os.path.isfile("prep/train_necks_size%d.npy" % (patch_size)):
        print "data is aleady preped.."
        return
    
    cmdline = [script_name, 
        "-m", "prep", 
        "-s", str(samples),
        "-p", str(patch_size) ]

    print cmdline
    try:
        cmd = subprocess.Popen(
            cmdline,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for stdout_line in iter(cmd.stdout.readline, ""):
            print stdout_line,

        cmd.stdout.close()
        return_code = cmd.wait()
    except Exception as exception:
        print('Exception occured: ' + str(exception))
        return False
    return True

def cmd_sample(ident, patch_size, samples, instance = 0):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # then run a sample
    tag = ident                         \
        + "_" + str(patch_size)         \
        + "_" + str(samples)            \
        + "_" + str(instance)
    hash_tag = hashlib.sha1(tag).hexdigest()
 
    if os.path.isfile("outputs/%s_metrics.npy" % hash_tag):
        print "sample:", hash_tag, "exists.. skiping"
        return

    cmdline = [script_name,
        "-m", "_sample", 
        "-t", hash_tag,
        "-s", str(samples),
        "-p", str(patch_size),
        "-i", ident ]

    print "Exection:", cmdline

    try:
        cmd = subprocess.Popen(
            cmdline,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with open("logs/%s.log" % hash_tag, "w") as fh:
            for stdout_line in iter(cmd.stdout.readline, ""):
                fh.write(stdout_line)
                print stdout_line,

        cmd.stdout.close()
        return_code = cmd.wait()

    except Exception as exception:
        print('Exception occured: ' + str(exception))
        return False

    return True

def flatten(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten(value).items():
                    yield key + "." + subkey, subvalue
            else:
                yield key, value

    return dict(items())

def cmd_gather():
    # ok search all sample points gather and build into one
    df = None
    for file in glob.glob("outputs/*_metrics.npy"):
        print(file)
        sample = np.load(file).item()
        sample = flatten(sample)

        if df is None:
            df = pd.DataFrame(columns=sample.keys())

        df = df.append(sample, ignore_index=True)

    #df.to_csv("outputs/all_metrics.csv")
    df.to_pickle("outputs/all_metrics.pickle") 

if __name__ == "__main__":
    # ./trainer.py -m _sample -t testtesttest -s 20 -p 100 -i test_model1
    # ./trainer.py -m _sample -tbf25c1041cfe38c6a33b8dd2de17c14b444a8b37 -s 20 -p 100 -i test_model1
    import argparse

    parser = argparse.ArgumentParser()  
    parser.add_argument("-m", "--mode",       help="mode [master, sample, prep]")
    parser.add_argument("-p", "--patch_size", help="patch size of input")
    parser.add_argument("-s", "--samples",    help="samples to train with")
    parser.add_argument("-t", "--tag",        help="training sample tag")
    parser.add_argument("-i", "--model",      help="patch size of input")

    args = parser.parse_args()      

    hash_tag = None
    ident = None
    patch_size = 100

    validation_samples = 10000
    samples = 16000  

    if not args.patch_size is None:
        patch_size = int(args.patch_size)
    if not args.samples is None:
        samples = int(args.samples)
    if not args.model is None:
        ident = args.model
    if not args.tag is None:
        hash_tag = args.tag

    if args.mode == "prep":
        # build prep data from the raw data..
        setup_data(patch_size, samples, validation_samples)

    elif args.mode == "_sample":
        # not recoded.. use other one
        take_sample_point(hash_tag, ident, patch_size, samples)

    elif args.mode == "sample":
        cmd_sample(ident, patch_size, samples)

    elif args.mode == "gather":
        # gather and merge current data points into one file..
        cmd_gather()
 
    elif args.mode == "master":
        # TODO.. MCMC bayesian grid search?
        
        # main data sample routine.. find a hole in the data and fill it
        sample_set = [10,20,50,100,200,500,1000,2000,5000,10000,14000, 20000,30000,50000] 

        for ident in sorted(test_models.models().keys()):
            for sample in sample_set:
                cmd_sample(ident, patch_size, sample)


