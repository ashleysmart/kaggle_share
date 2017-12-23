#!/usr/bin/env python

# ok having model problems its time to do some learning cuves and get an idea of what
# a good model myght be,, ModelTestBed.ipynb gave me a quick idea.. but not enogh info..

# SETUP
# ln -s ../per_class/prep

import numpy as np
import pandas as pd
import cv2
import os
import sys

import matplotlib.pyplot as plt
import sklearn.metrics

import test_models
import libs.patch_generator
import libs.balancer
import libs.timing

import hashlib
import subprocess
import datetime
import glob

from keras.callbacks import ModelCheckpoint, EarlyStopping

#script_name = "./roi_learning_curve.py"
script_name = "./" + os.path.basename(sys.argv[0])

class Trainer:
    def __init__(self, file, 
        train_patches,train_roi,
        valid_patches,valid_roi):

        self.file          = file
        self.train_patches = train_patches
        self.train_roi     = train_roi
        self.valid_patches = valid_patches
        self.valid_roi     = valid_roi
        
    def train(self, model):
        metrics = libs.metrics.Metrics()

        stopper = EarlyStopping(monitor='loss', patience=10, verbose=0)
        model_checkpoint = ModelCheckpoint(self.file, 
            monitor='loss', 
            save_best_only=True)

        hist = model.fit(self.train_patches, self.train_roi, 
                batch_size=4, 
                epochs=50, verbose=1, shuffle=True,
                callbacks=[model_checkpoint, stopper, metrics],
                validation_data=(self.valid_patches, self.valid_roi))

        print "final.. "
        print " confusion:"
        print metrics.confusion
        print " precision:", metrics.precision
        print " recall   :", metrics.recall
        print " f1s      :", metrics.f1s
        print " kappa    :", metrics.kappa
        print " auc      :", metrics.auc

def load_data(chan, patch_size):
    mark = libs.timing.sync()

    print "loading data set.."

    train_patches = np.load("prep/curves_train_patches_chan%d_patch%d.npy" % (chan,patch_size))
    train_roi     = np.load("prep/curves_train_roi_chan%d_patch%d.npy" % (chan,patch_size))
    valid_patches = np.load("prep/curves_valid_patches_chan%d_patch%d.npy" % (chan,patch_size))
    valid_roi     = np.load("prep/curves_valid_roi_chan%d_patch%d.npy" % (chan,patch_size))

    mark = libs.timing.timing(mark)

    return train_patches,train_roi,valid_patches,valid_roi

def save_data(chan, patch_size,
    train_patches,train_roi,valid_patches,valid_roi):

    np.save("prep/curves_train_patches_chan%d_patch%d.npy" % (chan,patch_size), train_patches)
    np.save("prep/curves_train_roi_chan%d_patch%d.npy"     % (chan,patch_size), train_roi)
    np.save("prep/curves_valid_patches_chan%d_patch%d.npy" % (chan,patch_size), valid_patches)
    np.save("prep/curves_valid_roi_chan%d_patch%d.npy"     % (chan,patch_size), valid_roi)

def setup_data(chan, patch_size, max_samples, validation_samples):
    mark = libs.timing.sync()
    if os.path.isfile("prep/curves_train_patches_chan%d_patch%d.npy" % (chan,patch_size)):
        print "data exists... bye"
        return

    print "generating data set.."

    imgs = np.load("../per_class/prep/3band_in.npy")
    mask = np.load("../per_class/prep/3band_chan%d.npy" % chan)  # trees

    print imgs.shape
    print mask.shape

    gen = libs.patch_generator.DataGenerator(imgs,mask,
                                             patchs_per_transform = 100, 
                                             patch_size           = patch_size,
                                             mode                 = libs.patch_generator.DataGenerator.ROI)


    train_patches,train_roi = gen(epochs=1, samples=max_samples       ).next()
    valid_patches,valid_roi = gen(epochs=1, samples=validation_samples).next()

    mark = libs.timing.timing(mark)

    # free some mem
    del gen
    del imgs
    del mask

    # save the setup..
    save_data(chan,patch_size,
        train_patches,train_roi,valid_patches,valid_roi)



def take_sample_point(hash_tag,
        ident, chan, patch_size, samples):
    # load data
    bestfile = "weights/%s_best.hdf5" % (hash_tag)
    print "trial run:", bestfile, " for samples:", samples 

    mark = libs.timing.sync()
    print "loading data..."

    train_patches,train_roi,valid_patches,valid_roi = load_data(chan, patch_size)

    train_ratio = float(np.sum(train_roi)) / train_roi.shape[0]
    valid_ratio = float(np.sum(valid_roi)) / valid_roi.shape[0]

    print np.sum(train_roi), "/", train_roi.shape[0], " (", train_ratio, ")"
    print np.sum(valid_roi), "/", valid_roi.shape[0], " (", valid_ratio, ")"

    mark = libs.timing.sync()
    print "data rebalance and selection..."

    select_patches,select_roi = libs.balancer.rebalance(train_patches, train_roi, samples, 0.5)

    mark = libs.timing.sync()
    print "model build..."
    
    model_builder = test_models.models()[ident]
    model = model_builder(patch_size, 3)

    mark = libs.timing.timing(mark)
    print "training..."

    trainer = Trainer(bestfile,
        select_patches,select_roi,
        valid_patches,valid_roi)

    trainer.train(model)

    mark = libs.timing.timing(mark)
    print "predict..."

    predict_train_roi = model.predict(select_patches)
    predict_valid_roi = model.predict(valid_patches)

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
        "chan": chan,
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


def cmd_prep(chan, patch_size, samples):
    if os.path.isfile("prep/curves_train_patches_chan%d_patch%d.npy" % (chan,patch_size)):
        print "data is aleady preped.."
        return

    cmdline = [script_name, 
        "-m", "prep", 
        "-c", str(chan),         
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

def cmd_sample(ident, chan, patch_size, samples):
    # then run a sample
    tag = ident                         \
        + "_" + str(chan)               \
        + "_" + str(patch_size)         \
        + "_" + str(samples)
    hash_tag = hashlib.sha1(tag).hexdigest()
 
    if os.path.isfile("outputs/%s_metrics.npy" % hash_tag):
        print "sample:", hash_tag, "exists.. skiping"
        return

    cmdline = [script_name,
        "-m", "_sample", 
        "-t", hash_tag,
        "-c", str(chan), 
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
    import argparse

    parser = argparse.ArgumentParser()  
    parser.add_argument("-m", "--mode",       help="mode [master, sample, prep]")
    parser.add_argument("-c", "--channel",    help="channel of data")
    parser.add_argument("-p", "--patch_size", help="patch size of input")
    parser.add_argument("-s", "--samples",    help="samples to train with")
    parser.add_argument("-t", "--tag",        help="training sample tag")
    parser.add_argument("-i", "--model",      help="patch size of input")

    args = parser.parse_args()      

    hash_tag = None
    ident = None
    chan = 10
    patch_size = 100

    validation_samples = 10000
    samples = 16000  

    if not args.channel is None:
        chan = int(args.channel)
    if not args.patch_size is None:
        patch_size = int(args.patch_size)
    if not args.samples is None:
        samples = int(args.samples)
    if not args.model is None:
        ident = args.model
    if not args.tag is None:
        hash_tag = args.tag

    if args.mode == "prep":
        # then build data..
        setup_data(chan, patch_size, samples, validation_samples)

    elif args.mode == "_sample":
        # not recoded.. use other one
        take_sample_point(hash_tag, ident, chan, patch_size, samples)

    elif args.mode == "sample":
        cmd_sample(ident, chan, patch_size, samples)

    elif args.mode == "gather":
        # gather and merge current data points into one file..
        cmd_gather()
 
    elif args.mode == "master":
        # main data sample routine.. find a hole in the data and fill it
        sample_set = [2,5,10,20,50,100,200,500,1000,2000,5000,10000,14000]

        # first prep data sets
        for i in range(10):
            # note.. start with cars channel..
            chan = i + 1
            cmd_prep(chan, patch_size, samples)           

        for i in range(10):
            chan = (i + 9) % 10 + 1
            for ident in sorted(test_models.models().keys()):
                for sample in sample_set:
                    cmd_sample(ident, chan, patch_size, sample)


