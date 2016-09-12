#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

import images2gif
from images2gif import writeGif

import tensorflow as tf

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return np.sum(ll)


def filter_raw(raw):
    # delete all bad GPS stuff its making a mess
    xy = raw[:,2:4]
    xy = xy.astype(float)

    badx_idx = np.logical_or(xy[:,0] < -123, xy[:,0] > -122)
    bady_idx = np.logical_or(xy[:,1] < 30, xy[:,1] > 40)

    idx = np.logical_not(np.logical_or(badx_idx, bady_idx))

    return raw[idx]


def xy_clean(xy):
    xy = xy.astype(float)
    means = np.mean(xy,axis=0)

    badx_idx = np.logical_or(xy[:,0] < -123, xy[:,0] > -122)
    xy[badx_idx,0] = means[0]

    bady_idx = np.logical_or(xy[:,1] < 30, xy[:,1] > 40)
    xy[bady_idx,1] = means[1]

    xy_max = np.amax(xy,axis=0)
    xy_min = np.amin(xy,axis=0)

    return (xy - xy_min)/(xy_max - xy_min)

def date_clean(d):
    d = d.astype(np.datetime64)

    d_max = np.amax(d,axis=0)
    d_min = np.amin(d,axis=0)

    d =  (d - d_min)/(d_max - d_min)

    return d.astype(float).reshape((d.shape[0], 1))

def labels_to_one_hots(labels):
    # find the unique class labels
    num_labels = labels.shape[0]
    classes = np.unique(labels)
    num_classes = classes.shape[0]

    # create index of label into the hots..
    #   ie covert the label to a number indexing into classes
    label_index = np.zeros(labels.shape[0], dtype=int)
    idx = 0
    for c in classes:
        label_index[labels == c] = idx
        idx += 1

    # compute the offsets into the one_hots (if it was flat)
    hot_indexes = np.arange(num_labels) * num_classes + label_index

    # and then write each item that is a hot as a hot
    one_hots = np.zeros((num_labels, num_classes))
    one_hots.flat[hot_indexes] = 1

    return classes, one_hots

def raw_read_data(trainfile):
    cats = {}
    dists = {}

    raw = []
    with open(trainfile) as csvfile:
        reader = csv.DictReader(csvfile)

        #print raw
        for row in reader:
            cat   = row['Category']
            dist  = row['PdDistrict']
            X     = row['X']
            Y     = row['Y']
            hours = row['Hour']
            crossroads = row['Crossroad']
            dow   = row['DayOfWeek']
            dtime = row['Dates']
            raw.append([cat, dist, X, Y, hours, crossroads, dow, dtime])

    raw =  np.array(raw)

    print "raw size:", raw.shape
    print "raw:", raw

    return raw

def select_data(raw):
    size_train = int(raw.shape[0] * 0.9)
    print "spliting at: ", size_train

    pd_labels,  pds  = labels_to_one_hots(raw[:,1])
    dow_labels, dow  = labels_to_one_hots(raw[:,6])

    cat_labels, cats = labels_to_one_hots(raw[:,0])

    # inputs = np.concatenate((pds, raw[:,2:4]), axis=1)
    # inputs = raw[:,2:4].astype(float)
    # inputs = raw[:,4:6].astype(float)
    # inputs = raw[:,4:6].astype(float)

    xys = xy_clean(raw[:,2:4])
    ds  = date_clean(raw[:,7])

    print xys.shape, ds.shape

    inputs = np.concatenate((xys,ds), axis=1)

    #inputs = np.concatenate((pds, raw[:,4:6].astype(float)), axis=1)
    #inputs = np.concatenate((pds,                                   \
    #                         raw[:,4:6].astype(float),             \
    #                         xy_clean(raw[:,2:4]),
    #                         dow), axis=1)

    return inputs[:size_train,:], inputs[size_train:,:], \
           cats[:size_train,:],   cats[size_train:,:], \
           pd_labels, dow_labels, cat_labels \

def raw_read_submission(testfile):
    cats = {}
    dists = {}

    raw = []
    with open(testfile) as csvfile:
        reader = csv.DictReader(csvfile)

        #print raw
        for row in reader:
            ident = row['Id']
            dist  = row['PdDistrict']
            X     = row['X']
            Y     = row['Y']
            hours = 0
            crossroads = 0
            dow   = row['DayOfWeek']
            dtime = row['Dates']
            raw.append([ident, dist, X, Y, hours, crossroads, dow, dtime])

    raw =  np.array(raw)

    print "raw size:", raw.shape
    print "raw:", raw

    return raw

def select_submission(raw_sub):
    idents = raw_sub[:,0].reshape((raw_sub.shape[0],1)).astype(int)
    xys = xy_clean(raw_sub[:,2:4])
    ds  = date_clean(raw_sub[:,7])

    print idents.shape, xys.shape, ds.shape

    inputs = np.concatenate((xys,ds,idents), axis=1)

    return inputs

def model(class1, test1, class2, test2):
    iwidth = class1.shape[1]
    owidth = class2.shape[1]

    print "debug:", iwidth, owidth

    # model 1
    # x  = tf.placeholder("float", [None, iwidth])
    # y_ = tf.placeholder("float", [None,owidth])
    #
    # W = tf.Variable(tf.random_normal([iwidth,owidth], stddev=0.35))
    # b = tf.Variable(tf.zeros([owidth]))
    # y = tf.nn.softmax(tf.matmul(x,W) + b)

    # model 2
    hidden = 50
    x  = tf.placeholder("float", [None, iwidth])
    y_ = tf.placeholder("float", [None,owidth])

    W1 = tf.Variable(tf.random_normal([iwidth,hidden], stddev=0.35))
    b1 = tf.Variable(tf.zeros([hidden]))

    W2 = tf.Variable(tf.random_normal([hidden,owidth], stddev=0.35))
    b2 = tf.Variable(tf.zeros([owidth]))

    h = tf.nn.relu(tf.matmul(x,W1) + b1)
    y = tf.nn.softmax(tf.matmul(h,W2) + b2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01) \
         .minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    for i in xrange(100000):
        step = 1000
        select = i*step % class1.shape[0]
        send   = select + step

        # step  = 10
        # select = 0
        # send   = select + step

        if send > class1.shape[0]:
            send = class1.shape[0]

        batch_xs = class1[select:send, :]
        batch_ys = class2[select:send, :]

        # print "selection:", select, send
        # print batch_xs
        # print batch_ys

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 100 == 0:
            print "step:", i,
            print " accuracy:", sess.run(accuracy, feed_dict={x: test1, y_: test2}),
            #print "        batch:", sess.run((accuracy, y), feed_dict={x: batch_xs, y_: batch_ys})
            print " batch   :", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})


def render_map(class1, class2, cat_labels):
    # de-one hot
    ident2 = np.argmax(class2, axis=1).reshape((class2.shape[0],1))

    for c in range(cat_labels.shape[0]-1):
        # 2003 - 2014 (10years) 3 month steps .. 30,
        idx = (ident2 == c).reshape(ident2.shape[0])
        selection = class1[idx]

        print idx.shape
        print class1.shape
        print class2.shape
        print selection.shape

        print selection

        images = np.zeros((31,101,101))
        for i in range(selection.shape[0]):
            t = int(30  *selection[i,2])
            x = int(100*selection[i,0])
            y = int(100*selection[i,1])

            images[t,x,y]  += 1
            images[30,x,y] += 1  # all images

        #animate gif
        label = cat_labels[c]
        label = re.sub(' ', '_', label)
        label = re.sub('/', '_', label)
        label = "labels/" + label + ".gif"
        writeGif(label, images,duration = 0.3)

        for t in range(31):
            label = cat_labels[c]
            label = re.sub(' ', '_', label)
            label = re.sub('\/', '_', label)
            if t == 30:
                label = "labels/" + label + "_all.png"
            else:
                label = "labels/" + label  + "_time_" + str(t) + ".png"

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if t == 30:
                ax.set_title(cat_labels[c] + " @ all")
            else:
                ax.set_title(cat_labels[c] + " @ time:" + str(t))
            y_dim = images.shape[2]
            x_dim = images.shape[1]
            im = images[t,:,:].reshape(y_dim, x_dim)
            #ax.imshow(im, cmap='Greys', interpolation='nearest')
            ax.imshow(im, cmap='gist_heat', interpolation='nearest')
            ax.axis('off')
            plt.savefig(label)
            plt.close(fig)


def create_tfidf(class1, class2, cat_labels):
    # de-one hot
    ident2 = np.argmax(class2, axis=1).reshape((class2.shape[0],1))

    all = np.ones((101,101))
    cats = np.ones((cat_labels.shape[0], 101,101))

    for c in range(cat_labels.shape[0]-1):
        # 2003 - 2014 (10years) 3 month steps .. 30,
        idx = (ident2 == c).reshape(ident2.shape[0])
        selection = class1[idx]

        images = np.zeros((31,101,101))
        for i in range(selection.shape[0]):
            #t = int(30  *selection[i,2])
            x = int(100*selection[i,0])
            y = int(100*selection[i,1])

            #images[t,x,y]  += 1
            cats[c,x,y] += 1  # all images
            all[x,y] += 1

    tfidf = np.log(all) - np.log(all/cats)

    for c in range(cat_labels.shape[0]-1):
        label = cat_labels[c]
        label = re.sub(' ', '_', label)
        label = re.sub('\/', '_', label)
        label = "tfidf/" + label + "_tfidfl.png"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(cat_labels[c] + " @ tfidf")
        y_dim = images.shape[2]
        x_dim = images.shape[1]
        im = tfidf[c,:,:].reshape(y_dim, x_dim)
        ax.imshow(im, cmap='gist_heat', interpolation='nearest')
        ax.axis('off')
        plt.savefig(label)
        plt.close(fig)


    return tfidf

def create_txy_tfidf(class1, class2, cat_labels):
    # de-one hot
    ident2 = np.argmax(class2, axis=1).reshape((class2.shape[0],1))

    all = np.ones((101,101))
    cats = np.ones((cat_labels.shape[0], 31, 101,101))

    for c in range(cat_labels.shape[0]-1):
        # 2003 - 2014 (10years) 3 month steps .. 30,
        idx = (ident2 == c).reshape(ident2.shape[0])
        selection = class1[idx]

        images = np.zeros((31,101,101))
        for i in range(selection.shape[0]):
            t = int(30  *selection[i,2])
            x = int(100*selection[i,0])
            y = int(100*selection[i,1])

            #images[t,x,y]  += 1
            cats[c,t,x,y] += 1  # all images
            all[x,y] += 1

    tfidf = np.log(all) - np.log(all/cats)

    for c in range(cat_labels.shape[0]-1):
        for t in range(30):
            label = cat_labels[c]
            label = re.sub(' ', '_', label)
            label = re.sub('\/', '_', label)
            label = "tfidf_txy/" + label  + "_time_" + str(t) + ".png"

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(cat_labels[c] + " @ time:" + str(t))
            y_dim = tfidf.shape[3]
            x_dim = tfidf.shape[2]
            im = tfidf[c,t,:,:].reshape(y_dim, x_dim)
            ax.imshow(im, cmap='gist_heat', interpolation='nearest')
            ax.axis('off')
            plt.savefig(label)
            plt.close(fig)

    return tfidf


def tfidf_model(tfidf, test1, test2):
    exp_tfidf = np.exp(tfidf)
    softmax_sum = np.sum(exp_tfidf, axis=0)
    softmax = exp_tfidf / softmax_sum

    most_likely = np.argmax(softmax,axis=0)

    label = "tfidf/most_likely_tfidfl.png"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("most likely tfidf")
    y_dim = most_likely.shape[1]
    x_dim = most_likely.shape[0]
    im = most_likely.reshape(y_dim, x_dim)
    ax.imshow(im, cmap='Set1', interpolation='nearest')
    ax.axis('off')
    plt.savefig(label)
    plt.close(fig)

    results = np.zeros((test1.shape[0], tfidf.shape[0]))
    for i in range(test1.shape[0]):
        x = int(100*test1[i,0])
        y = int(100*test1[i,1])
        res = softmax[:,x,y]

        results[i,:] = res

        # print test1[i,:], "-> e:", np.argmax(test2[i,:]), " a:", np.argmax(res),
        # print res

    if (not test2 is None):
        correct_prediction = np.argmax(results,axis=1) == np.argmax(test2,axis=1)
        accuracy = float(np.sum(correct_prediction)) / float(correct_prediction.shape[0])

        lloss = logloss(test2,results)

        print " correct:", np.sum(correct_prediction), "/", correct_prediction.shape[0]
        print " tfidf accuracy:", accuracy, "log loss", lloss

    return results

def tfidf_txy_model(tfidf_txy, test1, test2):
    exp_tfidf_txy = np.exp(tfidf_txy)
    softmax_sum = np.sum(exp_tfidf_txy, axis=0)
    softmax = exp_tfidf_txy / softmax_sum

    results = np.zeros(test2.shape)
    for i in range(test1.shape[0]):
        t = int(30 *test1[i,2])
        x = int(100*test1[i,0])
        y = int(100*test1[i,1])
        res = softmax[:,t,x,y]

        results[i,:] = res

        # print test1[i,:], "-> e:", np.argmax(test2[i,:]), " a:", np.argmax(res),
        # print res

    correct_prediction = np.argmax(results,axis=1) == np.argmax(test2,axis=1)
    accuracy = float(np.sum(correct_prediction)) / float(correct_prediction.shape[0])

    lloss = logloss(test2,results)

    print " correct:", np.sum(correct_prediction), "/", correct_prediction.shape[0]
    print " tfidf_txy accuracy:", accuracy, "log loss", lloss



def generate_sub(tfidf, sub1, cat_labels):
    results = tfidf_model(tfidf, sub1, None)

    idents = sub1[:,3].astype(int)

    with open('output.csv', 'wb') as outcsv:
        writer = csv.writer(outcsv)
        hdr = np.concatenate((['Id'],cat_labels))
        writer.writerow(hdr)

        for i in range(idents.shape[0]):
            row = [idents[i]] + results[i, :].tolist()
            writer.writerow(row)

def main():
    raw = raw_read_data('train_with_hour_crossroad.csv')

    raw = filter_raw(raw)

    class1, test1, class2, test2, \
    pd_labels, dow_labels, cat_labels = select_data(raw)

    print "labels:", pd_labels, dow_labels, cat_labels
    print
    print "inputs:", class1.shape, test1.shape,
    print class1, test1
    print
    print "outputs:", class2.shape, test2.shape
    print class2, test2

    model(class1, test1, class2, test2)


    ## tfidf model

    tfidf = create_tfidf(class1, class2, cat_labels)

    tfidf_model(tfidf, test1, test2)

    raw_sub = raw_read_submission("test.csv")
    sub1 = select_submission(raw_sub)

    generate_sub(tfidf, sub1, cat_labels)
