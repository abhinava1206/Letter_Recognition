#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:52:50 2018

@author: abhinavashriraam
"""
from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
def convert_to_greyscale(img):
    img = img/255.0
    def average(pixel):
        return (pixel[0]*0.21 + pixel[1]*0.72 + pixel[2]*0.07)
    greyImg = img[:,:,0].copy()
    for rownum in range(img.shape[0]):
        for colnum in range(img.shape[1]):
            greyImg[rownum][colnum] = average(img[rownum][colnum])
    return greyImg

def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)


def separate(img):
    orig_img = img.copy()
    pure_white = 255.
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped

big_img_a = plt.imread('as.jpg')
big_img_b = plt.imread('bs.jpg')
big_img_c = plt.imread('cs.jpg')
grey_big_img_a = convert_to_greyscale(big_img_a)
grey_big_img_b = convert_to_greyscale(big_img_b)
grey_big_img_c = convert_to_greyscale(big_img_c)
small_imgs_a = separate(grey_big_img_a)
small_imgs_b = separate(grey_big_img_b)
small_imgs_c = separate(grey_big_img_c)
del small_imgs_a[0]
del small_imgs_b[0]
del small_imgs_c[10]

for i in range(len(small_imgs_a)):
    small_imgs_a[i] = resize(small_imgs_a[i], (10,10))
    small_imgs_a[i] = small_imgs_a[i].reshape(100,)
    
for i in range(len(small_imgs_b)):
    small_imgs_b[i] = resize(small_imgs_b[i], (10,10))
    small_imgs_b[i] = small_imgs_b[i].reshape(100,)
    
for i in range(len(small_imgs_c)):
    small_imgs_c[i] = resize(small_imgs_c[i], (10,10))
    small_imgs_c[i] = small_imgs_c[i].reshape(100,)
    
big_list = []
for img in small_imgs_a:
    big_list.append(img)
    
for img in small_imgs_b:
    big_list.append(img)

for img in small_imgs_c:
    big_list.append(img)

T_a = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
T_b = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
T_c = np.array([2.,2.,2.,2.,2.,2.,2.,2.,2.,2.])
T_a = np.transpose(T_a)
T_b = np.transpose(T_b)
T_c = np.transpose(T_c)
T = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.])
T = np.transpose(T)

def partition(training_data, target_values, p):
    data_a = training_data[0:10]
    data_b = training_data[10:20]
    data_c = training_data[20:30]
    target_a = T[0:10]
    target_b = T[10:20]
    target_c = T[20:30]
    num = (10*p)/100
    num = int(num)
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    for i in range(0,num):
        train_data.append(data_a[i])
        train_target.append(target_a[i])
        train_data.append(data_b[i])
        train_target.append(target_b[i])
        train_data.append(data_c[i])
        train_target.append(target_c[i])
    for i in range(num, 10):
        test_data.append(data_a[i])
        test_target.append(target_a[i])
        test_data.append(data_b[i])
        test_target.append(target_b[i])
        test_data.append(data_c[i])
        test_target.append(target_c[i])
        
    return train_data, train_target, test_data, test_target

L = partition(big_list,T,80)


def svc_train(L):
    classification = svm.LinearSVC()    
    classification.fit(L[0],L[1])
    print "Predicted: ", classification.predict(L[2])
    print "Truth: ", L[3]
    count = 0.
    for i in range(len(L[3])):
        if L[3][i] == classification.predict(L[2])[i]:
            count += 1
        
    success = (count*100.0)/len(L[3])
    print "Accuracy: ", success, "%"
        
    
svc_train(L)        
    
    
    
    


