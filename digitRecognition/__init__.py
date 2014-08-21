import numpy as np
import cv2
import utils
import os

FEATURE = 0
CLASS = 1

imgs_train = [ [], [] ]
for i in range( 10 ):
    for r, d, f in os.walk( "digitos/train/digit_" + str( i ) ):
        for img in f:
            img = cv2.imread( img, 0 )  
            ret, img = cv2.threshold( img, 127, 255, cv2.THRESH_BINARY )
            img_feature = utils.apply4CC(img) #extraccion del histograma, deberia ser una funcion de img
            hist = cv2.calcHist([img_feature], [0], None,[16],[0,16])
            imgs_train[FEATURE].append( hist )#histograma
            imgs_train[CLASS].append( i )#clase
        

imgs_test = [ [],[] ]
for i in range( 10 ):
    for r, d, f in os.walk( "digitos/test/digit_" + str( i ) ):
        for img in f:
            img = cv2.imread( img, 0 )  
            ret, img = cv2.threshold( img, 127, 255, cv2.THRESH_BINARY )
            img_feature = utils.apply4CC(img) #extraccion del histograma, deberia ser una funcion de img
            hist = cv2.calcHist([img_feature], [0], None,[16],[0,16])
            imgs_test[FEATURE].append( hist )#histograma
            imgs_test[CLASS].append( i )#clase


knn = cv2.KNearest()
# Training
knn.train( imgs_train[FEATURE], imgs_train[CLASS] )

'''
#Prediction
k = 5 #Debemos calcular k
retval, results, neigh_resp, dists = knn.find_nearest( imgs_test, k )
#results.ravel() ?

'''
