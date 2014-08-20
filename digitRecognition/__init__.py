import numpy as np
import cv2
import utils
import os


imgs_train = [ [], [] ]
for i in range( 10 ):
    for r, d, f in os.walk( "digitos/train/digit_" + str( i ) ):
        for img in f:
            img = cv2.imread( img, 0 )  
            ret, img = cv2.threshold( img, 127, 255, cv2.THRESH_BINARY )
            img_feature = img #extraccion del histograma, deberia ser una funcion de img
            imgs_train[0].append( img_feature )
            imgs_train[1].append( i )
        

knn = cv2.KNearest()
# Training
knn.train( imgs_train[0], imgs_train[1] )

'''
#Prediction
k = 5 #Debemos calcular k
retval, results, neigh_resp, dists = knn.find_nearest( imgs_test, k )
#results.ravel() ?

'''
