import numpy as np
import cv2
import utils
import os

#constantes
TRAIN = 0
TEST = 1
FEATURE = 0
CLASS = 1

experiment = '4CC'

imgs = [ [ [], [] ], [ [], [] ] ]
imgs_2x2 = [ [ [], [] ], [ [], [] ] ]
imgs_4x4 = [ [ [], [] ], [ [], [] ] ]
for i in range( 0, 10 ):
    path = [ "digitos/train/digit_" + str( i ) + "/", "digitos/test/digit_" + str( i ) + "/" ]
    for j in range( 2 ):
        for r, d, f in os.walk( path[j] ):
            for img in f:
                img = cv2.imread( path[j] + img, 0 )
                ret, img = cv2.threshold( img, 127, utils.BACKGROUND, cv2.THRESH_BINARY )
                #img = utils.mBlackBottomRight(img)
                #cv2.imshow('',img); cv2.waitKey(0); cv2.destroyAllWindows(); quit()
                img_feature = utils.apply4CCv2( img )  # extraccion del histograma, deberia ser una funcion de img
                #histograma 1x1
                hist = cv2.calcHist( [img_feature], [0], None, [16], [0, 16] )
                imgs[j][FEATURE].append( hist )  # histograma
                imgs[j][CLASS].append( i )  # clase
                #histograma 2x2
                hist2x2 = []
                n = 2
                for k in range(n):
                    for l in range(n):
                        hist2x2 = np.append(
                            hist2x2 , cv2.calcHist( [img_feature[k*len(img_feature)/n : (k+1)*len(img_feature)/n, l*len(img_feature)/n : (l+1)*len(img_feature)/n]],
                                        [0], None, [16], [0, 16] ) )
                imgs_2x2[j][FEATURE].append( hist2x2 )  # histograma
                imgs_2x2[j][CLASS].append( i )  # clase
                #histograma 4x4
                hist4x4 = []
                n = 4
                for k in range(n):
                    for l in range(n):
                        hist4x4 = np.append(
                            hist4x4 , cv2.calcHist( [img_feature[k*len(img_feature)/n : (k+1)*len(img_feature)/n, l*len(img_feature)/n : (l+1)*len(img_feature)/n]],
                                        [0], None, [16], [0, 16] ) )
                imgs_4x4[j][FEATURE].append( hist4x4 )  # histograma
                imgs_4x4[j][CLASS].append( i )  # clase
    np.save('backup'+experiment+'_digit='+i,imgs)
    np.save('backup_2x2'+experiment+'_digit='+i,imgs_2x2)
    np.save('backup_4x4'+experiment+'_digit='+i,imgs_4x4)

np.save('process'+experiment,imgs)
np.save('process_2x2'+experiment,imgs_2x2)
np.save('process_4x4'+experiment,imgs_4x4)

#knn = cv2.KNearest()
# Training
#knn.train( np.asarray(imgs[TRAIN][FEATURE]), np.asarray(imgs[TRAIN][CLASS]) )

#guardar entrenamiento o calculos antes del entrenamiento

# Prediction
#k = 5  # Debemos calcular k
#retval, results, neigh_resp, dists = knn.find_nearest( imgs[TEST][FEATURE], k )

#matches = results == imgs[TEST][CLASS]
#correct = np.count_nonzero( matches )
#accuracy = correct * 100.0 / results.size
#print accuracy