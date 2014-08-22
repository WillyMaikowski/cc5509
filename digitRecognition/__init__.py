import numpy as np
import cv2
import utils
import os

#constantes
TRAIN = 0
TEST = 1
FEATURE = 0
CLASS = 1

imgs = [ [ [], [] ], [ [], [] ] ]
for i in range( 3, 10 ):
    path = [ "digitos/train/digit_" + str( i ) + "/", "digitos/test/digit_" + str( i ) + "/" ]
    for j in range( 2 ):
        for r, d, f in os.walk( path[j] ):
            for img in f:
                img = cv2.imread( path[j] + img, 0 )
                ret, img = cv2.threshold( img, 127, utils.BACKGROUND, cv2.THRESH_BINARY )
                #img = utils.mBlackBottomRight(img)
                #cv2.imshow('',img); cv2.waitKey(0); cv2.destroyAllWindows(); quit()
                img_feature = utils.apply13C( img )  # extraccion del histograma, deberia ser una funcion de img
                hist = cv2.calcHist( [img_feature], [0], None, [13], [0, 12] )
                print hist; quit()
                imgs[j][FEATURE].append( hist )  # histograma
                imgs[j][CLASS].append( i )  # clase

np.save('result.txt',imgs)

knn = cv2.KNearest()
# Training
knn.train( imgs[TRAIN][FEATURE], imgs[TRAIN][CLASS] )

#guardar entrenamiento o calculos antes del entrenamiento

# Prediction
k = 5  # Debemos calcular k
retval, results, neigh_resp, dists = knn.find_nearest( imgs[TEST][FEATURE], k )

matches = results == imgs[TEST][CLASS]
correct = np.count_nonzero( matches )
accuracy = correct * 100.0 / results.size
print accuracy