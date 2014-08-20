import numpy as np
import cv2
import utils


src = 'digitos/train/digit_0/img001-00003.png'
img = cv2.imread( src, cv2.IMREAD_GRAYSCALE ) #Lo mismo que 0
#print img[0, 0] #pixel [x,y]

ret,img = cv2.threshold( img, 127, 255, cv2.THRESH_BINARY )#255 es blanco, 0 es negrito
img_vals = np.zeros( shape = ( len( img ), len( img ) ) ) - 1 #matriz con -1

#recorrer cada pixel
for i in range( 0, len( img ) ):
    for j in range( 0, len( img ) ): #imagen cuadrada?
        print img[i][j]

cv2.imshow( "imagename", img )

cv2.waitKey(0) #espera hasta que se aprete una tecla
cv2.destroyAllWindows()