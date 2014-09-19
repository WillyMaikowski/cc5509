#inicio de la tarea
import numpy as np
import cv2
import os
import utils

path = [ "dataSet/train/", "dataSet/test/" ]

j=0
totalPixels = 0
totalSkinPixels = 0
totalNonSkinPixels = 0
train = []
train_responses = []
test = []
test_responses = []

#train = np.asarray( train )
#train_responses = np.asarray( train_responses )
#test = np.asarray( test )
#test_responses = np.asarray( test_responses )
for j in range(2):
    count = 0
    for r, d, f in os.walk(path[j]+"original/"):
        for imgName in f:
            count = count + 1
            img = cv2.imread(path[j]+"original/" + imgName, cv2.IMREAD_COLOR)
            #img2 = cv2.imread(path[j]+"original/img5.jpg", cv2.IMREAD_COLOR)
            imgGroundThruth = cv2.imread(path[j]+"groundThruth/" + imgName, 0)
            #ret, imgGroundThruth = cv2.threshold(imgGroundThruth, 127, 255, cv2.THRESH_BINARY)#todos los grises los tira a blanco
            ret, imgGroundThruth = cv2.threshold(imgGroundThruth, 127, 1, cv2.THRESH_BINARY_INV)#uno indica piel, cero indica no piel
            #print(imgGroundThruth[len(imgGroundThruth[0])/2,])
            
            #hist = cv2.calcHist( [imgGroundThruth], [0], None, [2], [0, 256], False )
            
            #hist = cv2.calcHist( [imgGroundThruth], [0], None, [2], [0, 2], False )
            #print(hist)
            
            #totalSkinPixels += hist[0][0]
            #totalNonSkinPixels += hist[1][0]
            #totalPixels += img[:,:,0].size
            
            #print(hist) 
            #print( totalPixels)
            #print( totalNonSkinPixels)
            #print( totalSkinPixels) 
            #print( totalNonSkinPixels + totalSkinPixels )
            
            #cv2.imshow('',img[:,:,1]); cv2.waitKey(0); cv2.destroyAllWindows(); quit()        
            
            #curve = utils.hist_curve(img)
            #cv2.imshow('histogram',curve)
            #print(img[0,0])
            
            
            #img = np.reshape(img, (img[:,:,0].size,3))
            #img2 = np.reshape(img2, (img2[:,:,0].size,3))
            #print(img)
            #print(img2)
            
            #print(np.vstack((img, img2)))
            
            #print(np.reshape(img, (img[:,:,0].size,3)))
            #print(np.vstack(np.reshape(img, (img[:,:,0].size,3))))
            #print(np.reshape(imgGroundThruth, imgGroundThruth.size))
            #cv2.imshow('image',img); cv2.waitKey(0); cv2.destroyAllWindows(); quit()
            
            #ver http://stackoverflow.com/questions/13254234/train-skin-pixels-using-opencv-cvnormalbayesclassifier
            img = np.reshape(img, (img[:,:,0].size,3))
            imgGroundThruth = np.reshape(imgGroundThruth, imgGroundThruth.size)

            if( j == 0 ):
                if count == 1:
                    train = img
                    train_responses = imgGroundThruth
                else:
                    train = np.vstack((train, img))
                    train_responses = np.hstack((train_responses, imgGroundThruth))
                    #train.append(img)
                    #train_responses.append(imgGroundThruth)
            else:
                if count == 1:
                    test = img
                    test_responses = imgGroundThruth
                else:
                    test = np.vstack((test, img))
                    test_responses = np.hstack((test_responses, imgGroundThruth))
                    #test.append(img)
                    #test_responses.append(imgGroundThruth)

print("Entrenando")
bayes = cv2.NormalBayesClassifier(np.float32(train), np.float32(train_responses))
#np.save('backup_train',bayes)

#cv2.NormalBayesClassifier.predict(samples)

# Prediction
retval, results = bayes.predict(np.float32(test))

results = np.hstack(results)


matches = results == test_responses
correct = np.count_nonzero( matches )
accuracy = correct * 100.0 / results.size
print accuracy



#print( "total pixels = " + str(totalPixels))
#print( "total non skin pixels = " + str(totalNonSkinPixels))
#print( "total skin pixels = " + str(totalSkinPixels) )
#print( "total pixels check = " + str(totalNonSkinPixels + totalSkinPixels ))

