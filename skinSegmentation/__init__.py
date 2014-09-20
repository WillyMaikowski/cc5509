#inicio de la tarea
import numpy as np
import cv2
import os
import utils

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


path = [ "dataSet/train/", "dataSet/test/" ]

j=0
totalPixels = 0
totalSkinPixels = 0
totalNonSkinPixels = 0
train = []
train_responses = []
test = []
test_responses = []

skin_freq = {}
non_skin_freq = {}

skin_points = []
non_skin_points = []

for j in range(2):
    count = 0
    for r, d, f in os.walk(path[j]+"original/"):
        for imgName in f:
            #imgName = "img22.jpg"
            count = count + 1
            img = cv2.imread(path[j]+"original/" + imgName, cv2.IMREAD_COLOR)
            img_skin = img.copy()
            #img2 = cv2.imread(path[j]+"original/img5.jpg", cv2.IMREAD_COLOR)
            imgGroundThruth = cv2.imread(path[j]+"groundThruth/" + imgName, 0)
            #ret, imgGroundThruth = cv2.threshold(imgGroundThruth, 127, 255, cv2.THRESH_BINARY)#todos los grises los tira a blanco
            ret, imgGroundThruth_inv = cv2.threshold(imgGroundThruth, 127, 1, cv2.THRESH_BINARY)#uno indica no piel, cero indica piel
            ret, imgGroundThruth = cv2.threshold(imgGroundThruth, 127, 1, cv2.THRESH_BINARY_INV)#uno indica piel, cero indica no piel

            skin_img = cv2.bitwise_and(img,img,mask = imgGroundThruth)
            non_skin_img = cv2.bitwise_and(img,img,mask = imgGroundThruth_inv)
            #print(imgGroundThruth[len(imgGroundThruth[0])/2,])

            #hist = cv2.calcHist( [imgGroundThruth], [0], None, [2], [0, 256], False )

            hist = cv2.calcHist( [imgGroundThruth], [0], None, [2], [0, 2], False )
            #print(hist)

            totalSkinPixels += hist[0][0]
            totalNonSkinPixels += hist[1][0]
            totalPixels += img[:,:,0].size

            #print(hist)
            #print( totalPixels)
            #print( totalNonSkinPixels)
            #print( totalSkinPixels)
            #print( totalNonSkinPixels + totalSkinPixels )

            #cv2.imshow('',img[:,:,1]); cv2.waitKey(0); cv2.destroyAllWindows(); quit()
            #cv2.imshow('',skin_img); cv2.waitKey(0); cv2.destroyAllWindows(); quit()

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


            # if( j == 0 ):
            #     #generar un histograma de los distintos colores
            #     for k in range(len(imgGroundThruth)):
            #         for l in range(len(imgGroundThruth[0])):
            #             key = tuple(img[k][l])
            #             if imgGroundThruth[k][l] == 1:#piel
            #                 skin_freq[key] = skin_freq.get(key, 0) + 1
            #             else:
            #                 non_skin_freq[key] = non_skin_freq.get(key, 0) + 1
            #
            #     np.save('count', count)
            #     np.save('piel_dict',skin_freq)
            #     np.save('no_piel_dict',non_skin_freq)





            if( j == 0 ):
                #separar entre piel y no piel
                skin_img = np.reshape(skin_img, (skin_img[:,:,0].size,3))
                #skin_img = skin_img[ ~np.all(skin_img == (0,0,0)) ]

                non_skin_img = np.reshape(non_skin_img, (non_skin_img[:,:,0].size,3))
                #non_skin_img = non_skin_img[ ~(non_skin_img == (0,0,0))  ]
                if len(skin_points) <= 0:
                    skin_points = skin_img
                else:
                    skin_points = np.vstack((skin_points, skin_img))

                if len( non_skin_points ) <= 0:
                    non_skin_points = non_skin_img
                else:
                    non_skin_points = np.vstack((non_skin_points, non_skin_img))


            img = np.reshape(img, (img[:,:,0].size,3))
            imgGroundThruth = np.reshape(imgGroundThruth, imgGroundThruth.size)

            print( "count",count )


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

            # index = 0
            # for k in range(len(img_skin)):
            #     for l in range(len(img_skin[0])):
            #         if imgGroundThruth[index] == 1:
            #             img_skin[k,l] = (0,0,0)
            #         index +=1

            #cv2.imshow('image',img_skin); cv2.waitKey(0); cv2.destroyAllWindows();

#np.save('piel_dict',skin_freq)
#np.save('no_piel_dict',non_skin_freq)


skin_freq = np.load('piel_dict.npy').item()
non_skin_freq = np.load('no_piel_dict.npy').item()


#print(skin_freq)
#print(non_skin_freq)
#print(non_skin_freq)

#print(skin_points)
print("imagenes leidas")

fig = plt.figure( figsize = (20,20) )
ax = fig.add_subplot(255, projection='3d')
max_it = 500
it = 0
for i in skin_points:
    if ~i.all():
        continue
    it = it + 1
    xs = i[0]
    ys = i[1]
    zs = i[2]
    ax.scatter(xs, ys, zs, c='r', marker='o')
    if it >= max_it:
        break


it = 0
for i in non_skin_points:
    if ~i.all():
        continue
    it = it + 1
    xs = i[0]
    ys = i[1]
    zs = i[2]
    ax.scatter(xs, ys, zs, c='b', marker='^')
    if it >= max_it:
        break



ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.tight_layout()
plt.show()

print("Entrenando")
bayes = cv2.NormalBayesClassifier(np.float32(train), np.float32(train_responses))


#cv2.NormalBayesClassifier.predict(samples)

 #Prediction
predictions = []
j=1
#usando clasificador de bayes
for r, d, f in os.walk(path[j]+"original/"):
    for imgName in f:
        img_original = cv2.imread(path[j]+"original/" + imgName, cv2.IMREAD_COLOR)
        img = img_original.copy()
        imgGroundThruth = cv2.imread(path[j]+"groundThruth/" + imgName, 0)
        ret, imgGroundThruth = cv2.threshold(imgGroundThruth, 128, 1, cv2.THRESH_BINARY_INV)#uno indica piel, cero indica no piel

        img = np.reshape(img, (img[:,:,0].size,3))
        imgGroundThruth = np.reshape(imgGroundThruth, imgGroundThruth.size)

        retval, results = bayes.predict(np.float32(img))
        results = np.hstack(results)
        matches = results == imgGroundThruth
        correct = np.count_nonzero( matches )
        accuracy = correct * 100.0 / results.size
        predictions.append( (imgName, accuracy) )
        print((imgName, accuracy))

        index = 0
        img_reconstruct = img_original.copy()
        cv2.imshow(imgName+"original",img_original);

        for k in range(len(img_original)):
            for l in range(len(img_original[0])):
                img_reconstruct[k,l] = img[index]
                if results[index] == 1.0:#predijo piel
                    img_original[k,l]=(0,255,0)
                else:
                    img_original[k,l]=(255,0,0)
                index+=1

        #cv2.imshow(imgName+"_recontru",img_reconstruct);
        cv2.imshow(imgName,img_original); cv2.waitKey(0); cv2.destroyAllWindows();
print(predictions)

retval, results = bayes.predict(np.float32(test))
results = np.hstack(results)
matches = results == test_responses
correct = np.count_nonzero( matches )
accuracy = correct * 100.0 / results.size
print "Total",accuracy


####################################################################################
predictions = []
j=1
phi = 1
PC1 = 1.0*totalSkinPixels/totalPixels#probabilidad ser piel
PC2 = 1.0*totalNonSkinPixels/totalPixels#probabilidad no ser piel
#usando clasificador casero
for r, d, f in os.walk(path[j]+"original/"):
    for imgName in f:
        #imgName = "img22.jpg"

        img_original = cv2.imread(path[j]+"original/" + imgName, cv2.IMREAD_COLOR)
        img = img_original.copy()
        imgGroundThruth = cv2.imread(path[j]+"groundThruth/" + imgName, 0)
        ret, imgGroundThruth = cv2.threshold(imgGroundThruth, 128, 1, cv2.THRESH_BINARY_INV)#uno indica piel, cero indica no piel
        #print(imgGroundThruth[len(imgGroundThruth[0])/2,])
        imgGroundThruth = np.reshape(imgGroundThruth, imgGroundThruth.size)

        results = []

        for k in range(len(img_original)):
            for l in range(len(img_original[0])):
                key = tuple(img_original[k][l])
                PXi_C1 = 1.0*skin_freq.get(key,0)/totalSkinPixels
                PXi_C2 = 1.0*non_skin_freq.get(key,0)/totalNonSkinPixels
                PXi = PXi_C1*PC1 + PXi_C2*PC2
                PC1_Xi = PXi_C1*PC1/PXi
                PC2_Xi = PXi_C2*PC2/PXi
                if PC1_Xi > PC2_Xi*phi:#es piel
                    results.append(1)
                else:
                    results.append(0)

        matches = results == imgGroundThruth
        correct = np.count_nonzero( matches )
        accuracy = correct * 100.0 / len(results)
        predictions.append( (imgName, accuracy) )
        print((imgName, accuracy))

        index = 0

        cv2.imshow(imgName+"original",img_original);

        for k in range(len(img_original)):
            for l in range(len(img_original[0])):
                if results[index] == 1:#predijo piel
                    img_original[k,l]=(0,255,0)
                #else:
                #    img_original[k,l]=(255,0,0)
                if imgGroundThruth[index] == 1:
                    img[k,l] = (255,255,255)

                index+=1

        cv2.imshow(imgName+"mascara",img);
        cv2.imshow(imgName,img_original); cv2.waitKey(0); cv2.destroyAllWindows();

# retval, results = bayes.predict(np.float32(test))
# results = np.hstack(results)
# matches = results == test_responses
# correct = np.count_nonzero( matches )
# accuracy = correct * 100.0 / results.size
# print "Total",accuracy



#print( "total pixels = " + str(totalPixels))
#print( "total non skin pixels = " + str(totalNonSkinPixels))
#print( "total skin pixels = " + str(totalSkinPixels) )
#print( "total pixels check = " + str(totalNonSkinPixels + totalSkinPixels ))

