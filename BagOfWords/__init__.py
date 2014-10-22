import numpy as np
import cv2
import os

descriptors = []
keypoints = []
path = "datos/corpus/"
num = 0
start = 0

for r, d, f in os.walk(path):
    for img_name in f:
        num += 1


        img = cv2.imread(path + img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if( num == 2874 ):
            cv2.imshow(img_name, img);
            cv2.waitKey(0);
            cv2.destroyAllWindows();


        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(gray, None)

        # print(len(kp))
        #print(des)
        #print(len(des))
        #print(gray.size)

        #cv2.imshow("", img);
        #cv2.waitKey(0);
        #cv2.destroyAllWindows();


        if( des is None ):
            continue
        for i in range(len(des)):
            descriptors.append(des[i])

        #print(len(descriptors))
        print(num)
        #print(len(descriptors))
        #print("#######################")

        #img = cv2.drawKeypoints(gray, kp)

        #cv2.imwrite('sift_keypoints.jpg', img)
        #cv2.imshow("", img);
        #cv2.waitKey(0);
        #cv2.destroyAllWindows();


np.save('backup_descriptors', descriptors)



# knn = cv2.KNearest()
# Training
# knn.train( np.asarray(imgs[TRAIN][FEATURE]), np.asarray(imgs[TRAIN][CLASS]) )

#guardar entrenamiento o calculos antes del entrenamiento

# Prediction
#k = 5  # Debemos calcular k
#retval, results, neigh_resp, dists = knn.find_nearest( imgs[TEST][FEATURE], k )

#matches = results == imgs[TEST][CLASS]
#correct = np.count_nonzero( matches )
#accuracy = correct * 100.0 / results.size
#print accuracy