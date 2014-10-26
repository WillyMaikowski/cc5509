import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc

descriptors = []
keypoints = []
path = "datos/corpus/"
num = 0
start = 0
print("start reading images")

for r, d, f in os.walk(path):
    for img_name in f:
        num += 1


        img = cv2.imread(path + img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #if( num == 2874 ):
        #    cv2.imshow(img_name, img);
        #    cv2.waitKey(0);
        #    cv2.destroyAllWindows();


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
        if num % 1500 == 0:
            print(num)

        #if( num % 500 == 0):
        #     break

        #print(len(descriptors))
        #print("#######################")

        #img = cv2.drawKeypoints(gray, kp)

        #cv2.imwrite('sift_keypoints.jpg', img)
        #cv2.imshow("", img);
        #cv2.waitKey(0);
        #cv2.destroyAllWindows();


descriptors = random.sample(descriptors, 50000)
#descriptors = random.sample(descriptors, 100000)
print("Desciptors generados")

K = 200
while K <= 2000:
    norm = cv2.NORM_L2
    ret, labels, centers = cv2.kmeans(np.asarray( descriptors ), K, (cv2.TERM_CRITERIA_EPS, 30, 0.1),10,cv2.KMEANS_RANDOM_CENTERS)
    print("kmeans entrenado")
    #np.save('backup_descriptors', descriptors)

    knn_clusters = cv2.KNearest()
    knn_clusters.train( centers, np.asarray( list(xrange(len(centers))) ) )
    print("knn entrenado")


    #print(centers)
    TRAIN = 0
    TEST = 1
    FEATURE = 0
    CLASS = 1
    path = ["datos/train/perro/", "datos/train/no_perro/", "datos/test/perro/", "datos/test/no_perro/"]
    TYPE = [TRAIN, TRAIN, TEST, TEST]

    if True:
        final_descriptor = [ [ [], [] ], [ [], [] ] ]
        n = 0
        for i in range(len(path)):
            for r, d, f in os.walk(path[i]):
                for img_name in f:

                    if n % 500 == 0:
                        print(n)
                    n = n + 1

                    img = cv2.imread(path[i] + img_name)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    sift = cv2.SIFT()
                    kp, des = sift.detectAndCompute(gray, None)

                    img_descp = []
                    if( des is None ):
                        continue

                    #print(des)
                    retval, results, neigh_resp, dists = knn_clusters.find_nearest( np.asarray( des ), 1 )

                    for k in range(len(results)):#un resultado para cada desciptor de la imagen
                        img_descp.append( results[k] )#encuentro cluster mas cercano al descriptor

                    img_descp = np.asarray( img_descp )

                    hist = cv2.calcHist( [img_descp], [0], None, [K], [0, K] )

                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX )
                    #print(hist)
                    final_descriptor[TYPE[i]][FEATURE].append(  hist.copy() )
                    if( i == 1 or i == 3 ):#no perro
                        final_descriptor[TYPE[i]][CLASS].append( -1 )
                    else:#perro
                        final_descriptor[TYPE[i]][CLASS].append( 1 )

        print("Datos entrenamiento y testeo leidos")


    C = 1.0
    classifier = svm.SVC(kernel='rbf', probability=True, gamma=0.7, C=C)#.fit(train_descriptor, train_class)

    # print(len(final_descriptor[TRAIN][FEATURE][0]))
    # print(len(final_descriptor[TRAIN][CLASS]))
    # print(final_descriptor[TRAIN][FEATURE][0])
    # print(final_descriptor[TRAIN][CLASS][0])
    # print(final_descriptor[TEST][CLASS][0])
    # print(final_descriptor[TEST][CLASS][0])
    #
    # print(np.asarray( final_descriptor[TRAIN][CLASS] ).shape)
    # print(np.asarray( final_descriptor[TRAIN][FEATURE] ).shape[:2])
    # print(np.squeeze( np.asarray( final_descriptor[TRAIN][FEATURE] )).shape)
    # print(np.asarray( final_descriptor[TRAIN][FEATURE] ).reshape((999,200)).shape)
    # print(np.asarray( final_descriptor[TEST][FEATURE] ).reshape((400,200)).shape)
    # print(np.asarray( final_descriptor[TEST][CLASS] ).shape)

    final_descriptor[TRAIN][FEATURE] = np.asarray( np.asarray( final_descriptor[TRAIN][FEATURE] ).reshape(  np.asarray( final_descriptor[TRAIN][FEATURE] ).shape[:2] ) )
    final_descriptor[TEST][FEATURE] = np.asarray( np.asarray( final_descriptor[TEST][FEATURE] ).reshape(  np.asarray( final_descriptor[TEST][FEATURE] ).shape[:2] ) )
    # print( final_descriptor[TRAIN][FEATURE].shape )
    # print( final_descriptor[TEST][FEATURE].shape )


    y_score = classifier.fit( final_descriptor[TRAIN][FEATURE], final_descriptor[TRAIN][CLASS] ).decision_function( final_descriptor[TEST][FEATURE])


    fpr, tpr, _ = roc_curve(final_descriptor[TEST][CLASS], y_score)
    roc_auc = auc(fpr, tpr)



    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve K='+str(K))
    plt.legend(loc="lower right")
    plt.savefig("K="+str(K)+".png")
    #plt.show()

    print(fpr)
    print("---------------------------------------------------------------------")
    print(tpr)

    np.savetxt('fpr_K='+str(K)+".txt", fpr)
    np.savetxt('tpr_K='+str(K)+".txt", tpr)

    K = K + 300

#mean_tpr = 0.0
#mean_fpr = np.linspace(0, 1, 100)


# probas_ = classifier.fit(train_descriptor, train_class).predict_proba(test_descriptor)
# # Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(test_class, probas_[:, 1])
# mean_tpr += np.interp(mean_fpr, fpr, tpr)
# mean_tpr[0] = 0.0
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
# mean_tpr /= len(centers)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()



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