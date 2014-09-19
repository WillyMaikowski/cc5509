# inicio de la tarea
import numpy as np
import cv2
import os
import utils

cluster_n = 5
img_size = 512

print 'sampling distributions...'
points, ref_distrs = utils.make_gaussians(cluster_n, img_size)

print 'EM (opencv) ...'
em = cv2.EM(cluster_n, cv2.EM_COV_MAT_GENERIC)
em.train(points)
means = em.getMat('means')
covs = em.getMatVector('covs')
found_distrs = zip(means, covs)
print 'ready!\n'

img = np.zeros((img_size, img_size, 3), np.uint8)
for x, y in np.int32(points):
    cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
for m, cov in ref_distrs:
    utils.draw_gaussain(img, m, cov, (0, 255, 0))
for m, cov in found_distrs:
    utils.draw_gaussain(img, m, cov, (0, 0, 255))

cv2.imshow('gaussian mixture', img)
ch = 0xFF & cv2.waitKey(0)
if ch == 27:
    break
cv2.destroyAllWindows()
