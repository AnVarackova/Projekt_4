# -*- coding: utf-8 -*-

import cv2
import os.path
import glob
import numpy as np
import time

def featureTracking(img1, img2, points1):
    winSize = (21, 21)
    err = []
    termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    features, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, winSize, maxLevel=2, criteria=termCrit, flags=0, minEigThreshold=0.001)

    f = np.zeros([len(status[status == 1]), 1, 2], np.float32)  #prazdna pole, naplnovat, kdyz bude 1
    pf = np.zeros([len(status[status == 1]), 1, 2], np.float32)
    k = 0

    for i in range(len(status)):
        if status[i] == 1:
            f[k, :] = features[i, :]
            pf[k, :] = points1[i, :]
            k += 1
    features = f
    points1 = pf

    return features, points1, err

def featureDetection(img):
    fast_treshold = 20
    nonmaxSuppression = True
    fast = cv2.FastFeatureDetector_create(fast_treshold, nonmaxSuppression)
    kp = fast.detect(img, None)
    kp2 = np.reshape(np.array([k.pt for k in kp]), (len(kp), 1, 2))
    kp2 = kp2.astype(np.float32)
    return kp2

def main():

    min_num_feat = 2000
    scale = 1

    ts = []                             # translace
    Rs = []                             # rotace
    Rf = np.identity(3, np.float)
    tf = np.zeros([3, 1], np.float)
    ts.append(tf)
    Rs.append(Rf)

    fns = sorted(glob.glob("./image_0/*.png"))
    focal = 718.8560                        # ohnisko
    pp = (607.1928, 185.2157)               # stred roviny promitani, stred obrazku
    cam_matrix = np.zeros([3, 3], dtype = np.float)     # matice kamery
    cam_matrix[0, 0] = focal
    cam_matrix[1, 1] = focal
    cam_matrix[0, 2] = pp[0]
    cam_matrix[1, 2] = pp[1]
    cam_matrix[2, 2] = 1

    points1 = []
    points2 = []

    img_1 = cv2.imread(fns[0], 0)
    points1 = featureDetection(img_1)

    for fn in fns[1:]:
        img_2 = cv2.imread(fn, 0)
        points2, points1, err =  featureTracking(img_2, img_1, points1)
        essMat, mask = cv2.findEssentialMat(points2, points1, focal, pp, cv2.RANSAC, 0.999, 1.0)
        retval, R, t, mask = cv2.recoverPose(essMat, points2, points1, cam_matrix)
        
        ts.append(t)
        Rs.append(R)
        tf = tf + (np.dot(Rf, t))
        Rf = np.dot(R, Rf)

        print(tf)

        if len(points1) < min_num_feat:
            points1 = featureDetection(img_1)
            points2, points1, err = featureTracking(img_2, img_1, points1)

        img_1 = img_2.copy()
        points1 = points2

        cv2.waitKey(1)

if __name__ == "__main__":
    main()

# tracking, essential matrix, recover pose

# scale neím, místo něj tam dám 1
# zatím neřeším ani vykreslování
# t je změna pozice
# na konci je důležité cv2.waitKey(1)
