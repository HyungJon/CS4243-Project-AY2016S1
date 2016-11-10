import numpy as np
import cv2
import cv2.cv as cv
import math
import numpy.linalg as la

target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 120.0], [390.0, 420.0]])

def computeHomography(pts):
    M = np.zeros([8,8])
    b = np.zeros([8])
    
    idx = 0
    for i in range(len(pts)):
        if pts[i][0] != -1:
            x = pts[i][0] # original x-coord: up
            y = pts[i][1] # original y-coord: vp

            u = target_coords[i][0] # transformed x-coord: uc
            v = target_coords[i][1] # transformed y-coord: vc

            M[2*idx] = [x, y, 1, 0, 0, 0, -1*u*x, -1*u*y]
            M[2*idx+1] = [0, 0, 0, x, y, 1, -1*v*x, -1*v*y]

            b[2*idx] = u
            b[2*idx+1] = v
            
            idx = idx + 1

    a, e, r, s = la.lstsq(M, b)

    h = np.zeros(9)
    for i in range(len(a)):
        h[i] = a[i]
    h[8] = 1

    homography = h.reshape(3,3)
    # print homography
    
    return homography

lines1 = [line.rstrip('\n') for line in open('video1_rightCourtCorner.txt', 'r')]
lines0 = [line.rstrip('\n') for line in open('video1_leftCourtCorner.txt', 'r')]
lines4 = [line.rstrip('\n') for line in open('video1_rightNetpole.txt', 'r')]
lines5 = [line.rstrip('\n') for line in open('video1_leftNetpole.txt', 'r')]

coords = np.zeros([len(lines1)*2 - 1, 6, 2])

for i in range(len(lines1)):
    coords[2*i] = [lines0[i].split(), lines1[i].split(), [-1,-1], [-1,-1], lines4[i].split(), lines5[i].split()]
    if i < len(lines1) - 1:
        coords[2*i+1] = coords[2*i]

cap = cv2.VideoCapture("beachVolleyball1.mov")
fcount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

for fr in range(len(coords)):
    _, frame = cap.read()
    h = computeHomography(coords[fr])
    topdown = cv2.warpPerspective(frame, h, (780, 540))

    cv2.imshow('Top-down view frame', cv2.convertScaleAbs(topdown))
    cv2.waitKey(10)
cv2.waitKey(1000)
cv2.destroyAllWindows()
cap.release()
