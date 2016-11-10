import cv2
import numpy as np
import math
import numpy.linalg as la

target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 120.0], [390.0, 420.0]])

# pass in an array of size 6, each element containing 2D coordinates
# 4 of them should contain points on image to be mapped to top-down view
# the other 2 should be [-1, -1] to indicate points not identified
# example param: [[196.0,293.0], [437.0,137.0], [-1.0, -1.0], [40.0,88.0], [-1.0,-1.0], [35.0,148]]
# refer to top-down img dimensions.jpg for how to populate the input array
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

# im_out = cv2.warpPerspective(img, h, (780, 540))
# cv2.imwrite("ss1_homog_computed.jpg", im_out)
