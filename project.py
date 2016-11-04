import cv2
import numpy as np
import math
import numpy.linalg as la

target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 150.0], [390.0, 390.0]])

out = np.zeros([250,400,3])

img = cv2.imread("ss1.jpg")

def getHomography(pts):
    M = np.zeros([10,9])
    
    idx = 0
    for i in range(len(pts)):
        if pts[i][0] != -1:
            x = pts[i][0] # original x-coord: up
            y = pts[i][1] # original y-coord: vp

            u = target_coords[i][0] # transformed x-coord: uc
            v = target_coords[i][1] # transformed y-coord: vc

            M[2*idx] = [x, y, 1, 0, 0, 0, -1*u*x, -1*u*y, -1*u]
            M[2*idx+1] = [0, 0, 0, x, y, 1, -1*v*x, -1*v*y, -1*v]
            idx = idx + 1

    # print "M"
    # print M
    U, S, V = la.svd(M, full_matrices=False)

    min_index = -1
    min_eigen = 99999

    # print "U . S . V"
    # print np.dot(U, np.dot(np.diag(S), V))
    # print "M"
    # print M

    for i in range(len(S)):
        # print S[i]
        if abs(S[i]) < min_eigen:
            min_eigen = abs(S[i])
            min_index = i

    # L = V[min_index]
    # print min_index
    # print L
    # print np.dot(M, L)
    # print ""
    # print np.dot(M, (L * 1.0 / L[8]))
    # h = L.reshape(3,3) * 1.0 / L[8]
    # print h
    L = np.transpose(V[-1,:]) * 1.0
    # print np.dot(M, L)

    h = L.reshape(3,3) / L[8]
    print "Computed homography:"
    print h

    pts_src = np.array([[196.0,293.0],[437.0,137.0],[191.0,62.0],[40.0,88.0],[35.0,148]])
    pts_dst = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 390.0]])
    homog, status = cv2.findHomography(pts_src, pts_dst)
    # for checking
    print "Correct homography:"
    print homog

    print "computed H * L"
    print np.dot(M, L)
    print "correct H * L"
    print np.dot(M, homog.reshape(9))
    
    return h

h = getHomography(np.array([[196.0,293.0],[437.0,137.0],[191.0,62.0],[40.0,88.0],[-1.0,-1.0],[35.0,148]]))
hi = la.inv(h)

coords = np.array([[196.0, 293.0, 1.0],[437.1, 137.0, 1.0],[191.0, 62.0, 1.0],[40.0, 88.0, 1.0], [35.0,148.0, 1.0]])
# coords = np.array([[196.0, 293.1, 0],[437.0, 137.1, 0],[191.0, 62.1, 0],[40.0, 88.1, 1.0], [35.0, 148.0, 1.0]])

res1 = np.zeros([5, 2])
res2 = np.zeros([5, 3])

# print "Original coordinates:"
# print coords

for i in range(len(coords)):
    r = np.dot(h, coords[i]) 
    res1[i] = [r[0]/r[2],r[1]/r[2]]
    res2[i] = [r[0]/r[2],r[1]/r[2],1]

print "Homography transformation result"
print res1

im_out = cv2.warpPerspective(img, h, (780, 540))
cv2.imwrite("ss1_homog_computed.jpg", im_out)

im_recon = cv2.warpPerspective(im_out, hi, (632, 300))
cv2.imwrite("ss1_homog_reconstructed.jpg", im_recon)
