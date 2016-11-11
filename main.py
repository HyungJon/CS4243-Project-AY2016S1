import numpy as np
import cv2
import cv2.cv as cv
import math
import numpy.linalg as la

videos = ["beachVolleyball1.mov"]
target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 120.0], [390.0, 420.0]])
player_dist = np.zeros([4])

def getHomogBuiltIn(pts):
    pts_src = np.zeros([5, 2])
    pts_dst = np.zeros([5, 2])

    index = 0
    for i in range(len(pts)):
        if i != 2:
            pts_src[index] = pts[index]
            pts_dst[index] = target_coords[index]
            index = index + 1

    homography, status = cv2.findHomography(pts_src, pts_dst)
    return homography

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

    U, S, V = la.svd(M, full_matrices=False)

    min_index = -1
    min_eigen = 99999

    for i in range(len(S)):
        # print S[i]
        if abs(S[i]) < min_eigen:
            min_eigen = abs(S[i])
            min_index = i
    L = np.transpose(V[-1,:]) * 1.0

    h = L.reshape(3,3) / L[8]
    
    return h

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

def createOutputFrame(orig):
    for y in range(149, 390):
        orig[y][149] = [0, 0, 255]
        orig[y][150] = [0, 0, 255]
        orig[y][151] = [0, 0, 255]
        orig[y][629] = [0, 0, 255]
        orig[y][630] = [0, 0, 255]
        orig[y][631] = [0, 0, 255]
    for x in range(149, 630):
        orig[149][x] = [0, 0, 255]
        orig[150][x] = [0, 0, 255]
        orig[151][x] = [0, 0, 255]
        orig[389][x] = [0, 0, 255]
        orig[390][x] = [0, 0, 255]
        orig[391][x] = [0, 0, 255]
    for y in range(119,419):
        orig[y][389] = [0, 0, 255]
        orig[y][390] = [0, 0, 255]
        orig[y][391] = [0, 0, 255]
    return orig

def trackPlayerStats(fr, coords_new, coords_last, player_dist):
    if fr == 0:
        coords_new = coords_last

    for i in range(len(coords_new)):
        dist = abs(math.hypot((coords_new[i][0] - coords_last[i][0]), (coords_new[i][1] - coords_last[i][1])))
        player_dist[i] = player_dist[i] + dist * 0.0333
        # 1m = 30px => each pixel represents 3.33cm
    coords_new = coords_last
    return player_dist

# def trackBallPos(player1, player2, player4, len):
#     ballPos = np.zeros([len, 2])
#     for i in range

lines0 = [line.rstrip('\n') for line in open('video1_point0.txt', 'r')]
lines1 = [line.rstrip('\n') for line in open('video1_point1.txt', 'r')]
lines3 = [line.rstrip('\n') for line in open('video1_point3.txt', 'r')]
lines4 = [line.rstrip('\n') for line in open('video1_point4.txt', 'r')]
lines5 = [line.rstrip('\n') for line in open('video1_point5.txt', 'r')]

player1 = [line.rstrip('\n') for line in open('video1_player1.txt', 'r')]
player2 = [line.rstrip('\n') for line in open('video1_player2.txt', 'r')]
player3 = [line.rstrip('\n') for line in open('video1_player3.txt', 'r')]
player4 = [line.rstrip('\n') for line in open('video1_player4.txt', 'r')]

coords = np.zeros([len(lines1)*2 - 1, 6, 2])
player_coords = np.zeros([len(player1)*2 - 1, 4, 2])

for i in range(len(lines1)):
    coords[2*i] = [lines0[i].split(), lines1[i].split(), [-1,-1], lines3[i].split(), lines4[i].split(), lines5[i].split()]
    if i < len(lines1) - 1:
        coords[2*i+1] = coords[2*i]

for i in range(len(player1)):
    player_coords[2*i] = [player1[i].split(), player2[i].split(), player3[i].split(), player4[i].split()]
    if i < len(player1) - 1:
        player_coords[2*i+1] = player_coords[2*i]

cap = cv2.VideoCapture("beachVolleyball1.mov")
fcount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

for fr in range(len(coords)):
    _, frame = cap.read()
    h = getHomography(coords[fr])
    topdown = cv2.warpPerspective(frame, h, (780, 540))
    # topdown = np.zeros([540,780,3])
    topdown = createOutputFrame(topdown)

    for i in range(len(player_dist)):
        orig = [player_coords[fr][i][0], player_coords[fr][i][1], 1]
        res = np.dot(h, orig)
        player_coords[fr][i] = [res[0]/res[2], res[1]/res[2]]

    if fr == 0:
        player_dist = [0,0,0,0]
    else:
        player_dist = trackPlayerStats(fr, player_coords[fr], player_coords[fr-1], player_dist)

    for i in range(len(player_dist)):
        msg = "Distance moved by player " + str(i+1) + ": " + str(player_dist[i]) + "m"
        print msg
        cv2.circle(topdown, (int(player_coords[fr][i][0]), int(player_coords[fr][i][1])), int(15), (255,0,0), int(-1))
        filename = "td_fr"+str(fr+1)+".jpg"
        cv2.imwrite(filename, topdown)

    cv2.imshow('Top-down view frame', cv2.convertScaleAbs(topdown))
    cv2.waitKey(10)
cv2.waitKey(1000)
cv2.destroyAllWindows()
cap.release()
