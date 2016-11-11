import numpy as np
import cv2
import cv2.cv as cv
import math
import random # just for simulating player movements: later remove
import numpy.linalg as la

target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 120.0], [390.0, 420.0]])
player_coords = np.zeros([4, 2])
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
    return orig

def trackPlayerStats(fr, coords, player_coords, player_dist):
    if fr == 0:
        player_coords = coords

    for i in range(len(coords)):
        dist = abs(math.hypot((coords[i][0] - player_coords[i][0]), (coords[i][1] - player_coords[i][1])))
        player_dist[i] = player_dist[i] + dist * 0.0333
        # 1m = 30px => each pixel represents 3.33cm
    player_coords = coords
    return (player_coords, player_dist)

def createRandomMovement(coords):
    ydist = random.randint(0, 10) - 5
    xdist = random.randint(0, 10) - 5
    x = coords[0]
    y = coords[1]
    if x < 15:
        xdist = xdist + 5
    if x > 525:
        xdist = xdist - 5
    if y < 15:
        ydist = ydist + 5
    if y > 765:
        ydist = ydist - 5
    print xdist
    print ydist
    return [x + xdist, y + ydist]

lines0 = [line.rstrip('\n') for line in open('video1_point0.txt', 'r')]
lines1 = [line.rstrip('\n') for line in open('video1_point1.txt', 'r')]
lines3 = [line.rstrip('\n') for line in open('video1_point3.txt', 'r')]
lines4 = [line.rstrip('\n') for line in open('video1_point4.txt', 'r')]
lines5 = [line.rstrip('\n') for line in open('video1_point5.txt', 'r')]

coords = np.zeros([len(lines1)*2 - 1, 6, 2])

for i in range(len(lines1)):
    coords[2*i] = [lines0[i].split(), lines1[i].split(), [-1,-1], lines3[i].split(), lines4[i].split(), lines5[i].split()]
    if i < len(lines1) - 1:
        coords[2*i+1] = coords[2*i]

cap = cv2.VideoCapture("beachVolleyball1.mov")
fcount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

for i in range(len(player_coords)):
    player_coords[i][0] = random.randint(0, 780)
    player_coords[i][1] = random.randint(0, 540)
print player_coords

for fr in range(len(coords)):
    _, frame = cap.read()
    h = getHomography(coords[fr])
    topdown = cv2.warpPerspective(frame, h, (780, 540))
    topdown = createOutputFrame(topdown)

    # 
    # add the fn to calculate player stats here

    newCoords = np.zeros([4, 2])
    for i in range(len(player_coords)):
        print i
        newCoords[i] = createRandomMovement(player_coords[i])
    player_coords, player_dist = trackPlayerStats(fr, newCoords, player_coords, player_dist)
    print player_dist

    for i in range(len(player_coords)):
        cv2.circle(topdown, (int(player_coords[i][0]), int(player_coords[i][1])), int(15), (255,0,0), int(-1))

    cv2.imshow('Top-down view frame', cv2.convertScaleAbs(topdown))
    cv2.waitKey(10)
cv2.waitKey(1000)
cv2.destroyAllWindows()
cap.release()
