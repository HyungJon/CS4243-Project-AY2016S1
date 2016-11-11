import numpy as np
import cv2
import cv2.cv as cv
import math
import numpy.linalg as la

# videos = ["beachVolleyball1.mov"]
target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 120.0], [390.0, 420.0]])

def computeHomography5pts(pts):
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

def computeHomography4pts(pts):
    M = np.zeros([8,8])
    b = np.zeros([8])
    
    idx = 0
    print pts
    for i in range(len(pts)):
        if pts[i][0] != -1 and pts[i][0] != 0:
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

def drawCourt(orig):
    for y in range(149, 390):
        orig[y][149] = [0, 0, 255]
        orig[y][150] = [0, 0, 255]
        orig[y][151] = [0, 0, 255]
        orig[y][627] = [0, 0, 255]
        orig[y][628] = [0, 0, 255]
        orig[y][629] = [0, 0, 255]
    for x in range(149, 630):
        orig[149][x] = [0, 0, 255]
        orig[150][x] = [0, 0, 255]
        orig[151][x] = [0, 0, 255]
        orig[387][x] = [0, 0, 255]
        orig[388][x] = [0, 0, 255]
        orig[389][x] = [0, 0, 255]
    for y in range(119,419):
        orig[y][388] = [0, 0, 255]
        orig[y][389] = [0, 0, 255]
        orig[y][390] = [0, 0, 255]
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

def convert2TopDown(oddFrames, fcount, points, players, coords, player_coords):
    cap = cv2.VideoCapture("beachVolleyball1.mov")

    homogArray = np.zeros([fcount, 3, 3])
    for fr in range(fcount):
        # print fr
        _, frame = cap.read()
        h = np.zeros([3,3])
        if points == 5:
            h = computeHomography5pts(coords[fr])
        elif points == 4:
            h = computeHomography4pts(coords[fr])
        homogArray[fr] = h
        topdown = cv2.warpPerspective(frame, h, (780, 540))
        # topdown = np.zeros([540,780,3])
        topdown = drawCourt(topdown)

        for i in range(players):
            orig = [player_coords[i][fr][0], player_coords[i][fr][1], 1]
            res = np.dot(h, orig)
            player_coords[i][fr] = [int(res[0]/res[2]), int(res[1]/res[2])]
            cv2.circle(topdown, (int(player_coords[i][fr][0]), int(player_coords[i][fr][1])), int(15), (255,0,0), int(-1))
            # filename = "td_fr"+str(fr+1)+".jpg"
            # cv2.imwrite(filename, topdown)
        cv2.imshow('Top-down view frame', cv2.convertScaleAbs(topdown))
        cv2.waitKey(10)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    # cap.release()

    return player_coords

# Video 1 start
lines1_0 = [line.rstrip('\n').split() for line in open('video1_point0.txt', 'r')]
lines1_1 = [line.rstrip('\n').split() for line in open('video1_point1.txt', 'r')]
lines1_3 = [line.rstrip('\n').split() for line in open('video1_point3.txt', 'r')]
lines1_4 = [line.rstrip('\n').split() for line in open('video1_point4.txt', 'r')]
lines1_5 = [line.rstrip('\n').split() for line in open('video1_point5.txt', 'r')]

player1_1 = [line.rstrip('\n').split() for line in open('video1_player1.txt', 'r')]
player1_2 = [line.rstrip('\n').split() for line in open('video1_player2.txt', 'r')]
player1_3 = [line.rstrip('\n').split() for line in open('video1_player3.txt', 'r')]
player1_4 = [line.rstrip('\n').split() for line in open('video1_player4.txt', 'r')]

oddFrames1 = len(lines1_0)
fcount1 = oddFrames1*2-1
players1 = 4

coords1 = np.zeros([fcount1, 6, 2])
player_coords1 = np.zeros([players1, fcount1, 2])

for i in range(len(player1_1)):
    player_coords1[0][2*i] = player1_1[i]
    player_coords1[1][2*i] = player1_2[i]
    player_coords1[2][2*i] = player1_3[i]
    player_coords1[3][2*i] = player1_4[i]
    if i < len(player1_1) - 1:
        player_coords1[0][2*i+1] = player1_1[i]
        player_coords1[1][2*i+1] = player1_2[i]
        player_coords1[2][2*i+1] = player1_3[i]
        player_coords1[3][2*i+1] = player1_4[i]

for i in range(len(lines1_1)):
    coords1[2*i] = [lines1_0[i], lines1_1[i], [-1,-1], lines1_3[i], lines1_4[i], lines1_5[i]]
    if i < (oddFrames1-1):
        coords1[2*i+1] = coords1[2*i]

outputFiles1 = convert2TopDown(oddFrames1, fcount1, 5, players1, coords1, player_coords1)

for i in range(players1):
    filename = "video1_player" + str(i+1) + "_topdown.txt"
    np.savetxt(filename, outputFiles1[i], delimiter=' ', fmt='%i %i')
# Video 1 end

# Video 5 start
cap = cv2.VideoCapture("beachVolleyball5.mov")

lines5_2 = [line.rstrip('\n').split() for line in open('video5_point2.txt', 'r')]
lines5_3 = [line.rstrip('\n').split() for line in open('video5_point3.txt', 'r')]
lines5_4 = [line.rstrip('\n').split() for line in open('video5_point4.txt', 'r')]
lines5_5 = [line.rstrip('\n').split() for line in open('video5_point5.txt', 'r')]

player5_1 = [line.rstrip('\n').split() for line in open('video5_playerclosesttocamera.txt', 'r')]
player5_2 = [line.rstrip('\n').split() for line in open('video5_player2.txt', 'r')]
player5_3 = [line.rstrip('\n').split() for line in open('video5_player3.txt', 'r')]
player5_4 = [line.rstrip('\n').split() for line in open('video5_player4.txt', 'r')]

oddFrames5 = len(lines5_2)
fcount5 = oddFrames5 * 3 - 1
players5 = 4

coords5 = np.zeros([fcount5, 6, 2])
player_coords5 = np.zeros([players5, fcount5, 2])

for i in range(fcount5 / 2):
    # player_coords5[0][2*i] = player5_1[i]
    player_coords5[1][2*i] = player5_2[i]
    player_coords5[2][2*i] = player5_3[i]
    player_coords5[3][2*i] = player5_4[i]
    if i < fcount5 / 2 - 1:
        # player_coords5[0][2*i+1] = player5_1[i]
        player_coords5[1][2*i+1] = player5_2[i]
        player_coords5[2][2*i+1] = player5_3[i]
        player_coords5[3][2*i+1] = player5_4[i]

for i in range(len(lines5_2)):
    coords5[3*i] = [[-1,-1], [-1,-1], lines5_2[i], lines5_3[i], lines5_4[i], lines5_5[i]]
    if i < (oddFrames5 - 2):
        coords5[3*i+1] = coords5[3*i]
    if i < (oddFrames5 - 1):
        coords5[3*i+2] = coords5[3*i]

# outputFiles5 = convert2TopDown(oddFrames5, fcount5, 4, players5, coords5, player_coords5)

# for i in range(players5):
#     filename = "video5_player" + str(i+1) + "_topdown.txt"
#     np.savetxt(filename, outputFiles5[i], delimiter=' ', fmt='%i %i')
# Video 5 end
