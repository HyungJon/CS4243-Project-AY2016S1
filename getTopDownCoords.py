import numpy as np
import cv2
import cv2.cv as cv
import math
import numpy.linalg as la

videos = ["beachVolleyball1.mov"]
target_coords = np.array([[630.0, 390.0], [630.0, 150.0], [150.0, 150.0], [150.0, 390.0], [390.0, 120.0], [390.0, 420.0]])
# player_dist = np.zeros([4])

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

def convert2TopDown(oddFrames, fcount, players, coords, player_coords):
    cap = cv2.VideoCapture("beachVolleyball1.mov")

    for fr in range(fcount):
        # print fr
        _, frame = cap.read()
        h = getHomography(coords[fr])
        homogArray[fr] = h
        # topdown = cv2.warpPerspective(frame, h, (780, 540))
        topdown = np.zeros([540,780,3])
        topdown = drawCourt(topdown)

        for i in range(players):
            orig = [player_coords[i][fr][0], player_coords[i][fr][1], 1]
            res = np.dot(h, orig)
            player_coords[i][fr] = [int(res[0]/res[2]), int(res[1]/res[2])]
            # msg = "Distance moved by player " + str(i+1) + ": " + str(player_dist[i]) + "m"
            # print msg
            cv2.circle(topdown, (int(player_coords[i][fr][0]), int(player_coords[i][fr][1])), int(15), (255,0,0), int(-1))
            # filename = "td_fr"+str(fr+1)+".jpg"
            # cv2.imwrite(filename, topdown)
        cv2.imshow('Top-down view frame', cv2.convertScaleAbs(topdown))
        cv2.waitKey(10)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cap.release()

    return player_coords

# Video 1 start
lines0 = [line.rstrip('\n').split() for line in open('video1_point0.txt', 'r')]
lines1 = [line.rstrip('\n').split() for line in open('video1_point1.txt', 'r')]
lines3 = [line.rstrip('\n').split() for line in open('video1_point3.txt', 'r')]
lines4 = [line.rstrip('\n').split() for line in open('video1_point4.txt', 'r')]
lines5 = [line.rstrip('\n').split() for line in open('video1_point5.txt', 'r')]

player1 = [line.rstrip('\n').split() for line in open('video1_player1.txt', 'r')]
player2 = [line.rstrip('\n').split() for line in open('video1_player2.txt', 'r')]
player3 = [line.rstrip('\n').split() for line in open('video1_player3.txt', 'r')]
player4 = [line.rstrip('\n').split() for line in open('video1_player4.txt', 'r')]

oddFrames = len(lines0)
fcount = oddFrames*2-1
players = 4

coords = np.zeros([fcount, 6, 2])
player_coords = np.zeros([players, fcount, 2])
homogArray = np.zeros([fcount, 3, 3])

for i in range(len(player1)):
    player_coords[0][2*i] = player1[i]
    player_coords[1][2*i] = player2[i]
    player_coords[2][2*i] = player3[i]
    player_coords[3][2*i] = player4[i]
    if i < len(player1) - 1:
        player_coords[0][2*i+1] = player1[i]
        player_coords[1][2*i+1] = player2[i]
        player_coords[2][2*i+1] = player3[i]
        player_coords[3][2*i+1] = player4[i]

for i in range(len(lines1)):
    coords[2*i] = [lines0[i], lines1[i], [-1,-1], lines3[i], lines4[i], lines5[i]]
    if i < (oddFrames-1):
        coords[2*i+1] = coords[2*i]

outputFiles = convert2TopDown(oddFrames, fcount, players, coords, player_coords)

for i in range(players):
    filename = "video1_player" + str(i+1) + "_topdown.txt"
    np.savetxt(filename, outputFiles[i], delimiter=' ', fmt='%i %i')
# Video 1 end




        
    # if fr == 0:
    #     player_dist = [0,0,0,0]
    # else:
    #     player_dist = trackPlayerStats(fr, player_coords[fr], player_coords[fr-1], player_dist)


# for fr in range(len(coords)):
#     filename = "td_fr"+str(fr+1)+".jpg"
#     topdown = cv2.imread(filename)
#     ballPos = np.zeros(2)

  #   if fr < 403:
#         ballPos = player_coords[fr][3]
#     elif fr >= 403 and fr < 491:
#         ballPos = player_coords[fr][0]
#     elif fr >= 491 and fr < 583:
#         ballPos = player_coords[fr][1]
#     elif fr >= 583 and fr < 623:
#         ballPos = player_coords[fr][0]
#     else:
#         coords = np.dot(homographyArray[fr], [170,244,1])
#         ballPos = [coords[0]/coords[2], coords[1]/coords[2]]
#         print ballPos

#     cv2.circle(topdown, (int(ballPos[0]), int(ballPos[1])), int(8), (0,255,0), int(-1))


