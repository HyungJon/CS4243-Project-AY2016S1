import numpy as np
import cv2
import cv2.cv as cv
import math
import numpy.linalg as la

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

def trackPlayerDist(coords_new, coords_last, dist):
    displacement = abs(math.hypot((coords_new[0] - coords_last[0]), (coords_new[1] - coords_last[1])))
    dist = dist + displacement * 0.0333
    # 1m = 30px => each pixel represents 3.33cm
    return dist

def drawTopDown(fcount, playerCount, players):
    player_dist = np.zeros(playerCount)

    for fr in range(fcount):
        topdown = drawCourt(np.zeros([540,780,3]))
        print "Frame " + str(fr+1)

        if fr > 0:
            for i in range(playerCount):
                player_dist[i] = trackPlayerDist(players[i][fr], players[i][fr-1], player_dist[i])
            
        for i in range(playerCount):
            print "Distance moved by player " + str(i+1) + ": " + str(player_dist[i]) + "m"
            cv2.circle(topdown, (int(players[i][fr][0]), int(players[i][fr][1])), int(15), (255,0,0), int(-1))
        cv2.imshow('Top-down view frame', cv2.convertScaleAbs(topdown))
        cv2.waitKey(10)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# later change these files to topdown_stabilized
player1 = [line.rstrip('\n').split() for line in open('video1_player1_topdown_smooth.txt', 'r')]
player2 = [line.rstrip('\n').split() for line in open('video1_player2_topdown_smooth.txt', 'r')]
player3 = [line.rstrip('\n').split() for line in open('video1_player3_topdown_smooth.txt', 'r')]
player4 = [line.rstrip('\n').split() for line in open('video1_player4_topdown_smooth.txt', 'r')]
playersPts = [player1, player2, player3, player4]

fcount = len(player1)
playerCount = len(playersPts)

players = np.zeros([playerCount, fcount, 2])

for i in range(len(players)):
    for j in range(len(players[i])):
        players[i][j][0] = int(playersPts[i][j][0])
        players[i][j][1] = int(playersPts[i][j][1])

drawTopDown(fcount, playerCount, players)

