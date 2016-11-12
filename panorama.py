import numpy as np
import cv2
import cv2.cv as cv
import math
import numpy.linalg as la

extendX = 1200  #extra width after adding black border
extendY = 700   #extra height after adding black border

leftX = 380  #width of left black border
rightX = extendX - leftX   #width of right black border
topY = 310   #width of top black border 
bottomY = extendY - topY   #width of bottom black border 

warpSizeX = 1600  #Width of warp image
warpSizeY = 1200  #Height of warp image

# The coords of the four court corners and the 2 net poles
# if coord is (-1, -1), means we are not using the point for homography
target_coords = np.array([[122.0, 285.0], [596.0, 245.0], [0.0, 0.0], [0.0, 0.0], [448.0, 124.0], [163.0, 130.0]])

def computeHomography(pts):
    M = np.zeros([8,8])
    b = np.zeros([8])
    
    idx = 0
    for i in range(len(pts)):
        if pts[i][0] != -1:
            x = pts[i][0]+leftX # original x-coord: up
            y = pts[i][1]+topY # original y-coord: vp

            u = target_coords[i][0]+leftX # transformed x-coord: uc
            v = target_coords[i][1]+topY # transformed y-coord: vc

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

dstFolder = "panorama"
if not os.path.exists(dstFolder):
    os.makedirs(dstFolder)

#This is the image that will store the panorama
#We first run it one time to get the panorama, then we run it again so that we can see the player's actions in the panorama
stitchedWarps = np.zeros([500, 1000, 3], dtype=np.uint8)   #in the first run, start with an empty image
#stitchedWarps = cv2.imread('panorama.jpg')   #in the second run, start with the panorama from the first run

blackRow = np.zeros([1000,3], dtype=np.uint8)
for fr in range(len(coords)):
    _, frame = cap.read()
    h = computeHomography(coords[fr])
    if (fr+1) % 2 != 1:
        continue
    
    print fr+1,
    imageWithBlack = np.zeros([frame.shape[0]+extendX, frame.shape[1]+extendY, frame.shape[2]], dtype=np.uint8)
    imageWithBlack[leftX:imageWithBlack.shape[0]-rightX, topY:imageWithBlack.shape[1]-bottomY] = frame
    
    warp = cv2.warpPerspective(imageWithBlack, h, (warpSizeX, warpSizeY))
    print "...",
    warp = cv2.resize(warp, (1000, 500))

    #Scan row by row to find non-black pixels
    #We want to keep the non-black pixels and add them to the panorama
    row,col,cha = warp.shape
    for r in range(row):
        if np.array_equal(warp[r], blackRow):
            continue
        else:
            scanningFrame = False
            notBlack = 0
            isBlack = 0
            c = 0
            newPixels = []
            frameStart = -1
            while c < col:
                bgr = warp[r,c]
                if scanningFrame:
                    newPixels.append(bgr)
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        isBlack += 1
                        if isBlack == 10:
                            i = 0
                            for cc in range(frameStart, c-15):
                                stitchedWarps[r,cc] = newPixels[i]
                                i += 1
                            break
                elif bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                    notBlack += 1
                    if notBlack == 5:  
                        scanningFrame = True
                        frameStart = c+1
                c += 1

    print "..."

    result = stitchedWarps
    
    if (fr+1) < 10: cv2.imwrite(dstFolder+"\\00"+str(fr+1)+".jpg", result)
    elif (fr+1) < 100: cv2.imwrite(dstFolder+"\\0"+str(fr+1)+".jpg", result)
    else: cv2.imwrite(dstFolder+"\\"+str(fr+1)+".jpg", result)
    
cap.release()
