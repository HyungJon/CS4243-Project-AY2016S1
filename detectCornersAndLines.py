import numpy as np
import cv2
import math
from operator import itemgetter

class detectCornersAndLines:
    def __init__(self):
        True

    def findLowerUpper(self, colour, rng):
        lower = [0,0,0]
        upper = [0,0,0]
        for i in range(3):
            if colour[i] - rng < 0:
                lower[i] = 0
            else:
                lower[i] = colour[i] - rng
            if colour[i] + rng > 255:
                upper[i] = 255
            else:
                upper[i] = colour[i] + rng
        return (lower, upper)

    def displayResult(self, result):
        display = result
        cv2.namedWindow("Result")
        cv2.moveWindow("Result", 0, 0)
        while True:    
            cv2.imshow("Result", display)
            k = cv2.waitKey(0)
            if k == 27 or cv2.getWindowProperty('window-name', 0) < 0:
                break
        cv2.destroyAllWindows()


    def detect(self, image):
        row,col,cha = image.shape
        blackRow = np.zeros([col,3], dtype=np.uint8)

        
        ######## Step 1: Keep the sand pixels, black out the rest of the image ########
        print "Step 1: Keep the sand pixels"
        
        sandColour = [175, 216, 225]
        rng = 20
        lowerUpper = self.findLowerUpper(sandColour, rng)  # Get the range of colours that are considered as 'sand'
        lower = lowerUpper[0]
        upper= lowerUpper[1]
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        sandOnly = image.copy()
        k = 2      # Determine size of sliding window
        thres = 1  # score threshold for whether pixel is sand

        # Using a sliding window, determine for every pixel, if the pixel is likely to be surrounded by sand
        for r in range(k, row-k-1):
            for c in range(k, col-k-1):
                window = image[r-k:r+k, c-k:c+k]
                windowCorners = [window[0,0], window[0,k], window[k,k], window[k,0]]  # the 4 corners of the window
                score = 0
                for wincorn in windowCorners:
                    inRange = wincorn[0] > lower[0] and wincorn[1] > lower[1] and wincorn[2] > lower[2] \
                          and wincorn[0] < upper[0] and wincorn[1] < upper[1] and wincorn[2] < upper[2]
                    if inRange:
                        score += 1
                if score < thres:  #  if score is lower than threshold, black out pixel
                    sandOnly[r,c] = [0,0,0]

        # Black out the border pixels for now (since sliding window can't center on them)
        for r in range(k):
            sandOnly[r] = blackRow
        for r in range(row-k-1, row):
            sandOnly[r] = blackRow
        for c in range(k):
            for r in range(row):
                sandOnly[r,c] = [0,0,0]
        for c in range(col-k-1, col):
            for r in range(row):
                sandOnly[r,c] = [0,0,0]



        ######## Step 2: Keep the part of the image that is within sand, black out the rest of the image ########
        print "Step 2: Keep the part of the image that is within sand"
         
        court = sandOnly.copy()
        
        # For each row of pixels, find where the sand starts and ends
        sandStart = [-1] * row
        sandEnd = [-1] * row
        
        # Find the starting point of sand (scan from left to right)
        for r in range(row):
            if np.array_equal(sandOnly[r], blackRow):
                continue
            else:
                hasSand = 0
                for c in range(col):
                    bgr = court[r,c]
                    if bgr[0] != 0 and bgr[1] != 0 and bgr[2] != 0:
                        hasSand += 1
                    else:
                        hasSand = 0
                    if hasSand == 10:  # if 10 pixels in a row is sand, we consider the leftmost one as starting point
                        sandStart[r] = c-9
                        break
                
        # Find the ending point of sand (scan from right to left)
        for r in range(row):
            if np.array_equal(sandOnly[r], blackRow):
                continue
            else:
                hasSand = 0
                for c in range(col-1, -1, -1):
                    bgr = court[r,c]
                    if bgr[0] != 0 and bgr[1] != 0 and bgr[2] != 0:
                        hasSand += 1
                    else:
                        hasSand = 0
                    if hasSand == 10:  # if 10 pixels in a row is sand, we consider the rightmost one as ending point
                        sandEnd[r] = c+9
                        break

        # Infer if border pixels are part of sand (if yes, include them in)
        # For top and bottom borders, just take sandStart/End from nearest row
        for r in range(k):
            sandStart[r] = sandStart[k+1]
            sandEnd[r] = sandEnd[k+1]
        for r in range(row-k-1, row):
            sandStart[r] = sandStart[row-k-2]
            sandEnd[r] = sandEnd[row-k-2]
        # For left and right borders, check if adjacent pixel is part of sand
        for r in range(k+1, row-k-1):
            if sandStart[r] <= k+1 and sandStart[r] != -1:
                sandStart[r] = 0
            if sandEnd[r] >= col-k-2:
                sandEnd[r] = col-1

        #From the original image, keep only the pixels that are within sand, black out the rest
        for r in range(row):
            if sandStart[r] == -1:
                continue
            else:
                for s in range(sandStart[r], sandEnd[r]+1):
                    court[r,s] = image[r,s]



        ######## Step 3: Black out the sand and the olympics logo ########
        print "Step 3: Black out the sand and the olympics logo"
        
        noSandAndLogo = court.copy()
                    
        rng = 25
        lowerUpper = self.findLowerUpper(sandColour, rng)  # Get the range of colours that are considered as 'sand'
        lower = lowerUpper[0]
        upper = lowerUpper[1]
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # Create an extract where only the sand pixels are kept
        mask = cv2.inRange(court, lower, upper)
        extract = cv2.bitwise_and(court, court, mask = mask)

        # Use info from extract to black out all the sand pixels
        for r in range(row):
            if np.count_nonzero(extract[r] == 0) > (col*3 - 10):
                continue
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] != 0 and bgr[1] != 0 and bgr[2] != 0:
                        noSandAndLogo[r,c] = [0,0,0]

        # Black out olympic logo
        logox1,logoy1 = 579,257
        logox2,logoy2 = 624,293

        for r in range(logoy1, logoy2+1):
            for c in range(logox1, logox2+1):
                noSandAndLogo[r,c] = [0,0,0]



        ######## Step 4: Find the bottoms of the yellow net poles ########
        print "Step 4: Find the bottoms of the yellow net poles"
                
        yellowColour = (14,215,238)
        rng = 75
        lowerUpper = self.findLowerUpper(yellowColour, rng)  # Get the range of colours that are considered as 'yellow'
        lower = lowerUpper[0]
        upper = lowerUpper[1]
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # Create an extract where only the yellow pixels are kept
        mask = cv2.inRange(court, lower, upper)
        extract = cv2.bitwise_and(court, court, mask = mask)

        # Use info from extract to black out all the non-yellow pixels
        yellowPolesOnly = noSandAndLogo.copy()
        for r in range(row):
            if np.array_equal(extract[r], blackRow):
                yellowPolesOnly[r] = blackRow
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        yellowPolesOnly[r,c] = [0,0,0]
                    
        # Scan each row of pixels to find the bottom of each yellow pole
        poleBorders = []
        poleMids = []
        poleCount = 0
        
        for r in range(row-1, -1, -1):
            if np.array_equal(yellowPolesOnly[r], blackRow):
                continue
            if poleCount >= 3:
                break
            
            hasYellow = 0
            hasBlack = 0
            isScanningPole = False
            cStart = -1
            cEnd = -1

            c = -1
            while c < col-1:
                c += 1
                bgr = yellowPolesOnly[r,c]
                
                if not np.array_equal(bgr,[0,0,0]):
                    for pb in poleBorders:
                        if c >= pb[0] and c < pb[1]:
                            c = pb[1]
                            hasYellow = 0
                    else:
                        hasYellow += 1
                    hasBlack = 0
                else:
                    hasYellow = 0
                    hasBlack += 1
                    
                if hasYellow == 5 and not isScanningPole:  # if 5 pixels in a row is yellow, means yellow pole is found
                    cStart = c-4
                    isScanningPole = True
                    
                elif isScanningPole and hasBlack >= 5:  # if 5 pixels in a row is black, means is not yellow pole anymore
                    isScanningPole = False
                    cEnd = c-4
                    cMid = int(round(cStart+(cEnd-cStart)/2.0))
                    poleMids.append((cMid,r))
                    cv2.circle(yellowPolesOnly, (cMid,r), 4, (255,170,0), -1)

                    rr = r
                    oldr = r
                    sides = [cStart,cEnd]
                    while True:
                        hasBlack = 0
                        while hasBlack < 5:
                            if np.array_equal(yellowPolesOnly[rr,cMid],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            rr -= 1
                        rr += 6

                        if abs(rr-oldr) < 3: break
                        
                        cc = cMid+1
                        hasBlack = 0
                        while hasBlack < 5:
                            if np.array_equal(yellowPolesOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc += 1
                        cEnd = cc-6
                        sides.append(cEnd)
                        
                        cc = cMid-1
                        hasBlack = 0
                        while hasBlack < 5:
                            if np.array_equal(yellowPolesOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc -= 1
                        cStart = cc+6
                        sides.append(cStart)

                        cMid = int(round(cStart+(cEnd-cStart)/2.0))
                        oldr = rr
                        
                    cStart = min(sides)-3
                    cEnd = max(sides)+3
                    cv2.line(yellowPolesOnly, (cStart,rr), (cStart,r), (0,255,0), 1)
                    cv2.line(yellowPolesOnly, (cEnd,rr), (cEnd,r), (0,255,0), 1)
                    
                    poleBorders.append((cStart,cEnd))
                    poleCount += 1
        
        netPolePoints = []
        if len(poleMids) == 2:
            for mid in poleMids:
                cv2.circle(yellowPolesOnly, (mid[0],mid[1]), 4, (255,170,0), -1)
                cv2.circle(image, (mid[0],mid[1]), 4, (255,170,0), -1)
        elif len(poleMids) == 3:
            poleMids = sorted(poleMids,key=itemgetter(0))
            # Determine which 2 posts are the net poles based on their distances apart
            # The referee post will be very close to one of the net poles, while the 2 net poles should be far apart
            if abs(poleMids[0][0] - poleMids[1][0]) > 50 or abs(poleMids[0][1] - poleMids[1][1]) > 50:
                x1,y1 = poleMids[0]
                x2,y2 = poleMids[1]
            else:
                x1,y1 = poleMids[1]
                x2,y2 = poleMids[2]
            cv2.circle(image, (x1,y1), 4, (255,170,0), -1)
            cv2.circle(image, (x2,y2), 4, (255,170,0), -1)
            netPolePoints.append((x1,y1))
            netPolePoints.append((x2,y2))
        else:
            print "Too many poles detected!"
        

        
        ######## Step 5: Keep the purple pixels ########
        print "Step 5: Keep the purple pixels"
            
        purpleColour = [146,140,187]
        rng = 45
        lowerUpper = self.findLowerUpper(purpleColour, rng)
        lower = lowerUpper[0]
        upper = lowerUpper[1]
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(noSandAndLogo, lower, upper)
        extract = cv2.bitwise_and(noSandAndLogo, noSandAndLogo, mask = mask)
        
        purplePixels = noSandAndLogo.copy()
        for r in range(row):
            if np.count_nonzero(extract[r] == 0) > (col*3 - 10):
                purplePixels[r] = blackRow
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        purplePixels[r,c] = [0,0,0]


        
        ######## Step 6: Detect straight lines ########
        print "Step 6: Detect straight lines"
        
        straightLines = np.zeros(image.shape, dtype = np.uint8)
        
        # Use Hough Line Transform to detect straight lines from purple pixels
        gray = cv2.cvtColor(purplePixels,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        thres = 80
        lines = cv2.HoughLines(edges,1,np.pi/180,thres)
        
        lineGrps = []
        angGrps = []
        avgAng = []
        avgP = []
        
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            if x1 < x2:
                leftP = x1
                rightP = x2
            else:
                leftP = x2
                rightP = x1
            if y1 < y2:
                topP = y1
                bottomP = y2
            else:
                topP = y2
                bottomP = y1
            thisP = [leftP,topP,rightP,bottomP]
            
            # Group lines that are very similar in angles
            if x2-x1 == 0:
                angle = 0
            else:
                angle = math.atan(abs(y2-y1)*1.0 / abs(x2-x1)*1.0)
                
            gotGroup = False
            angleThres = 0.15
            distThres = 200
            for i in range(len(angGrps)):
                dist = 0
                for j in range(4):
                    dist += abs(avgP[i][j] - thisP[j])
                if abs(avgAng[i] - angle) < angleThres and dist < distThres:
                    angGrps[i].append(angle)
                    lineGrps[i].append((x1,y1,x2,y2))
                    avgAng[i] = sum(angGrps[i]) / len(angGrps[i])
                    newP = [0,0,0,0]
                    for j in range(4):
                        newP[j] = (avgP[i][j]*(len(angGrps[i])-1) + thisP[j]) / len(angGrps[i])
                    avgP[i] = newP
                    gotGroup = True
                    break
                
            if not gotGroup:
                angGrps.append([angle])
                avgAng.append(angle)
                lineGrps.append([(x1,y1,x2,y2)])
                avgP.append([leftP,topP,rightP,bottomP])

            cv2.line(straightLines,(x1,y1),(x2,y2),(0,0,255),2)  # Draw straight lines



        ######## Step 7: Find the lines that best fit each court line ########
        print "Step 7: Find the lines that best fit each court line"
        
        # For each line group, keep only the best line (line that best represents the group)
        bestFitLines = np.zeros(image.shape, dtype = np.uint8)
        bestFitLinesCoors = []
        
        for i in range(len(angGrps)):
            closest = lineGrps[i][0]
            minDiff = abs(angGrps[i][0] - avgAng[i])
            if len(angGrps[i]) > 1:
                for j in range(1, len(angGrps[i])):
                    diff = abs(angGrps[i][j] - avgAng[i])
                    if diff < minDiff:
                        minDiff = diff
                        closest = lineGrps[i][j]
            bestFitLinesCoors.append(closest)
            
        for line in bestFitLinesCoors:
            cv2.line(bestFitLines,(line[0],line[1]),(line[2],line[3]),(0,255,0),2)  # Draw best line



        ######## Step 8: Attempt to plot the true court lines based on what has been detected ########
        # (This step is not important, can be skipped completely)
        '''
        print "Step 8: Attempt to plot the true court lines"

        trueLines = np.zeros(image.shape, dtype = np.uint8)
        
        # Draw the true court lines (by keeping only the purple pixels that are in the straight lines)
        for i in range(len(lineGrps)):
            for line in lineGrps[i]:
                cv2.line(trueLines,(line[0],line[1]),(line[2],line[3]),(255,0,255),2)
        for r in range(row):
            for c in range(col):
                bgr = purplePixels[r,c]
                if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                    trueLines[r,c] = [0,0,0]
        '''
        


        ######## Step 9: Use the best-fit lines to find the court corners ########
        print "Step 9: Use the best-fit lines to find the court corners"
                    
        gray = cv2.cvtColor(bestFitLines, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, 10, 0.3, 10, blockSize=5)
        courtCorners = []
        if features is not None:
            features = np.int0(features)
            goodFeatures = []
            goodCount = 0
            
            for feat in features:
                x,y = feat.ravel()
                if sandStart[y] == -1:
                    #cv2.circle(bestFitLines,(x,y), 4, (0,255,255), -1)
                    continue
                else:
                    cv2.circle(bestFitLines,(x,y), 4, (0,255,255), -1)
                    goodFeatures.append((x,y))
                    goodCount += 1

            if goodCount == 1:
                bestX,bestY = goodFeatures[0]
                courtCorners.append((bestX,bestY))
                cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)
                
            elif goodCount == 2:
                x1,y1 = goodFeatures[0]
                x2,y2 = goodFeatures[1]
                diff = abs(x1-x2)+abs(y1-y2)
                if diff < 50:
                    #Take mid-point of the 2 features as the corner
                    if x1 > x2:
                        temp = x1
                        x1 = x2
                        x2 = temp
                    if y1 > y2:
                        temp = y1
                        y1 = y2
                        y2 = temp
                    bestX = int(round(x1 + (x2 - x1)/2.0))
                    bestY = int(round(y1 + (y2 - y1)/2.0))
                    courtCorners.append((bestX,bestY))
                    cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)
                else:
                    #Take both features as corners
                    print x1,y1
                    print x2,y2
                    cv2.circle(image,(x1,y1), 6, (0,0,255), -1)
                    cv2.circle(image,(x2,y2), 6, (0,0,255), -1)
                    
            elif goodCount == 3:
                #Find the first pair of features
                x1,y1 = goodFeatures[0]
                hasPair = False
                for i in range(2):
                    diff = abs(x1-goodFeatures[i+1][0])+abs(y1-goodFeatures[i+1][1])
                    if diff < 50:
                        x2,y2 = goodFeatures[i+1]
                        del goodFeatures[i+1]
                        del goodFeatures[0]
                        hasPair = True
                        break
                if not hasPair:
                    x1,y1 = goodFeatures[1]
                    x2,y2 = goodFeatures[2]
                if x1 > x2:
                    temp = x1
                    x1 = x2
                    x2 = temp
                if y1 > y2:
                    temp = y1
                    y1 = y2
                    y2 = temp
                bestX = int(round(x1 + (x2 - x1)/2.0))
                bestY = int(round(y1 + (y2 - y1)/2.0))
                courtCorners.append((bestX,bestY))
                cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)

                #Find the lone feature
                bestX,bestY = goodFeatures[0]
                courtCorners.append((bestX,bestY))
                cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)
                
            elif goodCount == 4:
                #Find the first pair of features
                x1,y1 = goodFeatures[0]
                for i in range(3):
                    diff = abs(x1-goodFeatures[i+1][0])+abs(y1-goodFeatures[i+1][1])
                    if diff < 50:
                        x2,y2 = goodFeatures[i+1]
                        del goodFeatures[i+1]
                        del goodFeatures[0]
                        break
                if x1 > x2:
                    temp = x1
                    x1 = x2
                    x2 = temp
                if y1 > y2:
                    temp = y1
                    y1 = y2
                    y2 = temp
                bestX = int(round(x1 + (x2 - x1)/2.0))
                bestY = int(round(y1 + (y2 - y1)/2.0))
                courtCorners.append((bestX,bestY))
                cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)

                #Find the second pair of features
                x1,y1 = goodFeatures[0]
                x2,y2 = goodFeatures[1]
                if x1 > x2:
                    temp = x1
                    x1 = x2
                    x2 = temp
                if y1 > y2:
                    temp = y1
                    y1 = y2
                    y2 = temp
                bestX = int(round(x1 + (x2 - x1)/2.0))
                bestY = int(round(y1 + (y2 - y1)/2.0))
                courtCorners.append((bestX,bestY))
                cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)



        ######## Step 10: Plot the corner points and best-fit lines on one image ########
        print "Step 10: Plot the corner points and best-fit lines on one image"

        linesAndCorners = np.zeros([row,col,cha], dtype=np.uint8)
        for line in bestFitLinesCoors:
            cv2.line(linesAndCorners, (line[0],line[1]), (line[2],line[3]), (0,255,0), 2)
        for corn in courtCorners:
            cv2.circle(linesAndCorners, corn, 6, (0,0,255), -1)
        for pt in netPolePoints:
            cv2.circle(linesAndCorners, pt, 6, (255,255,0), -1)



        print "\n------ DETECTION RESULT ------"
        print "Court corners found:"
        if len(courtCorners) == 0:
            print "nil"
        for corn in courtCorners:
            print corn
        print "Net pole points found:"
        if len(netPolePoints) == 0:
            print "nil"
        for pt in netPolePoints:
            print pt
        print "Total number of corner points found:", len(courtCorners)+len(netPolePoints)
        print "------------------------------"
        print "Best-fit court lines found:"
        if len(bestFitLinesCoors) == 0:
            print "nil"
        for line in bestFitLinesCoors:
            print "({},{}), ({},{})".format(line[0],line[1],line[2],line[3])
        print "Number of best fit court lines found:", len(bestFitLinesCoors), "\n"

        result = np.vstack([np.hstack([sandOnly, court, noSandAndLogo]), np.hstack([yellowPolesOnly, purplePixels, straightLines]), np.hstack([bestFitLines, linesAndCorners, image])])
        cv2.imwrite('linesAndCorners.jpg', linesAndCorners)
        result = cv2.resize(result, None, fx=0.85, fy=0.85)
        self.displayResult(result)

        return (courtCorners, netPolePoints, bestFitLinesCoors)
    
        
if __name__ == '__main__':
    dc = detectCornersAndLines()
    image = cv2.imread('frame3.jpg')
    data = dc.detect(image)
