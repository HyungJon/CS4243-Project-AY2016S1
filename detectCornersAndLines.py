import numpy as np
import cv2
import math
from operator import itemgetter
import sys
import os
import glob

class detectCornersAndLines:
    def __init__(self, srcFolder):
        self.dstFolder = srcFolder+"_results"
        if not os.path.exists(self.dstFolder):
            os.makedirs(self.dstFolder)
        self.log = file(self.dstFolder+'/log.txt', 'w')

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

    def findGreenLowerUpper(self, colour, rng):
        lower = [0,0,0]
        upper = [0,0,0]
        blueRng = int(rng*1.2)
        if colour[0] - blueRng < 0:
            lower[0] = 0
        else:
            lower[0] = colour[0] - blueRng
        if colour[0] + blueRng > 255:
            upper[0] = 255
        else:
            upper[0] = colour[0] + blueRng

        for i in range(1,3):
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

    def closeLog(self):
        self.log.close()


    def detect(self, imageFile):
        image = cv2.imread(imageFile)
        row,col,cha = image.shape
        blackRow = np.zeros([col,3], dtype=np.uint8)
        print "Detecting for", imageFile, "..."

        orig_stdout = sys.stdout
        sys.stdout = self.log


        
        ######## Step 1: Keep the sand pixels, black out the rest of the image ########
        
        sandColour = [175, 216, 225]
        rng = 20
        lowerUpper = self.findLowerUpper(sandColour, rng)  # Get the range of colours that are considered as 'sand'
        lower = np.array(lowerUpper[0], dtype = "uint8")
        upper = np.array(lowerUpper[1], dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)
        extract = cv2.bitwise_and(image, image, mask = mask)

        sandOnly = image.copy()

        for r in range(row):
            if np.count_nonzero(extract[r] == 0) > (col*3 * (3/4.0)):  # if more than 3/4 of the rgb values are 0
                sandOnly[r] = blackRow
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        sandOnly[r,c] = [0,0,0]

                        

        ######## Step 2: Keep the part of the image that is within sand, black out the rest of the image ########
         
        court = sandOnly.copy()
        
        # For each row of pixels, find where the sand starts and ends
        sandStart = [-1] * row
        sandEnd = [-1] * row
        sandThres = 5
        
        # Find the starting point of sand (scan from left to right)
        for r in range(row):
            if np.array_equal(sandOnly[r], blackRow):
                continue
            else:
                hasSand = 0
                for c in range(col):
                    bgr = court[r,c]
                    if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                        hasSand += 1
                    else:
                        hasSand = 0
                    if hasSand == sandThres:  # if n pixels in a row is sand, we consider the leftmost one as starting point
                        sandStart[r] = c-(sandThres-1)
                        break
                
        # Find the ending point of sand (scan from right to left)
        for r in range(row):
            if np.array_equal(sandOnly[r], blackRow):
                continue
            else:
                hasSand = 0
                for c in range(col-1, -1, -1):
                    bgr = court[r,c]
                    if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                        hasSand += 1
                    else:
                        hasSand = 0
                    if hasSand == sandThres:  # if n pixels in a row is sand, we consider the rightmost one as ending point
                        sandEnd[r] = c+(sandThres-1)
                        break

        yellowColour = (14,215,238)
        rng = 72
        lowerUpper = self.findLowerUpper(yellowColour, rng)  # Get the range of colours that are considered as 'yellow'
        lower = np.array(lowerUpper[0], dtype = "uint8")
        upper = np.array(lowerUpper[1], dtype = "uint8")

        # Create an extract from orignal image where only the yellow pixels are kept
        mask = cv2.inRange(image, lower, upper)
        extract = cv2.bitwise_and(image, image, mask = mask)

        # Extend the start/end of sand to include the yellow poles
        for r in range(row):
            if sandStart[r] == -1:
                continue
            else:
                yellowCount = 0
                blackCount = 0
                for s in range(sandStart[r], -1, -1):
                    bgr = extract[r,s]
                    if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                        yellowCount += 1
                        if yellowCount >= 3:
                            sandStart[r] = s
                    else:
                        blackCount += 1
                        if blackCount >= 10:
                            if s > 25:
                                break
                            else:
                                sandStart[r] += 5
                                break
                yellowCount = 0
                blackCount = 0
                for s in range(sandEnd[r], col-1):
                    bgr = extract[r,s]
                    if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                        yellowCount += 1
                        if yellowCount >= 3:
                            sandEnd[r] = s
                    else:
                        blackCount += 1
                        if blackCount >= 10:
                            if (col-1)-s < 25:
                                break
                            else:
                                sandEnd[r] -= 5
                                break

        # Keep the pixels that are within the start/end of sand, black out the rest
        for r in range(row):
            if sandStart[r] == -1:
                continue
            else:
                for s in range(sandStart[r], sandEnd[r]+1):
                    court[r,s] = image[r,s]

        #Black out olympics logo
        logox1,logoy1 = 579,257
        logox2,logoy2 = 624,293
        for r in range(logoy1, logoy2+1):
            for c in range(logox1, logox2+1):
                court[r,c] = [0,0,0]



        ######## Step 3: Find the bottoms of the yellow net poles ########

        mask = cv2.inRange(court, lower, upper)
        extract = cv2.bitwise_and(court, court, mask = mask)

        # Use info from extract to black out all the non-yellow pixels
        yellowPolesOnly = court.copy()
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
        yellowThres = 5
        horiBlackThres = 3
        vertBlackThres = 5
        poleLimitThres = 3
        
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
            relevantBorders = []
            for pb in poleBorders:
                if r >= pb[2] and r <= pb[3]:
                    relevantBorders.append(pb)

            c = -1
            while c < col-1:
                c += 1
                bgr = yellowPolesOnly[r,c]
                
                if not np.array_equal(bgr,[0,0,0]):
                    for pb in relevantBorders:
                        if c >= pb[0] and c < pb[1]:
                            c = pb[1]
                            hasYellow = 0
                    else:
                        hasYellow += 1
                    hasBlack = 0
                else:
                    hasYellow = 0
                    hasBlack += 1
                    
                if hasYellow == yellowThres and not isScanningPole:  # if N pixels in a row is yellow, means yellow pole is found
                    cStart = c-(yellowThres-1)
                    isScanningPole = True
                    
                elif isScanningPole and hasBlack >= horiBlackThres:  # if N pixels in a row is black, means is not yellow pole anymore
                    isScanningPole = False
                    cEnd = c-(horiBlackThres-1)
                    cMid = int(round(cStart+(cEnd-cStart)/2.0))
                    poleMids.append((cMid,r))
                    bottomMid = cMid

                    rr = r
                    oldr = r
                    sides = [cStart,cEnd]
                    while True:
                        hasBlack = 0
                        rowScanned = 0
                        while hasBlack < vertBlackThres and rr < row:
                            if np.array_equal(yellowPolesOnly[rr,cMid],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            rr -= 1
                            rowScanned += 1
                        if rr == row:
                            rr += rowScanned+1
                        else:
                            rr += vertBlackThres+1

                        if abs(rr-oldr) < poleLimitThres: break
                        
                        cc = cMid+1
                        hasBlack = 0
                        while hasBlack < horiBlackThres:
                            if np.array_equal(yellowPolesOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc += 1
                        cEnd = cc-(horiBlackThres+1)
                        sides.append(cEnd)
                        
                        cc = cMid-1
                        hasBlack = 0
                        while hasBlack < horiBlackThres:
                            if np.array_equal(yellowPolesOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc -= 1
                        cStart = cc+(horiBlackThres+1)
                        sides.append(cStart)

                        cMid = int(round(cStart+(cEnd-cStart)/2.0))
                        oldr = rr
                        
                    cStart = min(sides) - poleLimitThres
                    cEnd = max(sides) + poleLimitThres
                    rStart = rr - poleLimitThres
                    rEnd = r + poleLimitThres

                    if rEnd - rStart <= 10:
                        break
                    cv2.rectangle(yellowPolesOnly, (cStart,rStart), (cEnd,rEnd), (0,255,0), 1)
                    cv2.circle(yellowPolesOnly, (bottomMid,r), 4, (255,170,0), -1)
                    poleBorders.append((cStart,cEnd, rStart, rEnd))
                    poleCount += 1
        
        netPolePoints = []
        if len(poleMids) <= 2:
            if len(poleMids) == 2:
                if poleMids[0][0] < poleMids[1][0]:
                    pole1 = poleMids[0]
                    pole2 = poleMids[1]
                else:
                    pole1 = poleMids[1]
                    pole2 = poleMids[0]
                if pole2[0] - pole1[0] <= 100 and (pole2[1] - pole1[1] > 50 or pole2[1] - pole1[1] < 15):
                    if pole1[0] < col/2:
                        poleMids.remove(pole1)
                    else:
                        poleMids.remove(pole2)
            for mid in poleMids:
                netPolePoints.append((mid[0],mid[1]))
                
        elif len(poleMids) == 3:
            # Determine which 2 posts are the net poles based on their distances apart
            # The referee post will be very close to one of the net poles, while the 2 net poles should be far apart
            if abs(poleMids[0][0] - poleMids[1][0]) < 50 and abs(poleMids[0][0] - poleMids[2][0]) < 50:
                poleMids = sorted(poleMids,key=itemgetter(1))
                if abs(poleMids[0][0] - poleMids[1][0]) > 50 or abs(poleMids[0][1] - poleMids[1][1]) > 50:
                    x1,y1 = poleMids[0]
                    x2,y2 = poleMids[1]
                else:
                    x1,y1 = poleMids[1]
                    x2,y2 = poleMids[2]
            else:
                poleMids = sorted(poleMids,key=itemgetter(0))
                if abs(poleMids[0][0] - poleMids[1][0]) > 50 or abs(poleMids[0][1] - poleMids[1][1]) > 50:
                    x1,y1 = poleMids[0]
                    x2,y2 = poleMids[1]
                else:
                    x1,y1 = poleMids[1]
                    x2,y2 = poleMids[2]
            netPolePoints.append((x1,y1))
            netPolePoints.append((x2,y2))
        else:
            print "Too many poles detected!"

        # Discard the point if net pole is blocked by some object (e.g. the player)
        notBlocked = []
        for pt in netPolePoints:
            sandScore = 0
            r,c = pt[1],pt[0]
            for i in range(10):
                bgr = sandOnly[r,c]
                if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                    sandScore += 1
                r += 1
                if r >= row: break
                if i == 9:
                    cc = c-2
                    if cc >= 0:
                        bgr = sandOnly[r,cc]
                        if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                            sandScore += 1
                    cc = c+2
                    if cc <= col-1:
                        bgr = sandOnly[r,cc]
                        if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                            sandScore += 1
            if sandScore >= 2:
                notBlocked.append(pt)
                cv2.circle(image, (pt[0],pt[1]), 4, (255,170,0), -1)
        netPolePoints = notBlocked
        


        ######## Step 4: Roughly mark out the players in the image ########

        markPlayers = image.copy()          
        borders = []

        # Find the players that are wearing the green shirts
        greenColour = (50,174,125)
        rng = 44  #old:50
        lowerUpper = self.findGreenLowerUpper(greenColour, rng)
        lower = np.array(lowerUpper[0], dtype = "uint8")
        upper = np.array(lowerUpper[1], dtype = "uint8")

        mask = cv2.inRange(court, lower, upper)
        extract = cv2.bitwise_and(court, court, mask = mask)

        greenShirtsOnly = court.copy()
        for r in range(row):
            if np.array_equal(extract[r], blackRow):
                greenShirtsOnly[r] = blackRow
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        greenShirtsOnly[r,c] = [0,0,0]

        count = 0
        greenThres = 9
        horiBlackThres = 9
        vertBlackThres = 20
        limitThres = 3
        horiExtend = 3/5.0
        vertExtendUp = 2/5.0
        vertExtendDown = 4/3.0
        
        for r in range(0, row):
            if np.array_equal(greenShirtsOnly[r], blackRow):
                continue
            if count >= 2:
                break
            
            hasGreen = 0
            hasBlack = 0
            isScanning = False
            cStart = -1
            cEnd = -1
            relevantBorders = []
            for b in borders:
                if r >= b[2] and r <= b[3]:
                    relevantBorders.append(b)

            c = -1
            while c < col-1:
                c += 1
                bgr = greenShirtsOnly[r,c]
                
                if not np.array_equal(bgr,[0,0,0]):
                    for b in relevantBorders:
                        if c >= b[0] and c < b[1]:
                            c = b[1]
                            hasGreen = 0
                    else:
                        hasGreen += 1
                    hasBlack = 0
                else:
                    hasGreen = 0
                    hasBlack += 1
                    
                if hasGreen == greenThres and not isScanning:
                    cStart = c-(greenThres-1)
                    isScanning = True
                    
                elif isScanning and hasBlack >= horiBlackThres:
                    isScanning = False
                    cEnd = c-(horiBlackThres-1)
                    cMid = int(round(cStart+(cEnd-cStart)/2.0))

                    rr = r
                    oldr = r
                    sides = [cStart,cEnd]
                    while True:
                        hasBlack = 0
                        rowScanned = 0
                        while hasBlack < vertBlackThres and rr < row:
                            if np.array_equal(greenShirtsOnly[rr,cMid],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            rr += 1
                            rowScanned += 1
                        if rr == row:
                            rr -= rowScanned+1
                        else:
                            rr -= vertBlackThres+1

                        if abs(rr-oldr) < limitThres: break
                        
                        cc = cMid+1
                        hasBlack = 0
                        while hasBlack < horiBlackThres:
                            if np.array_equal(greenShirtsOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc += 1
                        cEnd = cc-(horiBlackThres+1)
                        sides.append(cEnd)
                        
                        cc = cMid-1
                        hasBlack = 0
                        while hasBlack < horiBlackThres:
                            if np.array_equal(greenShirtsOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc -= 1
                        cStart = cc+(horiBlackThres+1)
                        sides.append(cStart)

                        cMid = int(round(cStart+(cEnd-cStart)/2.0))
                        oldr = rr

                    if rr-r <= 10: continue
                        
                    cStart = min(sides) - limitThres
                    cEnd = max(sides) + limitThres
                    rStart = r - limitThres
                    rEnd = rr + limitThres

                    cExtend = int((cEnd-cStart)*horiExtend)
                    cStart = cStart - cExtend
                    cEnd = cEnd + cExtend
                    if cStart < 0: cStart = 0
                    if cEnd > col-1: cEnd = col-1

                    rExtendUp = int((rEnd-rStart)*vertExtendUp)
                    rExtendDown = int((rEnd-rStart)*vertExtendDown)
                    rStart = rStart - rExtendUp
                    rEnd = rEnd + rExtendDown
                    if rStart < 0: rStart = 0
                    if rEnd > row-1: rEnd = row-1
                    
                    cv2.rectangle(markPlayers, (cStart,rStart), (cEnd,rEnd), (0,255,0), 1)

                    borders.append((cStart, cEnd, rStart, rEnd))
                    count += 1

        # Find the players that are wearing the white shirts (dark pants)
        pantsColour = (54,41,55)
        rng = 30
        lowerUpper = self.findLowerUpper(pantsColour, rng)
        lower = np.array(lowerUpper[0], dtype = "uint8")
        upper = np.array(lowerUpper[1], dtype = "uint8")

        mask = cv2.inRange(court, lower, upper)
        extract = cv2.bitwise_and(court, court, mask = mask)
        
        darkPantsOnly = court.copy()
        for r in range(row):
            if np.array_equal(extract[r], blackRow):
                darkPantsOnly[r] = blackRow
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        darkPantsOnly[r,c] = [0,0,0]

        count = 0
        pantsThres = 9
        horiBlackThres = 9
        vertBlackThres = 20
        limitThres = 3
        horiExtend = 3/4.0
        vertExtendUp = 2.0
        vertExtendDown = 2.0
        
        for r in range(0, row):
            if np.array_equal(darkPantsOnly[r], blackRow):
                continue
            if count >= 2:
                break
            
            hasPants = 0
            hasBlack = 0
            isScanning = False
            cStart = -1
            cEnd = -1
            relevantBorders = []
            for b in borders:
                if r >= b[2] and r <= b[3]:
                    relevantBorders.append(b)

            c = -1
            while c < col-1:
                c += 1
                bgr = darkPantsOnly[r,c]
                
                if not np.array_equal(bgr,[0,0,0]):
                    for b in relevantBorders:
                        if c >= b[0] and c < b[1]:
                            c = b[1]
                            hasPants = 0
                    else:
                        hasPants += 1
                    hasBlack = 0
                else:
                    hasPants = 0
                    hasBlack += 1
                    
                if hasPants == pantsThres and not isScanning:
                    cStart = c-(pantsThres-1)
                    isScanning = True
                    
                elif isScanning and hasBlack >= horiBlackThres:
                    isScanning = False
                    cEnd = c-(horiBlackThres-1)
                    cMid = int(round(cStart+(cEnd-cStart)/2.0))

                    rr = r
                    oldr = r
                    sides = [cStart,cEnd]
                    while True:
                        hasBlack = 0
                        rowScanned = 0
                        while hasBlack < vertBlackThres and rr < row:
                            if np.array_equal(darkPantsOnly[rr,cMid],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            rr += 1
                            rowScanned += 1
                        if rr == row:
                            rr -= rowScanned+1
                        else:
                            rr -= vertBlackThres+1

                        if abs(rr-oldr) < limitThres: break
                        
                        cc = cMid+1
                        hasBlack = 0
                        while hasBlack < horiBlackThres:
                            if np.array_equal(darkPantsOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc += 1
                        cEnd = cc-(horiBlackThres+1)
                        sides.append(cEnd)
                        
                        cc = cMid-1
                        hasBlack = 0
                        while hasBlack < horiBlackThres:
                            if np.array_equal(darkPantsOnly[rr,cc],[0,0,0]):
                                hasBlack += 1
                            else:
                                hasBlack = 0
                            cc -= 1
                        cStart = cc+(horiBlackThres+1)
                        sides.append(cStart)

                        cMid = int(round(cStart+(cEnd-cStart)/2.0))
                        oldr = rr
                        
                    cStart = min(sides) - limitThres
                    cEnd = max(sides) + limitThres
                    rStart = r - limitThres
                    rEnd = rr + limitThres

                    cExtend = int((cEnd-cStart)*horiExtend)
                    cStart = cStart - cExtend
                    cEnd = cEnd + cExtend
                    if cStart < 0: cStart = 0
                    if cEnd > col-1: cEnd = col-1

                    rExtendUp = int((rEnd-rStart)*vertExtendUp)
                    rExtendDown = int((rEnd-rStart)*vertExtendDown)
                    rStart = rStart - rExtendUp
                    rEnd = rEnd + rExtendDown
                    if rStart < 0: rStart = 0
                    if rEnd > row-1: rEnd = row-1
                    
                    cv2.rectangle(markPlayers, (cStart,rStart), (cEnd,rEnd), (0,255,0), 1)

                    borders.append((cStart, cEnd, rStart, rEnd))
                    count += 1
        
        # Black out the players (so that the next step will have less noise)
        noPlayers = court.copy()
        for b in borders:
            cStart, cEnd, rStart, rEnd = b[0], b[1], b[2], b[3]
            blackBox = np.zeros([rEnd+1-rStart,cEnd+1-cStart,3], dtype=np.uint8)
            noPlayers[rStart:rEnd+1, cStart:cEnd+1] = blackBox
        


        ######## Step 5: Keep the purple pixels ########
            
        purpleColour = [146,140,187]
        rng = 48  #old:45,50,56,48(gd for mov1)
        lowerUpper = self.findLowerUpper(purpleColour, rng)
        lower = np.array(lowerUpper[0], dtype = "uint8")
        upper = np.array(lowerUpper[1], dtype = "uint8")

        mask = cv2.inRange(noPlayers, lower, upper)
        extract = cv2.bitwise_and(noPlayers, noPlayers, mask = mask)

        purplePixels = noPlayers.copy()
        for r in range(row):
            if np.array_equal(extract[r], blackRow):
                purplePixels[r] = blackRow
            else:
                for c in range(col):
                    bgr = extract[r,c]
                    if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                        purplePixels[r,c] = [0,0,0]


        
        ######## Step 6: Detect straight lines ########
        
        # Use Hough Line Transform to detect straight lines from purple pixels
        gray = cv2.cvtColor(purplePixels,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)  #old:50,150
        
        thres = 55  #old:75,63,85   (100 for part1&2, 85 for rest)
        
        idealAccuracyThres = 3.07
        minAccuracyThres = 3
        minLineThres = 40
        maxLineThres = 160
        finalThres = None
        increasingThres = False
        decreasingThres = False
        optimalThresFound = False

        lineGrps = None
        angGrps = None
        avgAng = None
        avgP = None

        while thres > minLineThres and thres < maxLineThres:
            lines = cv2.HoughLines(edges,1,np.pi/180,thres)
            straightLines = np.zeros(image.shape, dtype = np.uint8)
            
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
                angleThres = 0.36   #old:0.18
                distThres = 170  #old:200,175
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
            
            accuracyScore = 0
            for grp in angGrps:
                minAng = min(grp)
                maxAng = max(grp)
                accuracyScore += 1 + (maxAng-minAng)
            if optimalThresFound:
                finalThres = thres
                thres = maxLineThres  #break out of while loop
            elif accuracyScore > idealAccuracyThres:
                thres += 5
                increasingThres = True
                if decreasingThres:
                    optimalThresFound = True
                if thres >= maxLineThres:
                    finalThres = maxLineThres
            elif accuracyScore < minAccuracyThres:
                thres -= 5
                decreasingThres = True
                if increasingThres:
                    optimalThresFound = True
                if thres <= minLineThres:
                    finalThres = minLineThres
            else:
                finalThres = thres
                thres = maxLineThres  #break out of while loop

        

        ######## Step 7: Find the lines that best fit each court line ########
        
        # For each line group, keep only the best line (line that best represents the group)
        bestFitLines = np.zeros(image.shape, dtype = np.uint8)
        bestFitLinesCoors = []
        
        for i in range(len(angGrps)):
            closest = lineGrps[i][0]
            minDiff = abs(angGrps[i][0] - avgAng[i])
            for j in range(1, len(angGrps[i])):
                diff = abs(angGrps[i][j] - avgAng[i])
                if diff < minDiff:
                    minDiff = diff
                    closest = lineGrps[i][j]
            bestFitLinesCoors.append(closest)
                
        for line in bestFitLinesCoors:
            cv2.line(bestFitLines,(line[0],line[1]),(line[2],line[3]),(0,255,0),3)  # Draw best line



        ######## Step 8: Use the best-fit lines to find the court corners ########
                    
        goodFeatures = []
        courtCorners = []
        goodCount = None

        gray = cv2.cvtColor(bestFitLines, cv2.COLOR_BGR2GRAY)
        thres = 0.3  #old:0.3,0.4
        
        while goodCount is None or goodCount > 4:
            if goodCount is None:
                goodCount = 0
            if goodCount > 4:
                thres += 0.1
                goodCount = 0
                del goodFeatures[:]
                
            features = cv2.goodFeaturesToTrack(gray, 50, thres, 10, blockSize=3)
            if features is not None:
                features = np.int0(features)
                
                for feat in features:
                    x,y = feat.ravel()
                    if sandStart[y] == -1:
                        continue
                    else:
                        goodFeatures.append((x,y))
                        goodCount += 1

                if goodCount >= 1 and goodCount <= 4:
                    for feat in goodFeatures:
                        cv2.circle(bestFitLines,(feat[0],feat[1]), 4, (0,255,255), -1)
                        
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
                    x1,y1 = goodFeatures[0]
                    neighbours = []
                    notNeighbours = []
                    for i in range(1,3):
                        diff = abs(x1-goodFeatures[i][0])+abs(y1-goodFeatures[i][1])
                        if diff < 50:
                            neighbours.append(goodFeatures[i])
                        else:
                            notNeighbours.append(goodFeatures[i])
                            
                    if len(neighbours) == 2:  # all 3 features are clustered together
                        xlist = sorted([x1, neighbours[0][0], neighbours[1][0]])
                        x1,x2,x3 = xlist[0],xlist[1],xlist[2]
                        ylist = sorted([y1, neighbours[0][1], neighbours[1][1]])
                        y1,y2,y3 = ylist[0],ylist[1],ylist[2]
                        bestX = int(round(x1 + (x3 - x1)/2.0))
                        bestY = int(round(y1 + (y3 - y1)/2.0))
                        courtCorners.append((bestX,bestY))
                        cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)
                    
                    elif len(neighbours) <= 1:  # 2 features are clustered together
                        if len(neighbours) == 1:  
                            x2,y2 = neighbours[0]
                        else:
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

                        #The lone feature
                        if len(neighbours) == 1:
                            bestX,bestY = notNeighbours[0]
                        else:
                            bestX,bestY = goodFeatures[0]
                        courtCorners.append((bestX,bestY))
                        cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)
                    
                elif goodCount == 4:
                    x1,y1 = goodFeatures[0]
                    neighbours = []
                    notNeighbours = []
                    for i in range(1,4):
                        diff = abs(x1-goodFeatures[i][0])+abs(y1-goodFeatures[i][1])
                        if diff < 50:
                            neighbours.append(goodFeatures[i])
                        else:
                            notNeighbours.append(goodFeatures[i])

                    if len(neighbours) == 2 or len(neighbours) == 0:  # 3 features are clustered together
                        if len(neighbours) == 2:
                            xlist = sorted([x1, neighbours[0][0], neighbours[1][0]])
                            x1,x2,x3 = xlist[0],xlist[1],xlist[2]
                            ylist = sorted([y1, neighbours[0][1], neighbours[1][1]])
                            y1,y2,y3 = ylist[0],ylist[1],ylist[2]
                        else:
                            xlist = sorted([notNeighbours[0][0], notNeighbours[1][0], notNeighbours[2][0]])
                            x1,x2,x3 = xlist[0],xlist[1],xlist[2]
                            ylist = sorted([notNeighbours[0][1], notNeighbours[1][1], notNeighbours[2][1]])
                            y1,y2,y3 = ylist[0],ylist[1],ylist[2]
                        bestX = int(round(x1 + (x3 - x1)/2.0))
                        bestY = int(round(y1 + (y3 - y1)/2.0))
                        courtCorners.append((bestX,bestY))
                        cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)

                        #The lone feature
                        if len(neighbours) == 2:
                            bestX,bestY = notNeighbours[0]
                        else:
                            bestX,bestY = goodFeatures[0]
                        courtCorners.append((bestX,bestY))
                        cv2.circle(image,(bestX,bestY), 6, (0,0,255), -1)

                    elif len(neighbours) <= 1:  # 2 pairs of features
                        #First pair of features
                        x2,y2 = neighbours[0]
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

                        #Second pair of features
                        x1,y1 = notNeighbours[0]
                        x2,y2 = notNeighbours[1]
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

        # Plot the corner points and best-fit lines on one image
        linesAndCorners = np.zeros([row,col,cha], dtype=np.uint8)
        for line in bestFitLinesCoors:
            cv2.line(linesAndCorners, (line[0],line[1]), (line[2],line[3]), (0,255,0), 3)
        for corn in courtCorners:
            cv2.circle(linesAndCorners, corn, 6, (0,0,255), -1)
        for pt in netPolePoints:
            cv2.circle(linesAndCorners, pt, 6, (255,255,0), -1)



        print "\n------ DETECTION RESULT for {} ------".format(imageFile)
        print "Line accuracy score:", accuracyScore
        print "Threshold used:", finalThres
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

        empty = np.zeros([row,col,cha], dtype=np.uint8)
        result = np.vstack([np.hstack([court, yellowPolesOnly, greenShirtsOnly]), np.hstack([straightLines, purplePixels, markPlayers]), np.hstack([bestFitLines, linesAndCorners, image])])
        result = cv2.resize(result, None, fx=0.85, fy=0.85)
        cv2.imwrite(self.dstFolder+"\\"+imageFile.split('\\')[1].split('.')[0]+"_result.jpg", result)

        sys.stdout = orig_stdout
        return (courtCorners, netPolePoints, bestFitLinesCoors)
    
    
        
if __name__ == '__main__':
    srcFolder = "testFolder"
    #srcFolder = "beachVolleyball2_frames"
    dc = detectCornersAndLines(srcFolder)

    for frame in glob.glob(srcFolder+"/*.jpg"):
        data = dc.detect(frame)

    dc.closeLog()
