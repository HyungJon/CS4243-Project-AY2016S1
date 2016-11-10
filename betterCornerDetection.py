import numpy as np
import cv2
import math
from operator import itemgetter
import sys
import os
import glob

class betterCornerDetection:
    def __init__(self, srcFolder):
        self.dstFolder = srcFolder+"_results"
        if not os.path.exists(self.dstFolder):
            os.makedirs(self.dstFolder)
        self.log = file(self.dstFolder+'/log.txt', 'w')
        self.corner = None
        self.cornerPicked = False
        self.imageshape = None
        self.cornerType = None
        self.refreshDisplay = False

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

    def chooseOption(self, image):
        def clickOption(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                row,col,cha = self.imageshape
                if y < row/2:
                    self.cornerType = 'court'
                else:
                    self.cornerType = 'pole'
        
        display = image
        cv2.namedWindow("Pick Corner Type")
        cv2.setMouseCallback("Pick Corner Type", clickOption)
        cv2.moveWindow("Pick Corner Type", 0, 0)
        while self.cornerType is None:    
            cv2.imshow("Pick Corner Type", display)
            cv2.waitKey(10)
        cv2.destroyAllWindows()
        
    def pickCorner(self, image):
        def getPoint(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.corner = (y,x)
                self.cornerPicked = True
        
        display = image
        cv2.namedWindow("Pick Corner")
        cv2.setMouseCallback("Pick Corner", getPoint)
        cv2.moveWindow("Pick Corner", 0, 0)
        while not self.cornerPicked:    
            cv2.imshow("Pick Corner", display)
            cv2.waitKey(10)
        cv2.destroyAllWindows()

    def playResult(self):
        cv2.namedWindow('Result')
        stop = False
        while not stop:
            for frame in glob.glob(self.dstFolder+"/*.jpg"):
                img = cv2.imread(frame)
                cv2.imshow('Result', img)
                k = cv2.waitKey(25)
                if k == 27:  #PRESS ESCAPE TO CLOSE
                    stop = True
                    break
        cv2.destroyAllWindows()

    def closeLog(self):
        self.log.close()
        


    def detect(self, imageFile, cropBound, prevFrameCropCorner, prevFrameRealCorner):
        image = cv2.imread(imageFile)
        row,col,cha = image.shape
        self.imageshape = image.shape
        print "Detecting for", imageFile, "..."

        orig_stdout = sys.stdout
        sys.stdout = self.log

        if cropBound is None:
            courtCornerOption = np.zeros([row/2,col,cha], dtype=np.uint8)
            courtCornerOption[:] = (128, 0, 128)
            cv2.putText(courtCornerOption, "Court Corner", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 255))
            poleCornerOption = np.zeros([row/2,col,cha], dtype=np.uint8)
            poleCornerOption[:] = (0, 255, 255)
            cv2.putText(poleCornerOption, "Pole Corner", (120,100), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(128, 0, 128))
            
            cornerOptions = np.vstack([courtCornerOption,poleCornerOption])
            self.chooseOption(cornerOptions)
            self.pickCorner(image)

        if self.cornerType == 'court':
            boundRange = 30
        elif self.cornerType == 'pole':
            boundRange = 20


        if cropBound is None:
            cornerR = self.corner[0]
            cornerC = self.corner[1]
            rTop = cornerR-boundRange
            if rTop < 0: rTop = 0
            rBottom = cornerR+boundRange
            if rBottom > row-1: rBottom = row-1
            cLeft = cornerC-boundRange
            if cLeft < 0: cLeft = 0
            cRight = cornerC+boundRange
            if cRight > col-1: cRight = col-1
        else:
            rTop, rBottom, cLeft, cRight = cropBound[0], cropBound[1], cropBound[2], cropBound[3]

        imageNoLogo = image.copy()
        logox1,logoy1 = 579,257
        logox2,logoy2 = 624,293
        for r in range(logoy1, logoy2+1):
            for c in range(logox1, logox2+1):
                imageNoLogo[r,c] = [0,0,0]

        crop = imageNoLogo[rTop:rBottom, cLeft:cRight]

        row,col,cha = crop.shape
        blackRow = np.zeros([col,3], dtype=np.uint8)

        if self.cornerType == 'pole':
            imagePole = crop.copy()

            yellowColour = (14,215,238)
            rng = 72
            lowerUpper = self.findLowerUpper(yellowColour, rng)
            lower = np.array(lowerUpper[0], dtype = "uint8")
            upper = np.array(lowerUpper[1], dtype = "uint8")

            mask = cv2.inRange(crop, lower, upper)
            extract = cv2.bitwise_and(crop, crop, mask = mask)

            yellowStart = None
            yellowEnd = None
            yellowMid = None
            yellowBottom = None
            for r in range(row-1, -1, -1):
                if np.count_nonzero(extract[r] == 0) > (col*3 * (9/10.0)):
                    continue
                else:
                    yellowCount = 0
                    blackCount = 0
                    isScanningYellow = False
                    for c in range(col):
                        bgr = extract[r,c]
                        if not isScanningYellow:
                            if bgr[0] != 0 or bgr[1] != 0 or bgr[2] != 0:
                                yellowCount += 1
                                if yellowCount >= 2:
                                    yellowStart = c-2
                                    isScanningYellow = True
                                    r -= 3
                            else:
                                yellowCount = 0
                        else:
                            if bgr[0] == 0 and bgr[1] == 0 and bgr[2] == 0:
                                blackCount += 1
                                if isScanningYellow and blackCount >= 5:
                                    yellowEnd = c-5
                                    break
                            else:
                                blackCount = 0
                            if isScanningYellow and c == col-1:
                                yellowEnd = col-1
                                break
                    yellowMid = yellowStart + (yellowEnd-yellowStart)/2
                    yellowBottom = r + 3
                    break
            cropCornerX = yellowMid
            cropCornerY = yellowBottom
            cv2.circle(imagePole, (cropCornerX,cropCornerY), 3, (255,0,0), -1)


        elif self.cornerType == 'court':
            ######## Step 5: Keep the purple pixels ########
                
            purpleColour = [146,140,187]
            rng = 48  #old:45,50,56,48(gd for mov1)
            lowerUpper = self.findLowerUpper(purpleColour, rng)
            lower = np.array(lowerUpper[0], dtype = "uint8")
            upper = np.array(lowerUpper[1], dtype = "uint8")

            mask = cv2.inRange(crop, lower, upper)
            extract = cv2.bitwise_and(crop, crop, mask = mask)

            purplePixels = crop.copy()
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
            #self.displayResult(edges)

            thres = 15  #old:75,63,85   (100 for part1&2, 85 for rest)
            
            idealAccuracyThres = 2.1
            minAccuracyThres = 2
            minLineThres = 0
            maxLineThres = 50
            finalThres = None
            increasingThres = False
            decreasingThres = False
            optimalThresFound = False

            lineGrps = []
            angGrps = []
            avgAng = []
            avgP = []

            while thres > minLineThres and thres < maxLineThres:
                lines = cv2.HoughLines(edges,1,np.pi/180,thres)
                straightLines = np.zeros(crop.shape, dtype = np.uint8)
                
                lineGrps = []
                angGrps = []
                avgAng = []
                avgP = []

                if lines is not None:
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
                            angle = math.atan((y2-y1)*1.0 / (x2-x1)*1.0)
                            
                        gotGroup = False
                        angleThres = 0.26   #old:0.18
                        #distThres = 10  #old:200,175
                        for i in range(len(angGrps)):
                            dist = 0
                            for j in range(4):
                                dist += abs(avgP[i][j] - thisP[j])
                            #if abs(avgAng[i] - angle) < angleThres and dist < distThres:
                            if abs(avgAng[i] - angle) < angleThres:
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
                #print thres, accuracyScore, len(angGrps)
                #self.displayResult(straightLines)
                if optimalThresFound:
                    finalThres = thres
                    thres = maxLineThres  #break out of while loop
                elif accuracyScore > idealAccuracyThres:
                    thres += 2
                    increasingThres = True
                    if decreasingThres:
                        optimalThresFound = True
                    if thres >= maxLineThres:
                        finalThres = maxLineThres
                elif accuracyScore < minAccuracyThres:
                    thres -= 2
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
            bestFitLines = np.zeros(crop.shape, dtype = np.uint8)
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
                cv2.line(bestFitLines,(line[0],line[1]),(line[2],line[3]),(0,255,0),1)

            ######## Step 8: Use the best-fit lines to find the court corners ########
                        
            goodFeatures = []
            courtCorners = []
            goodCount = None

            gray = cv2.cvtColor(bestFitLines, cv2.COLOR_BGR2GRAY)
            thres = 0.3  #old:0.3,0.4
                    
            features = cv2.goodFeaturesToTrack(gray, 50, thres, 10, blockSize=3)
            cropCornerX,cropCornerY = 0,0
            if features is not None:
                features = np.int0(features)
                cropCornerX,cropCornerY = features[0].ravel()
                courtCorners.append((cropCornerX,cropCornerY))
                cv2.circle(purplePixels,(cropCornerX,cropCornerY), 3, (0,0,255), -1)
            
            result = np.hstack([purplePixels, straightLines, bestFitLines])
            result = cv2.resize(result, (1002,150))

        movePercent = 0.4
        diffThres = boundRange*0.45
        realDiffThres = 50
        if prevFrameRealCorner is not None:
            x1 = min(cropCornerX, prevFrameCropCorner[0])
            x2 = max(cropCornerX, prevFrameCropCorner[0])
            y1 = min(cropCornerY, prevFrameCropCorner[1])
            y2 = max(cropCornerY, prevFrameCropCorner[1])
            xDiff = x2-x1
            yDiff = y2-y1
            xRealDiff = max(cLeft+cropCornerX, prevFrameRealCorner[0]) - min(cLeft+cropCornerX, prevFrameRealCorner[0])
            yRealDiff = max(rTop+cropCornerY, prevFrameRealCorner[1]) - min(rTop+cropCornerY, prevFrameRealCorner[1])
            #print xDiff, yDiff, xRealDiff, yRealDiff, imageFile,
            if (xDiff > diffThres or yDiff > diffThres) or (xRealDiff > realDiffThres or yRealDiff > realDiffThres):
                #print 'bad'
                cropCornerX = prevFrameCropCorner[0]
                cropCornerY = prevFrameCropCorner[1]
                realCornerX = prevFrameRealCorner[0]
                realCornerY = prevFrameRealCorner[1]
                crop = image.copy()[rTop:rBottom, cLeft:cRight]
            else:
                #print ''
                xDiff = xDiff*movePercent
                yDiff = yDiff*movePercent
                cropCornerX = x1 + int((x2-x1)*movePercent)
                cropCornerY = y1 + int((y2-y1)*movePercent)
                realCornerX = cLeft + cropCornerX
                realCornerY = rTop + cropCornerY
        else:
            realCornerX = cLeft + cropCornerX
            realCornerY = rTop + cropCornerY
        
        cv2.circle(image,(realCornerX,realCornerY), 5, (0,0,255), -1)
        #largeImage = cv2.resize(image, (1002,450))

        if self.cornerType == 'court':
            #result = np.vstack([result, image])
            result = image
        else:
            result = image
        
        cv2.imwrite(self.dstFolder+"\\"+imageFile.split('\\')[1].split('.')[0]+"_result.jpg", image)

        row,col,cha = self.imageshape
        rTop = realCornerY-boundRange
        if rTop < 0: rTop = 0
        rBottom = realCornerY+boundRange
        if rBottom > row-1: rBottom = row-1
        cLeft = realCornerX-boundRange
        if cLeft < 0: cLeft = 0
        cRight = realCornerX+boundRange
        if cRight > col-1: cRight = col-1
        nextCropBound = (rTop,rBottom,cLeft,cRight)
        
        print realCornerX, realCornerY
        sys.stdout = orig_stdout
        return nextCropBound, (cropCornerX,cropCornerY), (realCornerX,realCornerY)
        
        
        
if __name__ == '__main__':
    srcFolder = "cornerTest"
    #srcFolder = "beachVolleyball1_temp"
    bcd = betterCornerDetection(srcFolder)

    startDetect = False
    cropBound = None
    cropCorner = None
    realCorner = None
    for frame in glob.glob(srcFolder+"/*.jpg"):
        cropBound,cropCorner,realCorner = bcd.detect(frame, cropBound, cropCorner, realCorner)
    
    bcd.closeLog()
    bcd.playResult()
