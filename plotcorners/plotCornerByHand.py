import numpy as np
import cv2
import math
from operator import itemgetter
import sys
import os
import glob

class plotCornerByHand:
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

    def plotCorner(self, image):
        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.corner = (x,y)
                self.refreshDisplay = True

        display = image.copy()
        if self.corner is not None:
            cv2.circle(display, self.corner, 2, (0,0,255), -1)
        cv2.namedWindow("Plot Corner")
        cv2.setMouseCallback("Plot Corner", click)
        cv2.moveWindow("Plot Corner", 0, 0)
        while True:
            if self.refreshDisplay == True:
                display = image.copy()
                cv2.circle(display, self.corner, 2, (0,0,255), -1)
                self.refreshDisplay = False
            cv2.imshow("Plot Corner", display)
            k = cv2.waitKey(10)
            if k == 13 and self.corner is not None:   #PRESS ENTER TO CONFIRM
                break
            #PRESS WASD TO ADJUST POINT
            elif k == 119:
                x,y = self.corner[0],self.corner[1]
                self.corner = (x,y-1)
                self.refreshDisplay = True
            elif k == 97:
                x,y = self.corner[0],self.corner[1]
                self.corner = (x-1,y)
                self.refreshDisplay = True
            elif k == 115:
                x,y = self.corner[0],self.corner[1]
                self.corner = (x,y+1)
                self.refreshDisplay = True
            elif k == 100:
                x,y = self.corner[0],self.corner[1]
                self.corner = (x+1,y)
                self.refreshDisplay = True
        cv2.destroyAllWindows()
        return display

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



    def plotByHand(self, imageFile):
        image = cv2.imread(imageFile)
        row,col,cha = image.shape
        self.imageshape = image.shape
        print "Plotting for", imageFile, "..."

        orig_stdout = sys.stdout
        sys.stdout = self.log

        imageWithBlack = np.zeros([self.imageshape[0]+200, self.imageshape[1]+200, self.imageshape[2]], dtype=np.uint8)
        imageWithBlack[100:imageWithBlack.shape[0]-100, 100:imageWithBlack.shape[1]-100] = image

        result = self.plotCorner(imageWithBlack)
        realCorner = (self.corner[0]-100,self.corner[1]-100)
        print realCorner[0], realCorner[1]
        
        cv2.imwrite(self.dstFolder+"\\"+imageFile.split('\\')[1].split('.')[0]+"_result.jpg", result)
        sys.stdout = orig_stdout




if __name__ == '__main__':
    srcFolder = "cornerTest"
    pcbh = plotCornerByHand(srcFolder)

    for frame in glob.glob(srcFolder+"/*.jpg"):
        pcbh.plotByHand(frame)
    
    pcbh.closeLog()
    pcbh.playResult()
