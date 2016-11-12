import numpy as np
import cv2
import glob

srcFolder = "topdown"

images = []
print "Reading frames, please wait..."
for frame in glob.glob(srcFolder+"/*.jpg"):
    images.append(cv2.imread(frame))

cv2.namedWindow('Result')
cv2.moveWindow('Result', 0,0)
stop = False
while not stop:
    for img in images:
        cv2.imshow('Result', img)
        k = cv2.waitKey(16)
        if k == 27:
            stop = True
            break
cv2.destroyAllWindows()
