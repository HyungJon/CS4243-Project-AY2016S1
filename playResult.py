import numpy as np
import cv2
import glob

srcFolder = "beachVolleyball1_leftNetpole"

cv2.namedWindow('Result')
stop = False
while not stop:
    for frame in glob.glob(srcFolder+"/*.jpg"):
        img = cv2.imread(frame)
        cv2.imshow('Result', img)
        k = cv2.waitKey(25)
        if k == 27:
            stop = True
            break
cv2.destroyAllWindows()
