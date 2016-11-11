import cv2
import cv2.cv as cv

cap = cv2.VideoCapture("beachVolleyball1.mov")
fcount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

for fr in range(fcount):
    _, frame = cap.read()
    if fr % 2 == 0:
        filename = "frame"+str(fr+1)+".jpg"
        cv2.imwrite(filename, frame)
        cv2.waitKey(10)

cap.release()
