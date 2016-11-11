import cv2
import cv2.cv as cv

cap = cv2.VideoCapture("beachVolleyball1.mov")
fcount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

for fr in range(fcount):
    _, frame = cap.read()
    if fr % 2 == 0:
        filenum = str(fr+1)
        if fr < 99:
            filenum = "0" + filenum
            if fr < 9:
                filenum = "0" + filenum
        filename = "frame"+filenum+".jpg"
        cv2.imwrite(filename, frame)
        cv2.waitKey(10)

cap.release()
