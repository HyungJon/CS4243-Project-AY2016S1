import numpy as np
import cv2
import glob
import os

originalFrames = []
panoramaFrames = []
topdownFrames = []
statFrames = []

# Set from which frame to which frame to process (if too many frames, numpy may throw memory error)
start = 0
end = 300   

print "Reading frames, please wait...",
count = 0
for frame in glob.glob("frames/*.jpg"):
    count += 1
    if count >= start:
        originalFrames.append(cv2.imread(frame))
    if count == end: break
    
print "...",
count = 0
for frame in glob.glob("panorama/*.jpg"):
    count += 1
    if count >= start:
        panoramaFrames.append(cv2.imread(frame))
    if count == end: break
    
print "...",
count = 0
for frame in glob.glob("topdown/*.jpg"):
    count += 1
    if count >= start:
        topdownFrames.append(cv2.imread(frame))
    if count == end: break

print "..."
count = 0
for frame in glob.glob("stats/*.jpg"):
    count += 1
    if count >= start:
        statFrames.append(cv2.imread(frame))
    if count == end: break

dstfolder = "output"
if not os.path.exists(dstfolder):
    os.makedirs(dstfolder)
            
cv2.namedWindow("Result")
cv2.moveWindow("Result", 0, 0)
for i in range(end-start):
    fr1 = cv2.resize(originalFrames[i], (500,350))
    fr2 = cv2.resize(panoramaFrames[i], (860,350))
    fr3 = cv2.resize(topdownFrames[i], (500,350))
    fr4 = cv2.resize(statFrames[i], (860,350))
    result = np.vstack((np.hstack((fr1,fr2)),(np.hstack((fr3,fr4)))))

    cv2.imshow("Result", result)
    if i+start < 10: n = "00"+str(i+start)
    elif i+start < 100: n = "0"+str(i+start)
    else: n = str(i+start)
    cv2.imwrite("output\\"+n+".jpg", result)
    cv2.imshow("Result", result)
    k = cv2.waitKey(15)
    '''if k == 27:
        break'''
cv2.destroyAllWindows()
