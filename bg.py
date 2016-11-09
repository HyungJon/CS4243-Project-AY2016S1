import os
import cv2
import cv2.cv as cv
import numpy as np

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def isBackgroundPixel(pixel, bgPixel, threshold):
	if bgPixel[0] - threshold <= pixel[0] <= bgPixel[0] + threshold and \
	bgPixel[1] - threshold <= pixel[1] <= bgPixel[1] + threshold and \
	bgPixel[2] - threshold <= pixel[2] <= bgPixel[2] + threshold:
		return True
	else:
		return False
	
def imgThresholding(differences, img, rWeight = 1.0 / 3, gWeight = 1.0 / 3, bWeight = 1.0 / 3, fgThreshold = 40, bgThreshold = 20):
	newImg = np.copy(img)
	for r in range(img.shape[0]):
		for c in range(img.shape[1]):
			totalDifference = bWeight * differences[r][c][0] + gWeight * differences[r][c][1] + rWeight * differences[r][c][2]
			if r <= 1.0 / 5 * img.shape[0]:
				thresholdToUse = bgThreshold # more lenient threshold for objects in the background
			else:
				thresholdToUse = fgThreshold

			if totalDifference <= thresholdToUse: # only sufficiently different pixels are considered as foreground objects
				newImg[r][c] = [0, 0, 0] # replace background with black color

	return newImg
				
cap = cv2.VideoCapture("traffic.mp4")
cap.open("traffic.mp4")
frameWidth = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
print "frame width = ", frameWidth
frameHeight = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
print "frame height = ", frameHeight
fps = cap.get(cv.CV_CAP_PROP_FPS)
print "frames per second = ", fps
frameCount = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
print "frame count = ", frameCount
frameWidth = int(frameWidth)
frameHeight = int(frameHeight)
fps = int(fps)
frameCount = int(frameCount)

_, img = cap.read() 
avgImg = np.float32(img) # destination array
normImg = np.zeros(avgImg.shape)
for fr in range(1, frameCount):
	alpha = 1.0 / (fr + 1)
	_, img = cap.read()
	cv2.accumulateWeighted(img, avgImg, alpha)
	#cv2.imshow('img', img)
	#cv2.imshow('normImg', normImg)
	#print "fr = ", fr, "alpha = ", alpha

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
normImg = cv2.convertScaleAbs(avgImg)
cv2.imwrite("background.jpg", normImg)

cap = cv2.VideoCapture("traffic.mp4")
cap.open("traffic.mp4")

# static background subtraction
#for fr in range(frameCount):
	#_, img = cap.read()
	#img = np.float32(img) 
	#differences = np.fabs(img - normImg) # subtract background from current frame
	#img = imgThresholding(differences, img)
	#cv2.imwrite("img_" + str(fr) + ".jpg", img)
	
# frame differencing
alpha = 0.7
_, prevImg = cap.read()
prevImg = np.float32(prevImg)
for fr in range(1, frameCount):
	_, currImg = cap.read()
	currImg = np.float32(currImg) 
	differences = np.fabs(currImg - prevImg) # treat previous frame as background
	img = imgThresholding(differences, currImg)
	cv2.imwrite("img_" + str(fr) + ".jpg", img)
	prevImg = alpha * currImg + (1.0 - alpha) * prevImg

cap.release()
