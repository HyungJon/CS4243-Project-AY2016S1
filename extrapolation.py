import os
import cv2
import numpy as np
import PIL.ImageOps as ImageOps
import PIL.Image as Image
import copy

os.chdir("/Users/felix/pydir")
print(os.getcwd())


def generateBorder(filePath):
    img = Image.open(filePath)
    border = int(img.size[0] / 3)
    borderLeft = border
    w, h = img.size
    img = img.crop((5, 5, w-5, h-5))
    img_with_border = ImageOps.expand(img, border, fill=0)
    borderRight = img_with_border.size[0] - border
    img_with_border.save('tempWithBorder.jpg')
    img2 = cv2.imread('tempWithBorder.jpg')
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(borderLeft, borderRight)
    return gray, img2, borderLeft, borderRight


def generateBlackPicture(img):
    blackImg = copy.copy(img)
    for i in range(len(blackImg)):
        for j in range(len(blackImg[0])):
            blackImg[i][j] = [0, 0, 0]
    return blackImg


def areIdenticalLines(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    if rho1*0.9 < rho2 < rho1*1.1 and theta1*0.9 < theta2 < theta1*1.1:
        rho3 = (rho1+rho2)/2
        theta3 = (theta1+theta2)/2
        return True, [rho3, theta3]
    return False, [0, 0]


def processLines(lines):
    newLines = []
    linesx = []
    for i in range(len(lines)):
        linesx.append([lines[i][0][0], lines[i][0][1]])
    linesx = sorted(linesx)
    currentLine = linesx[0]
    for i in range(1, len(linesx)):
        valid, newline = areIdenticalLines(currentLine, linesx[i])
        if valid:
            currentLine = newline
        else:
            newLines.append(currentLine)
            currentLine = linesx[i]
    newLines.append(currentLine)
    print(newLines)
    return newLines


def extrapolateEdges(gray, blackImg):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    print(lines)
    newlines = processLines(lines)
    for i in range(len(newlines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * a)
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * a)
            cv2.line(blackImg, (x1, y1), (x2, y2), (255, 0, 255), 5)
    return cv2.cvtColor(blackImg, cv2.COLOR_BGR2GRAY)


def normalize(dst):
    max = dst[0][0]
    min = dst[0][0]
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            if dst[i][j] > max:
                max = dst[i][j]
            if dst[i][j] < min:
                min = dst[i][j]
    print(max, min)
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            dst[i][j] = (dst[i][j]-min)/(max - min)*255
    return dst


def filterCorners(corners):
    newCorners = []
    currentx = corners[0][0]
    currenty = corners[0][1]
    for i in range(1, len(corners)):
        x, y = corners[i]
        if currentx*0.9 < x < currentx*1.1 and currenty*0.9 < y < currenty*1.1:
            currentx = (currentx + x)/2
            currenty = (currenty + y)/2
        else:
            newCorners.append([currentx, currenty])
            currentx = x
            currenty = y
    x, y = corners[len(corners)-1]
    if currentx * 0.9 < x < currentx * 1.1 and currenty * 0.9 < y < currenty * 1.1:
        newCorners.append([currentx, currenty])
    return newCorners


def extractCorners(gray):
    corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 0.04)
    corners = np.int0(corners)
    cornerlist = []
    for i in corners:
        x, y = i.ravel()
        cornerlist.append([x, y])
    cornerlist = filterCorners(sorted(cornerlist))
    print(cornerlist)
    return cornerlist


def markCorners(corners, img):
    for i in range(len(corners)):
        x, y = corners[i]
        cv2.circle(img, (int(x), int(y)), 3, 255, -1)


def extrapolate(filePath):
    gray, img, borderLeft, borderRight = generateBorder(filePath)
    blackImg = generateBlackPicture(img)
    gray = extrapolateEdges(img, blackImg)
    corners = extractCorners(gray)
    markCorners(corners, gray)
    cv2.imwrite('houghlines3.jpg', gray)
    return corners

extrapolate("test11.jpg")
