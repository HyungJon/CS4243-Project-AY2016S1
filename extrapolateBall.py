def readCoords(filePath, coordsCount):
    ballCoords = []
    target = open(filePath, 'r')
    for i in range(coordsCount):
        Str = target.readline()
        frameIndex = int(Str[:3])
        x = int(Str[4:].split()[0])
        y = int(Str[4:].split()[1])
        ballCoords.append([frameIndex, x, y])
    return ballCoords


def extrapolateBallCoords(ballCoords, frameCount):
    completeCoords = []
    thisFrame = 1
    for i in range(ballCoords[0][0]):
        completeCoords.append([ballCoords[0][1], ballCoords[0][2]])
    for i in range(len(ballCoords)-1):
        x_interval = (ballCoords[i+1][1] - ballCoords[i][1])/(ballCoords[i+1][0] - ballCoords[i][0])
        y_interval = (ballCoords[i+1][2] - ballCoords[i][2])/(ballCoords[i+1][0] - ballCoords[i][0])
        for j in range(ballCoords[i+1][0]-ballCoords[i][0]):
            completeCoords.append([j*x_interval + ballCoords[i][1], j*y_interval + ballCoords[i][2]])
    for i in range(ballCoords[len(ballCoords) - 1][0], frameCount):
        completeCoords.append([ballCoords[len(ballCoords - 1)][1], ballCoords[len(ballCoords - 1)][2]])
    return completeCoords


coords = readCoords("test.txt", 3)
print(extrapolateBallCoords(coords, 347))
