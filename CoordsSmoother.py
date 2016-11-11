def smooth(list):
    for i in range(2, len(list)-2):
        list[i] = int((list[i-2] + list[i-1] + list[i] + list[i+1] + list[i+2])/5)
    return list


def smoothFile(filePath, outPath, coordsCount):
    target2 = open(outPath, 'w')
    target1 = open(filePath, 'r')
    x = []
    y = []
    for i in range(coordsCount):
        Str = target1.readline()
        x.append(int(Str.split()[0]))
        y.append(int(Str.split()[1]))

    x = smooth(x)
    y = smooth(y)

    for i in range(coordsCount):
        target2.write(str(x[i]))
        target2.write(" ")
        target2.write(str(y[i]))
        target2.write("\n")
    
