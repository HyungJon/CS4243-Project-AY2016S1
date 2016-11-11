import numpy as np

lines1 = [line.rstrip('\n') for line in open('video1_rightCourtCorner.txt', 'r')]
lines0 = [line.rstrip('\n') for line in open('video1_leftCourtCorner.txt', 'r')]
lines4 = [line.rstrip('\n') for line in open('video1_rightNetpole.txt', 'r')]
lines5 = [line.rstrip('\n') for line in open('video1_leftNetpole.txt', 'r')]

print len(lines1)

coords = np.zeros([len(lines1)*2 - 1, 6, 2])

for i in range(len(lines1)):
    coords[2*i] = [lines0[i].split(), lines1[i].split(), [-1,-1], [-1,-1], lines4[i].split(), lines5[i].split()]
    if i < len(lines1) - 1:
        coords[2*i+1] = coords[2*i]

# print coords
