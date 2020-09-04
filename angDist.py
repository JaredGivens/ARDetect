import cv2
import numpy as np
import math

# target is list of target x, y
# ang and fov in radians
# returns target dist, horDist reletive to the second camera position
def getDist(img, target1, target2, ang, offset, horFov, verFov):

    col, row, lay = img.shape

    horDist, X, Z = getHorDist(col, target1[0], target2[0], ang, offset, horFov)
    directDist, Y = getVertDist(row, target2[1], horDist, verFov)

    return directDist

# returns target horazontal distance X, Z reletive to the second camera position
def getHorDist(col, target1x, target2x, ang, offset, horFov):

    camDist = ((2 * offset ** 2) * (1 - math.cos(ang))) ** 0.5
    camAng = (math.pi - ang) / 2

    ang1 = target1x if target1x > col / 2 else col -target1x
    ang2 = col - target2x if target1x > col / 2 else target2x

    ang1 = math.pi * 1.5 - (ang1 / col * horFov + (math.pi - horFov) / 2) - camAng
    ang2 = math.pi * 1.5 - (ang2 / col * horFov + (math.pi - horFov) / 2) - camAng
    ang3 = math.pi - ang1 - ang2
    dist = camDist * (math.sin(ang2) / math.sin(ang3))
    return dist, dist * math.cos(ang1), dist * math.sin(ang1)

# returns distance and object y reletive to second camera position
def getVertDist(row, target2y, horDist, verFov):
    return 1, 2

