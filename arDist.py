import cv2
import numpy as np

spaceSize = 0.8 #in
codeSize = 7.0 #in
boardSize = (6, 9)


def createKnownBoardPos(boardSize, spaceSize, corners):
    for i in range(boardSize[1]):
        for j in range(boardSize[0]):
            corners.append((j*spaceSize, i * spaceSize, 0))

def getChessboardCorners(images, boardSize, foundCorners, showResults):
    for image in images:
        pointBuff = []
        found = cv2.findChessboardCorners(image, boardSize, pointBuff, cv2.CALIB_CB_ADAPTIVE_THRESH or cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            foundCorners.append(pointBuff)
        if showResults:
            cv2.drawChessboardCorners(image, boardSize, pointBuff, found)
            cv2.imshow("looking", image)
            cv2.waitKey(0)


def calcDist(img, codes, markerSize, horFov, verFov):
    row, col, lay = img.shape

    for code in codes:
        points = code[0]
        topDiff = points[2][0][0] - points[3][0][0]
        botDiff = points[0][0][0] - points[1][0][0]
        leftDiff = points[1][0][1] - points[3][0][1]
        rightDiff = points[0][0][1] - points[2][0][1]
        print(topDiff, botDiff, leftDiff, rightDiff)

        BRHAng = points[0][0][0] / col * horFov
        BLHAng = points[1][0][0] / col * horFov
        TRHAng = points[2][0][0] / col * horFov
        TLHAng = points[3][0][0] / col * horFov
        print(BRHAng, BLHAng, TRHAng, TLHAng)

cameraMatrix = np.eye(3, 3, cv2.CV_64F)
distCoef = np.zeros((1080,720,3))
savedFrames = []
markerCorners = []
rejectedCorners = []
vid = cv2.VideoCapture(0)
framesPerSec = 20
cv2.namedWindow("Webcam", cv2.WINDOW_AUTOSIZE)

while True:

    ret, frame = vid.read()
    drawFrame = frame.copy()

    if not(frame.any()):
        break

    foundPoints = np.array([])
    found = True


    cv2.findChessboardCorners(frame, boardSize, foundPoints, cv2.CALIB_CB_NORMALIZE_IMAGE or cv2.CALIB_CB_ADAPTIVE_THRESH)
    cv2.copyTo(frame, drawFrame)
    cv2.drawChessboardCorners(drawFrame, boardSize, foundPoints, found)
    if found:
        cv2.imshow("Webcam", drawFrame)
    else:
        cv2.imshow("Webcam", frame)
    char = cv2.waitKey(int(1000 / framesPerSec))

