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

