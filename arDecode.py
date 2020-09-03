import cv2
import numpy as np
import funcs

img = cv2.imread("/Users/jared/Pictures/arcodes.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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



def avg9x9(imgA, x, y):
    sum = 0
    x *= 9
    y *= 9
    for i in range(x, x + 9):
        for j in range(y, y + 9):
            sum += imgA[i, j]
    return sum / 81


def getListSum(img, vert):
    listSum = []
    for i in range(2, 7):
        sum = 0;
        for j in range(2, 7):
            if (avg9x9(img, j, i) if vert else avg9x9(img, i, j)) < 130:
                sum += 1
        listSum.append(sum)
    return listSum


def getDirection(img):
    colSum = getListSum(img, True)

    rowSum = getListSum(img, False)

    key = [colSum, rowSum]

    if key[0][:3] == [1, 1, 2]:
        return 1

    if key[0][2:] == [2, 1, 1]:
        return 3

    if key[1][2:] == [2, 1, 1]:
        return 2

    if key[1][:3] == [1, 1, 2]:
        return 0

    pass


def rotate90(img):
    for i in range(81):
        for j in range(i):
            temp = img[i][j]
            img[i][j] = img[j][i]
            img[j][i] = temp

    for i in range(81):
        for j in range(40):
            temp = img[i][j]
            img[i][j] = img[i][80 - j]
            img[i][80 - j] = temp
    return img


def getSum(img, rots):
    imgNew = img.copy()
    for i in range(rots):
        imgNew = rotate90(imgNew)

    sum = 0
    for i in range(5, 7):
        for j in range(2, 7):
            if avg9x9(imgNew, i, j) < 130:
                sum += 2 ** ((j - 2) + (i - 5) * 5)

    return sum


def decode(img):
    codes = {
        0: 0,
        275: 1,
        594: 2,
        833: 3,
        561: 4,
        802: 5,
        99: 6,
        368: 7,
        523: 8
    }

    rotate = getDirection(img)

    if rotate:
        sum = getSum(img, rotate)
    else:
        return -1

    return codes[sum]


def getContours(img):
    squares = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                squares.append(approx)
            objCor = len(approx)
    return squares


def reorder(points):
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    pointsNew[3] = points[np.argmin(add)]
    pointsNew[0] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[2] = points[np.argmin(diff)]
    pointsNew[1] = points[np.argmax(diff)]
    return pointsNew


def identify(img, points):
    points = reorder(points)
    size = 81
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (size, size))
    thresh, imgBAndW = cv2.threshold(imgOutput, 175, 255, cv2.THRESH_BINARY)
    id = decode(imgBAndW)
    if id != -1:
        return [points, id]
    pass


def findCodes(img):
    imgCnt = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(img, 200, 200)
    #    kernal = np.ones((5,5))
    #    imgDial = cv2.dilate(imgCanny,kernal,iterations = 2)
    #    imgThresh = cv2.erode(imgDial,kernal,iterations = 1)

    squares = getContours(imgCanny)
    codes = []
    for square in squares:
        id = identify(imgGray, square)
        if id:
            codes.append(id)
    return codes


def labelDemo(codes):
    finalImg = img.copy()
    for code in codes:
        cv2.drawContours(finalImg, code[0], -1, (0, 255, 0), 20)
        cv2.putText(finalImg, str(code[1]), tuple(code[0][2][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return finalImg


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



# stack1 = funcs.stackImages(2, findCodes(img))
# imgCodes = findCodes(img)
# print(img.shape)
# print(imgCodes)
# imgF = labelDemo(imgCodes)
# calcDist(img, imgCodes, 7, 72, 36)
# stack = funcs.stackImages(0.4, [img, imgF])
#cv2.imshow("stack", stack)
# cv2.imshow("tiny", tiny)
cv2.waitKey(0)