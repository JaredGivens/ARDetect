import cv2
import numpy as np
import funcs

# returns average grayscale value of 9x9 array of pixels
def avg9x9(img, x, y):
    sum = 0
    x *= 9
    y *= 9
    for i in range(x, x + 9):
        for j in range(y, y + 9):
            sum += img[i, j]
    return sum / 81

# returns a 2d list of booleans based on center of tag
def getMat(img):
    mat = []
    for i in range(2,7):
        row = []
        for j in range(2,7):
            row.append(avg9x9(img,j,i) < 130)
        mat.append(row)
    return mat

# returns the binary sum of a 2d list of booleans
def binarySum(lines):
    sum = 0
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j]:
                sum += 2 ** (j + i * 5)
    return sum

# returns whether the top three rows of a 2d list of booleans corresponds to a tag
# and the binary sum of the bottom two rows
def isARTag(mat):

    for i in range(4):
        if binarySum([mat[0]]) == 4 and binarySum([mat[1]]) == 4 and binarySum([mat[2]]) == 10:

            return True, binarySum(mat[3:])

        mat = rotate90(mat)

    return False, 0

# rotates 2d list clockwise
def rotate90(list):
    for i in range(len(list)):
        for j in range(i):
            temp = list[i][j]
            list[i][j] = list[j][i]
            list[j][i] = temp

    for i in range(len(list)):
        for j in range(int(len(list)/2)):
            temp = list[i][j]
            list[i][j] = list[i][len(list) - 1 - j]
            list[i][len(list) - 1 - j] = temp
    return list

# returns value of tag according to codes dict
def decode(img):
    tags = {
        0: 0,
        89: 1,
        297: 2,
        368: 3,
        561: 4,
        616: 5,
        792: 6,
        833: 7,
        58: 8
    }

    mat = getMat(img)
    isTag, id = isARTag(mat)

    if isTag:
        return tags[id]
    return -1

# returns a list of quadrilateral's corners
def getQuads(img):
    quads = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                quads.append(approx)
            objCor = len(approx)
    return quads

# orders points of 4 sided polygon bottom right, bottom left, top right, top left
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

# returns 81px by 81px sections of an image
def perspective(img, points):
    points = reorder(points)
    size = 81
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (size, size))
    return imgOutput


    id = decode(imgBAndW)
    if id != -1:
        return [points, id]
    pass

# returns list of tag corners followed by their ids
def findTags(img):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, imgBAW = cv2.threshold(imgGray, 175, 255, cv2.THRESH_BINARY)
    imgCanny = cv2.Canny(img, 200, 200)

    quads = getQuads(imgCanny)
    tags = []
    for quad in quads:
        tagCanadate = perspective(imgBAW, quad)
        id = decode(tagCanadate)
        if not(id == -1):
            tags.append([quad, id])
    return tags

# returns image with tag corners and ids
def labelDemo(tags):
    finalImg = img.copy()
    for tag in tags:
        cv2.drawContours(finalImg, tag[0], -1, (0, 255, 0), 20)
        cv2.putText(finalImg, str(tag[1]), tuple(tag[0][2][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return finalImg





img = cv2.imread("/Users/jared/Pictures/arcodes.jpg")
imgtags = findTags(img)
print(imgtags)
imgF = labelDemo(imgtags)
stack = funcs.stackImages(0.4, [img, imgF])
cv2.imshow("stack", stack)
cv2.waitKey(0)
