import math
import os

import cv2 as cv
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen

from Modelx import Model


# *****************************************************************************
def onMouse(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        param[0] = param[2].copy()
        cv.circle(param[0], (x, y), 30, (0, 0, 255), -1)
        param[1] = param[0]
        cv.imshow('Page Segmented Image', param[1])
    if event == cv.EVENT_LBUTTONDOWN:
        param[0] = param[2].copy()
        cv.circle(param[0], (x, y), 40, (0, 255, 0), -1)
        param[1] = param[0]
        cv.imshow('Page Segmented Image', param[1])
        coord.append((x, y))

# *****************************************************************************


def PageCorrection(image):
    globals()['coord'] = []
    height = image.shape[0]
    width = image.shape[1]
    imgx = image.copy()
    temp = image.copy()
    cv.namedWindow('Page Segmented Image', 0)
    param = [image, imgx, temp]
    cv.setMouseCallback('Page Segmented Image', onMouse, param)
    while 1:
        image = imgx
        cv.imshow("Page Segmented Image", image)
        k = cv.waitKey(1000) & 0XFF
        if k == 27:
            break
    cv.destroyAllWindows()
    pts1 = np.float32([coord[0], coord[1], coord[2], coord[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # print("Points:  ",pts1,pts2)
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    result = cv.warpPerspective(image, matrix, (width, height))
    return result

# *****************************************************************************


def countfreq(list):
    white = np.count_nonzero(list)
    return (white, len(list)-white)

# *****************************************************************************


def isNotWord(list, threshold):
    size = len(list)
    white, black = countfreq(list)
    ratio = (white/size, black/size)
    # print(white,black,ratio)
    if(ratio[0] > threshold):
        return 1
    else:
        return 0

# *****************************************************************************


def rectify(list, pixelThreshold):
    for i in range(pixelThreshold, len(list)-pixelThreshold, pixelThreshold):
        if(np.count_nonzero(list[i-pixelThreshold:i+pixelThreshold]) < pixelThreshold):
            list[i-pixelThreshold:i+pixelThreshold] = [0]*(pixelThreshold*2)
    return list

# *****************************************************************************


def WordPreProcessing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh2 = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 12)
    thresh2 = 255-thresh2
    thresh2 = cv.dilate(
        thresh2, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
    thresh2 = 255-thresh2
    return thresh2

# *****************************************************************************


def LineRemoval(thresh2):

    laplacian = cv.Laplacian(thresh2, -1, ksize=15)
    minLineLength = 100
    maxLineGap = 60
    lines = cv.HoughLinesP(laplacian, 5, np.pi/180, 550,
                           minLineLength, maxLineGap)
    if(lines is not None):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(thresh2, (x1, y1), (x2, y2), (255, 255, 255), 9)

    thresh2 = cv.medianBlur(thresh2, 3)
    thresh2 = 255-thresh2
    thresh2 = cv.dilate(thresh2, np.ones(5))
    thresh2 = 255-thresh2

    thresh2 = np.transpose(thresh2)

    height = thresh2.shape[0]
    width = thresh2.shape[1]

    wordornot = []

    for i in range(height):
        wordornot.append(isNotWord(thresh2[i], threshold=0.947))

    wordornot = rectify(wordornot, pixelThreshold=30)

    for i in range(height):
        if(wordornot[i] == True):
            thresh2[i] = [0]*width

    thresh2 = np.transpose(thresh2)

    return thresh2

# *****************************************************************************


def FindingContours(ProcessedLine, SegmentedLine):
    (components, _) = cv.findContours(
        ProcessedLine, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    wordslist = []
    for c in components:
        if cv.contourArea(c) < 5000:
            continue
        currBox = cv.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = SegmentedLine[y:y+h, x:x+w]
        if(currImg.shape[1] > 60):
            wordslist.append(currImg)
    wordslist.reverse()
    return wordslist

# *****************************************************************************


def getHorizintalLines(src):
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv.getStructuringElement(
        cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    return horizontal

# *****************************************************************************


def LineCreation(Horizontallineimage):
    image = Horizontallineimage
    # Preporcessing
    thresh2 = cv.GaussianBlur(image, (7, 7), 0)
    thresh2 = 255-thresh2
    thresh2 = cv.medianBlur(thresh2, 3)
    thresh2 = 255-thresh2
    # Find Thin Line Edges
    derivY = cv.Canny(thresh2, 100, 190)
    height = image.shape[0]
    width = image.shape[1]

    # Column Wise
    col1 = derivY[:, int(width/5)]
    col2 = derivY[:, int(width-width/7)]

    # Find non zero values(black) in column
    nonzerocol1 = np.nonzero(col1)[0]
    nonzerocol2 = np.nonzero(col2)[0]

    # Thresholding
    Linecoordcol1 = []
    for i in range(len(nonzerocol1)-1):
        if(nonzerocol1[i+1]-nonzerocol1[i] > 50):
            Linecoordcol1.append(nonzerocol1[i])
    Linecoordcol1.append(nonzerocol1[-1])

    Linecoordcol2 = []
    for i in range(len(nonzerocol2)-1):
        if(nonzerocol2[i+1]-nonzerocol2[i] > 50):
            Linecoordcol2.append(nonzerocol2[i])
    Linecoordcol2.append(nonzerocol2[-1])

    if(Linecoordcol1[0] not in range(50)):
        Linecoordcol1.insert(0, 0)
    if(Linecoordcol1[-1] not in range(height-20, height)):
        Linecoordcol1.append(height-1)

    if(Linecoordcol2[0] not in range(50)):
        Linecoordcol2.insert(0, 0)
    if(Linecoordcol2[-1] not in range(height-20, height)):
        Linecoordcol2.append(height-1)

    # Calculating avg Line Distance
    LineDistance = abs(Linecoordcol1[1]-Linecoordcol1[2])

    No_of_Lines = len(Linecoordcol1)-1

    # Adjusting
    for i in range(len(Linecoordcol1)):
        if(len(Linecoordcol1) == len(Linecoordcol2)):
            if(abs(Linecoordcol1[i]-Linecoordcol2[i]) > int(LineDistance/1.6)):
                # print(f"== Replaced @{i}")
                Linecoordcol2[i] = Linecoordcol1[i]
        elif(len(Linecoordcol1) > len(Linecoordcol2)):  # positive skew angle
            if(abs(Linecoordcol1[i]-Linecoordcol2[i]) > int(LineDistance/1.6)):
                # print(f"> Replaced @{i}")
                Linecoordcol2.insert(i, Linecoordcol1[i])
    else:  # negative skew angle
        if(abs(Linecoordcol1[i]-Linecoordcol2[i]) > int(LineDistance/1.6)):
            # print(f"< Deleted @{i}")
            Linecoordcol2.pop(i)

    return (Linecoordcol1, Linecoordcol2, No_of_Lines)

# *****************************************************************************


def CropLines(colorimage, image, Linecoordcol1, Linecoordcol2, width):
    croparrbw = []
    for y in range(len(Linecoordcol1)):
        if(Linecoordcol1[y] < 10):
            continue
        if(Linecoordcol1[y-1] < 10):
            clrtempimage = colorimage[0:Linecoordcol1[y] +
                                      abs(Linecoordcol1[y]-Linecoordcol2[y])+10, 0:width]
            bwtempimage = image[0:Linecoordcol1[y] +
                                abs(Linecoordcol1[y]-Linecoordcol2[y])+10, 0:width]
            if(clrtempimage.size <= 0):
                continue
            croparrbw.append((clrtempimage, bwtempimage))
        else:
            clrtempimage = colorimage[Linecoordcol1[y-1]-abs(
                Linecoordcol1[y-1]-Linecoordcol2[y-1])-10:Linecoordcol1[y]+abs(Linecoordcol1[y]-Linecoordcol2[y])+10, 0:width]
            bwtempimage = image[Linecoordcol1[y-1]-abs(Linecoordcol1[y-1]-Linecoordcol2[y-1])-10:Linecoordcol1[y]+abs(
                Linecoordcol1[y]-Linecoordcol2[y])+10, 0:width]
            if(clrtempimage.size <= 0):
                continue
            croparrbw.append((clrtempimage, bwtempimage))
    return croparrbw

# *****************************************************************************


def PreprocessLines(colorimg, bwimg):
    orignalimg = colorimg.copy()
    kernel = np.ones((1, 500), np.uint8)
    bwimg = cv.dilate(bwimg, kernel)

    height = colorimg.shape[0]
    width = colorimg.shape[1]
    white = 255
    for w in range(width):
        for h in range(height):
            if(bwimg[h][w] != 255):
                colorimg[h][w] = [255, 255, 255]
            else:
                break

    colorimg = cv.flip(colorimg, flipCode=0)
    bwimg = cv.flip(bwimg, flipCode=0)

    for w in range(width):
        for h in range(height):
            if(bwimg[h][w] != 255):
                colorimg[h][w] = [255, 255, 255]
            else:
                break

    colorimg = cv.flip(colorimg, flipCode=0)
    return colorimg

# *****************************************************************************


def _crop_add_border(img, orignalimg, height, threshold=50, border=False, border_size=15):
    """Crop and add border to word image of letter segmentation."""
    # Clear small values
    ret, img = cv.threshold(img, threshold, 255, cv.THRESH_TOZERO)
    x0 = 0
    y0 = 0
    x1 = img.shape[1]
    y1 = img.shape[0]
    for i in range(img.shape[0]):
        if np.count_nonzero(img[i, :]) > 1:
            y0 = i
            break
    for i in reversed(range(img.shape[0])):
        if np.count_nonzero(img[i, :]) > 1:
            y1 = i+1
            break
    for i in range(img.shape[1]):
        if np.count_nonzero(img[:, i]) > 1:
            x0 = i
            break
    for i in reversed(range(img.shape[1])):
        if np.count_nonzero(img[:, i]) > 1:
            x1 = i+1
            break
    img = orignalimg[y0:y1, x0:x1]
    return img

# *****************************************************************************


def word_normalization(image, height, border=False, tilt=False, border_size=15, hyst_norm=False):
    """ Preprocess a word - resize, binarize, tilt world."""
    # image = resize(image, height, True)
    orignal = image.copy()
    img = cv.bilateralFilter(image, 10, 30, 30)
    gray = 255 - cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    norm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    ret, th = cv.threshold(norm, 50, 255, cv.THRESH_TOZERO)
    return _crop_add_border(th, orignal, height, border, border_size)

# *****************************************************************************


def EnhanceContrast(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    yen_threshold = threshold_yen(img)
    bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
    # increase line width
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv.erode(bright, kernel, iterations=1)
    return imgMorph

# *****************************************************************************


def ClassifyImage(orignal):
    gray = cv.cvtColor(orignal, cv.COLOR_BGR2GRAY)
    ret3, thresh2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    flatpix = thresh2.flatten()
    size = thresh2.shape[0]*thresh2.shape[1]
    white = np.count_nonzero(flatpix)
    black = size-white
    ratio = white/size
    if(ratio < 0.89 and orignal.shape[1] < 1200):
        return True
    else:
        False

# *****************************************************************************


def resize(img, height=800, allways=False):
    """Resize image to given height."""
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv.resize(img, (int(rat * img.shape[1]), height))

    return img

# *****************************************************************************

def ratio(img, height=800):
    """Getting scale ratio."""
    return img.shape[0] / height

# *****************************************************************************

def img_extend(img, shape):
    # Extend 2D image (numpy array) in vertical and horizontal direction.
    x = np.zeros(shape, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    return x

# *****************************************************************************

def edges_det(img, min_val, max_val):
    """ Preprocessing (gray, thresh, filter, border) + Canny edge detection """
    img = cv.cvtColor(resize(img), cv.COLOR_BGR2GRAY)
    # Applying blur and threshold
    img = cv.bilateralFilter(img, 9, 75, 75)
    img = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 4)
    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    img = cv.medianBlur(img, 11)
    # Add black border - detection of border touching pages
    # Contour can't touch side of image
    img = cv.copyMakeBorder(
        img, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return cv.Canny(img, min_val, max_val)

# *****************************************************************************


def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])

# *****************************************************************************


def contour_offset(cnt, offset):
    """ Offset contour because of 5px border """
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt

# *****************************************************************************


def find_page_contours(edges, img):
    """ Finding corner points of page contour """
    # Getting contours
    contours, hierarchy = cv.findContours(
        edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)
    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                             [0, height-5],
                             [width-5, height-5],
                             [width-5, 0]])
    for cnt in contours:
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.03 * perimeter, True)
        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv.isContourConvex(approx) and
                max_area < cv.contourArea(approx) < MAX_COUNTOUR_AREA):
            max_area = cv.contourArea(approx)
            page_contour = approx[:, 0]
    # Sort corners and offset them
    page_contour = four_corners_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))

# *****************************************************************************


def persp_transform(img, s_points):
    """ Transform perspective from start points to target points """
    # Euclidean distance - calculate maximum height and width
    height = max(np.linalg.norm(s_points[0] - s_points[1]),
                 np.linalg.norm(s_points[2] - s_points[3]))
    width = max(np.linalg.norm(s_points[1] - s_points[2]),
                np.linalg.norm(s_points[3] - s_points[0]))
    # Create target points
    t_points = np.array([[0, 0],
                         [0, height],
                         [width, height],
                         [width, 0]], np.float32)
    # getPerspectiveTransform() needs float32
    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)
    M = cv.getPerspectiveTransform(s_points, t_points)
    return cv.warpPerspective(img, M, (int(width), int(height)))

# *****************************************************************************


def PageSegment(image):
    edges_image = edges_det(image, 200, 250)
    # Close gaps between edges (double page clouse => rectangle kernel)
    edges_image = cv.morphologyEx(
        edges_image, cv.MORPH_CLOSE, np.ones((5, 11)))
    # cv.namedWindow("EdgeImage",0)
    # cv.imshow("EdgeImage",edges_image)
    # cv.waitKey(0)
    page_contour = find_page_contours(edges_image, resize(image))
    # print(f"Page Contour: {page_contour}")

    cv.namedWindow("Page Boundary", 0)
    cv.imshow("Page Boundary", cv.drawContours(resize(image), [page_contour], -1, (0, 255, 0), 3))
    cv.waitKey(0)
    cv.destroyAllWindows()
    # Recalculate to original scale
    page_contour = page_contour.dot(ratio(image))
    newImage = persp_transform(image, page_contour)
    return newImage


# *****************************************************************************
def infer(model, fnImg):
    # "recognize text in image provided by file path"
    img = preprocess(fnImg, Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    # print('Recognized:', '"' + recognized[0] + '"')
    # print('Probability:', probability[0])
    return (recognized[0], probability[0])

# *****************************************************************************


class Batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

# *****************************************************************************


def preprocess(img, imgSize, dataAugmentation=False):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        # random width, but at least 1
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
        # stretch horizontally by factor 0.5 .. 1.5
        img = cv.resize(img, (wStretched, img.shape[0]))

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    # scale according to f (result at least 1 and at most wt or ht)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv.transpose(target)

    # normalize
    (m, s) = cv.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img

# *****************************************************************************


def line_rem(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # Remove horizontal
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    detected_lines = cv.morphologyEx(
        thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    (components, _) = cv.findContours(
        detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in components:
        if cv.contourArea(c) < image.shape[1]*0.9:
            continue
        currBox = cv.boundingRect(c)
        (x, y, w, h) = currBox
        if(w < image.shape[1]*0.70):
            if(y > image.shape[0]*0.2 and y < image.shape[0]*0.80):
                continue
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), thickness=-1)

    # Remove vertical
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 25))
    detected_lines = cv.morphologyEx(
        thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    (components, _) = cv.findContours(
        detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in components:
        if cv.contourArea(c) < image.shape[1]*0.9:
            continue
        currBox = cv.boundingRect(c)
        (x, y, w, h) = currBox
        if(h < image.shape[0]*0.97):
            # if(x > image.shape[1]*0.05 and x < image.shape[1]*0.85):
            continue
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), thickness=-1)
    return image
