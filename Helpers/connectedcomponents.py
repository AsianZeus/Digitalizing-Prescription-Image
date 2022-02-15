import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2



pathx = 'X/XX/XXXX/XXX'
filename = os.listdir(pathx)
for name in filename:
    newpath= pathx+'/'+name
    orignal =cv.imread(newpath)
    gray = cv.cvtColor(orignal, cv.COLOR_BGR2GRAY)
    ret3, thresh2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh2 = cv.erode(thresh2, np.ones(5))
    img2=1-(undesired_objects(thresh2)/255)
    img2=np.uint8(img2)
    # ret3, img2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # print(img2[0][0])
    
    (components, _) = cv.findContours(img2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in components:
        if cv.contourArea(c) < 4000:
            continue
        currBox = cv.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = orignal[y:y+h, x:x+w]
        cv.namedWindow('final Segemented word', 0)
        cv.imshow("final Segemented word",currImg)
        cv.imshow("img",thresh2)
        cv.waitKey(0)

# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv.minAreaRect(coords)[-1]
    if(angle < -45):
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated
