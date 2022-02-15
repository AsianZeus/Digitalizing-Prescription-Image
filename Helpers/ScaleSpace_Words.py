import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):

	"""Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

	Args:
		img: grayscale uint8 image of the text-line to be segmented.
		kernelSize: size of filter kernel, must be an odd integer.
		sigma: standard deviation of Gaussian function used for filter kernel.
		theta: approximated width/height ratio of words, filter function is distorted by this factor.
		minArea: ignore word candidates smaller than specified area.

	Returns:
		List of tuples. Each tuple contains the bounding box and the image of the segmented word.
	"""

	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv.threshold(imgFiltered, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	imgThres = 255 - imgThres

	kernela = cv.getStructuringElement(cv.MORPH_CROSS,(3,3)) 

	dilated = cv.dilate(imgThres,kernela,iterations = 2) 
	(components, _) = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	# append components to result
	res = []
	for c in components:
		# skip small word candidates
		if cv.contourArea(c) < minArea:
			continue
		# append bounding box and image of word to result list
		currBox = cv.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	# return list of words, sorted by x-coordinate
	return sorted(res, key=lambda entry:entry[0][0])


def createKernel(kernelSize, sigma, theta):

	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2

	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta

	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize

			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

pathx = 'C:/Users/akroc/Desktop/LineRemoved'
filename = os.listdir(pathx)
for name in filename:
	newpathx = pathx+'/'+name
	lcropped = cv.imread(newpathx)
	# print(lcropped.shape)
	gray = cv.cvtColor(lcropped, cv.COLOR_BGR2GRAY)
	res = wordSegmentation(gray, kernelSize=5, sigma=0.2, theta=0.1, minArea=700)
	# res = wordSegmentation(gray, kernelSize=25, sigma=11, theta=7, minArea=100)
	print('Segmented into %d words'%len(res))
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		currImg = lcropped[y:y+h, x:x+w]
		cv.rectangle(lcropped,(x,y),(x+w,y+h),(255,0,0),3)
	cv.namedWindow('ddc',0)
	cv.imshow("ddc",lcropped)
	cv.waitKey(0)
	cv.destroyAllWindows()

print("ok done!")