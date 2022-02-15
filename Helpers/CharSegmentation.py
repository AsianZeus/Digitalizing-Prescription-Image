import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import math
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, ResidualWrapper, DropoutWrapper, MultiRNNCell

sys.setrecursionlimit(10**6) 


"""
Provide functions and classes:
Model       = Class for loading and using trained models from tensorflow
create_cell = function for creatting RNN cells with wrappers
"""
SMALL_HEIGHT = 800

class Model():
    """Loading and running isolated tf graph."""
    def __init__(self, loc, operation='activation', input_name='x'):
        """
        loc: location of file containing saved model
        operation: name of operation for running the model
        input_name: name of input placeholder
        """
        self.input = input_name + ":0"
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.op = self.graph.get_operation_by_name(operation).outputs[0]

    def run(self, data):
        """Run the specified operation on given data."""
        return self.sess.run(self.op, feed_dict={self.input: data})
    
    def eval_feed(self, feed):
        """Run the specified operation with given feed."""
        return self.sess.run(self.op, feed_dict=feed)
    
    def run_op(self, op, feed, output=True):
        """Run given operation with the feed."""
        if output:
            return self.sess.run(
                self.graph.get_operation_by_name(op).outputs[0],
                feed_dict=feed)
        else:
            self.sess.run(
                self.graph.get_operation_by_name(op),
                feed_dict=feed)
        
    
    
def _create_single_cell(cell_fn, num_units, is_residual=False, is_dropout=False, keep_prob=None):
    """Create single RNN cell based on cell_fn."""
    cell = cell_fn(num_units)
    if is_dropout:
        cell = DropoutWrapper(cell, input_keep_prob=keep_prob)
    if is_residual:
        cell = ResidualWrapper(cell)
    return cell


def create_cell(num_units, num_layers, num_residual_layers, is_dropout=False, keep_prob=None, cell_fn=LSTMCell):
    """Create corresponding number of RNN cells with given wrappers."""
    cell_list = []
    
    for i in range(num_layers):
        cell_list.append(_create_single_cell(
            cell_fn=cell_fn,
            num_units=num_units,
            is_residual=(i >= num_layers - num_residual_layers),
            is_dropout=is_dropout,
            keep_prob=keep_prob
        ))

    if num_layers == 1:
        return cell_list[0]
    return MultiRNNCell(cell_list)


def resize(img, height=SMALL_HEIGHT, allways=False):
    """Resize image to given height."""
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    
    return img


def _crop_add_border(img, height, threshold=50, border=True, border_size=15):
    """Crop and add border to word image of letter segmentation."""
    # Clear small values
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

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

    if height != 0:
        img = resize(img[y0:y1, x0:x1], height, True)
    else:
        img = img[y0:y1, x0:x1]

    if border:
        return cv2.copyMakeBorder(img, 0, 0, border_size, border_size,
                                  cv2.BORDER_CONSTANT,
                                  value=[0, 0, 0])
    return img


def _word_tilt(img, height, border=True, border_size=15):
    """Detect the angle and tilt the image."""
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 30)

    if lines is not None:
        meanAngle = 0
        # Set min number of valid lines (try higher)
        numLines = np.sum(1 for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6)
        if numLines > 1:
            meanAngle = np.mean([l[0][1] for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6])

        # Look for angle with correct value
        if meanAngle != 0 and (meanAngle < 0.7 or meanAngle > 2.6):
            img = _tilt_by_angle(img, meanAngle, height)
    return _crop_add_border(img, height, 50, border, border_size)


def _tilt_by_angle(img, angle, height):
    """Tilt the image by given angle."""
    dist = np.tan(angle) * height
    width = len(img[0])
    sPoints = np.float32([[0,0], [0,height], [width,height], [width,0]])

    # Dist is positive for angle < 0.7; negative for angle > 2.6
    # Image must be shifed to right
    if dist > 0:
        tPoints = np.float32([[0,0],
                              [dist,height],
                              [width+dist,height],
                              [width,0]])
    else:
        tPoints = np.float32([[-dist,0],
                              [0,height],
                              [width,height],
                              [width-dist,0]])

    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    return cv2.warpPerspective(img, M, (int(width+abs(dist)), height))


def _sobel_detect(channel):
    """The Sobel Operator."""
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


class HysterThresh:
    def __init__(self, img):
        img = 255 - img
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        hist, bins = np.histogram(img.ravel(), 256, [0,256])

        self.high = np.argmax(hist) + 65
        self.low = np.argmax(hist) + 45
        self.diff = 255 - self.high

        self.img = img
        self.im = np.zeros(img.shape, dtype=img.dtype)

    def get_image(self):
        self._hyster()
        return np.uint8(self.im)

    def _hyster_rec(self, r, c):
        h, w = self.img.shape
        for ri in range(r-1, r+2):
            for ci in range(c-1, c+2):
                if (h > ri >= 0
                    and w > ci >= 0
                    and self.im[ri, ci] == 0
                    and self.high > self.img[ri, ci] >= self.low):
                    self.im[ri, ci] = self.img[ri, ci] + self.diff
                    self._hyster_rec(ri, ci)

    def _hyster(self):
        r, c = self.img.shape
        for ri in range(r):
            for ci in range(c):
                if (self.img[ri, ci] >= self.high):
                    self.im[ri, ci] = 255
                    self.img[ri, ci] = 255
                    self._hyster_rec(ri, ci)


def _hyst_word_norm(image):
    """Word normalization using hystheresis thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     img = cv2.bilateralFilter(gray, 0, 10, 30)
    img = cv2.bilateralFilter(gray, 10, 10, 30)
    return HysterThresh(img).get_image()


def word_normalization(image, height, border=True, tilt=True, border_size=15, hyst_norm=False):
    """ Preprocess a word - resize, binarize, tilt world."""
    image = resize(image, height, True)

    if hyst_norm:
        th = _hyst_word_norm(image)
    else:
        img = cv2.bilateralFilter(image, 10, 30, 30)
        gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        ret,th = cv2.threshold(norm, 50, 255, cv2.THRESH_TOZERO)

    if tilt:
        return _word_tilt(th, height, border, border_size)
    return _crop_add_border(th, height, border, border_size)


class Cycler:
    """ Cycle through boxes, separate words """
    width = 60
    height = 60
    step = 2
    
    def __init__(self, image, idx,char_segmentation,stri):
        # self.boxes = boxes       # Array of bounding boxes
        self.image = image       # Whole image
        self.index = idx         # Index of current bounding box
        self.actual = image      # Current image of word, drawing lines
        self.actualG = image     # Current slider image, evaluating
        self.char_segmentation = char_segmentation
        self.counter=1
        self.str=stri
        self.nextImg()
        
        
    
    def nextPos(self):
        """ Sliding over image and classifying regions """      
        
        length = (self.actual.shape[1] - self.width) // 2 + 1
        # print(f"length: {length}")
        try:
            input_seq = np.zeros((1, length, self.height*self.width), dtype=np.float32)
        except:
            return -1
        try:
            input_seq[0][:] = [self.actualG[:, loc * self.step:loc * self.step + self.width].flatten()
                           for loc in range(length)]
        except:
            return 0
        
        
        pred = self.char_segmentation.eval_feed({
            'inputs:0': input_seq,
            'length:0': [length],
            'keep_prob:0': 1})[0]
        

        # Finalize the gap positions from raw prediction
        gaps = []
        lastGap = 0
        gapCount = 1
        gapPositionSum = self.width / 2
        firstGap = True
        gapBlockFirst = 0
        gapBlockLast = self.width / 2
        
        gapsImage = self.actual.copy()
        CharImage = self.actual.copy()

        for i, p in enumerate(pred):
            if p == 1:
                gapPositionSum += i * self.step + self.width / 2
                gapBlockLast = i * self.step + self.width / 2
                gapCount += 1
                lastGap = 0
                if gapBlockFirst == 0:
                    gapBlockFirst = i * self.step + self.width / 2
            else:
                if gapCount != 0 and lastGap >= 1:
                    if firstGap:
                        gaps.append(int(gapBlockLast))
                        firstGap = False
                    else:
                        gaps.append(int(gapPositionSum // gapCount))
                    gapPositionSum = 0
                    gapCount = 0
                gapBlockFirst = 0
                lastGap += 1
            # Plotting all lines
            cv2.line(self.actual,
                     ((int)(i * self.step + self.width / 2),0),
                     ((int)(i * self.step + self.width / 2),self.height),
                     (int(255-(p*255)), int(p*255), 0), 1)

        # Adding final gap position
        if gapBlockFirst != 0:
            gaps.append(int(gapBlockFirst))
        else:
            gapPositionSum += (len(pred) - 1) * 2 + self.width/2
            gaps.append(int(gapPositionSum / (gapCount + 1)))
        
        for gap in gaps:
            cv2.line(gapsImage,
                     ((int)(gap),0),
                     ((int)(gap),
                     self.height), (0,255,0,0.5), 1)     
        
        cv2.namedWindow("GapsImage",0)
        cv2.imshow("GapsImage",gapsImage)
        cv2.waitKey(0)
        
        x=0
        # print(f"Gaps: {gaps}")          #Gaps: [30, 58, 90, 120]        0;30,   30:58
        for i in range(len(gaps)+1):
            if(i==len(gaps)):
                gapx=CharImage.shape[1]
                # print(f"x: {x} gap: {gaps[i]}")
                im=CharImage[0:CharImage.shape[0], x:gapx]
                # print(im.shape)

                
            else:
                gapx=gaps[i]
                # print(f"x: {x} gap: {gaps[i]}")
                im=CharImage[0:CharImage.shape[0], x:gapx]
                # print(im.shape)
                x=gaps[i]
            # cv2.imwrite(f"C:/Users/akroc/Desktop/Dataset/Characters/Character{self.str}{self.counter}.png",im)
            self.counter+=1
            return 1
            # cv2.namedWindow("CharImage",0)
            # cv2.imshow("CharImage",im)
            # cv2.waitKey(0)
        

    def nextImg(self):
        """ Getting next image from the array """

        img = self.image
        img = resize(img, self.height, True)                      
        img = word_normalization(
            img,
            self.height,
            border=False,
            border_size=int(self.width/2),
            hyst_norm=True)
        # implt(img,t='wordnormalize')
        
        # Remove horizontal lines
        # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
        # remove_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        # cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # for c in cnts:
        #     cv2.drawContours(img, [c], -1, (0,0,0), 4)

        self.actual = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.actualG = img         
        
        self.nextPos()
        
        # Printing index for recovery
        # print("Index: " + str(self.index))
        self.index += 1
        return 0
