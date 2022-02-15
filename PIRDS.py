import math
import os

import cv2 as cv
import numpy as np

from Modelx import Model
from ProcessingAndSegmentation import *

# import CharSegmentation
# *****************************************************************************

if __name__ == "__main__":
    #     try:
    #         cseg=CharSegmentation.Model('/RNN/Bi-RNN-new', 'prediction')
    #         print("Successfully loaded.")
    #     except:
    #         print("Couldn't Load the model!")

    try:
        model = Model(open('modelx/charList.txt').read(),
                      1, mustRestore=True, dump=False)
    except:
        print("Couldn't Load the model!")

    path = os.getcwd()+'/Dataset'
    coord = []
    filename = os.listdir(path+'/Images')
    cropimgdir = os.listdir(path+'/CroppedImage')
    imgcounter = 1
    for name in filename:
        wordfile = open(f"Result/Result_{name[:-4]}.txt", 'a')
        if(f"{name[:-4]}.png" not in cropimgdir):
            file = path+'/Images/'+name
            image = cv.imread(file)
            # cv.namedWindow('Orignal Image', 0)
            # cv.imshow("Orignal Image", image)
            # cv.waitKey(0)

    # *************************** Page Segmentation ***************************
            PageSegmentedImage = PageSegment(image)

    # ************ FourPointPerspective (Region of Interest) *******************
            FourPointPerspectiveImage = PageCorrection(PageSegmentedImage)
            # cv.namedWindow("FourPointPerspective",0)
            # cv.imshow("FourPointPerspective",FourPointPerspectiveImage)
            # cv.waitKey(0)
            cv.imwrite(f'{path}/CroppedImage/{name[:-4]}.png', FourPointPerspectiveImage)

# ************ Reading Cropped Image *******************
        else:
            file = path+'/CroppedImage/'+name[:-4]+'.png'
            FourPointPerspectiveImage = cv.imread(file)
            # cv.namedWindow("FourPointPerspective",0)
            # cv.imshow("FourPointPerspective",FourPointPerspectiveImage)
            # cv.waitKey(0)

# #************ Page Processing *******************
        # BinaryImage = PageProcessing(FourPointPerspectiveImage)
        # cv.namedWindow('Binary Image', 0)
        # cv.imshow("Binary Image", BinaryImage)
        # cv.waitKey(0)
        Height = FourPointPerspectiveImage.shape[0]
        Width = FourPointPerspectiveImage.shape[1]

# #************ Get Horizontal Lines *******************
        HorizontalLines = getHorizintalLines(FourPointPerspectiveImage)
        # cv.namedWindow("HorizontalLines",0)
        # cv.imshow("HorizontalLines",HorizontalLines)
        # cv.waitKey(0)

# #************ Line Creation *******************
        LineCoord1, LineCoord2, No_of_Lines = LineCreation(HorizontalLines)
        print(LineCoord1, LineCoord2, No_of_Lines, sep="\n")
# #************ Cropping Lines *******************
        CroppedLine = CropLines(FourPointPerspectiveImage,
                                HorizontalLines, LineCoord1, LineCoord2, Width)

# #************ Process Lines *******************
        counterline = 1
        for i in range(len(CroppedLine)):
            CroppedColourLine = CroppedLine[i][0]
            CroppedBinaryLine = CroppedLine[i][1]
            ProcessedLines = PreprocessLines(
                CroppedColourLine, CroppedBinaryLine)
            # cv.namedWindow("ProcessedLines",0)
            # cv.imshow("ProcessedLines",ProcessedLines)
            # cv.waitKey(0)
            # cv.imwrite(f"{path}/ProcessedLines/{name[:-4]}_{counter}.png",ProcessedLines)
            counterline += 1
        # print(f"**{No_of_Lines} Lines Added to the folder!**")


# **************************************Word***********************************************************

# ************ Word Pre-Processing *******************
            # wordfile.write("\n")
            SegmentedLine = ProcessedLines
            # cv.namedWindow('fd', 0)
            # cv.imshow("fd",SegmentedLine)
            # cv.waitKey(0)
            ProcessedLine = WordPreProcessing(SegmentedLine)

# #************ Line Removal *******************
            LinesRemoved = LineRemoval(ProcessedLine)
            # cv.namedWindow('fd', 0)
            # cv.imshow("fd",LinesRemoved)
            # cv.waitKey(0)
# #************ Process Lines *******************
            WordsArray = FindingContours(LinesRemoved, SegmentedLine)
            counterword = 1
            for i in WordsArray:
                NormalizedWord = word_normalization(
                    i, 100, border=False, tilt=False, hyst_norm=False)
                if(ClassifyImage(NormalizedWord)):
                    WhiteLinesWord = line_rem(NormalizedWord)
                    GrayscaleWord = EnhanceContrast(WhiteLinesWord)
                    # cv.namedWindow('final Segemented word', 0)
                    # cv.imshow("final Segemented word",GrayscaleWord)
                    # cv.waitKey(0)
                    # cv.imwrite(f"{name[:-4]}_{counterline}_{counterword}.png",GrayscaleWord)
                    counterword += 1

# ***************************************Recognition*********************************************************

# #************ Word Recognition *******************

                    recognized, probability = infer(model, GrayscaleWord)
                    # wordfile.write(f"{recognized} ")
                    print(recognized)
        wordfile.close()

# ***************************************Character*********************************************************|

# #************ Character Segmentation *******************
        #     CharSegmentation.Cycler(i,1,cseg,f"{name[4:-4]}_{counter}")
