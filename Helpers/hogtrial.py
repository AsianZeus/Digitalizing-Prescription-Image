# importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


pathx = 'X/XX/XXXX/XXX/'
filename = os.listdir(pathx)
for name in filename:
    newpath= pathx+'/'+name
    #reading the image
    img = imread(newpath)
    #resize image
    resized_img = resize(img, (128,64))

    #generating HOG features
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), visualize=True, multichannel=True)

    print('\n\nShape of Image Features\n\n')
    print(fd.shape)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 

    ax1.imshow(img, cmap=plt.cm.gray) 
    ax1.set_title('Input image') 

    # Rescale histogram for better display 
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

    ax2.imshow(hog_image, cmap=plt.cm.gray) 
    ax2.set_title('Histogram of Oriented Gradients')
    cv.imwrite(f"C:/Users/akroc/Desktop/{name}",hog_image)
    plt.show()
    