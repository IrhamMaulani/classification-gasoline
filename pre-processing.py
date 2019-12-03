import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.feature import greycomatrix, greycoprops
from pandas import DataFrame
path = 'image\IMG_0164.JPG'

image = cv2.imread(path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]

startRow, startCol = int(h*.25), int(w*.25)
endRow, endCol = int(h*.75), int(w*.75)

cropImage = image[startRow:endRow, startCol:endCol]

brightnessImage = cv2.addWeighted(cropImage, 2, np.zeros(cropImage.shape, cropImage.dtype), -1, 0)

negative = 255 - brightnessImage

grayImage  = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)

grayImage = grayImage.astype(np.uint8)

greycomatrix = greycomatrix(grayImage, [0, 1], [0, np.pi/2], levels=256)

energy = greycoprops(greycomatrix, 'energy')
average = np.mean(energy)

Energy = {'Nama': [path],
'Average': [average],
}

df = DataFrame(Energy, columns= ['Nama', 'Average'])

print(df)
#plt.subplot(1, 1, 1), plt.axis('Off'), plt.title(df)
#plt.show()