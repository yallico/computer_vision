# -*- coding: utf-8 -*-
"""Tutorial_Two.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ov6umi-j15xz_vYT1GQL8LlSkCEaswUp

# Import the matplotlib package

* Can be used to creating static, animated, and interactive visualisations in Python
* plotting in a MATLAB style
"""

import matplotlib.pyplot as plt
import numpy as np

"""# Plotting"""

x = [-2, -1, 0, 1, 2]
y = [10, 100, 200, 500, 1000]
# plt.plot(x, y)
#plt.show()

"""* Add labels"""

# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
#plt.show()

"""* Change colour/marker and specify axes"""

# plt.plot(x, y, 'g--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis([0, 2, 0, 800])
#plt.show()

"""* Scatter plot"""

x = np.arange(10)
y = np.arange(10)
c = np.arange(10)
s = np.arange(1, 20, 2)

# plt.scatter(x=x, y=y, c=c, s=s)
#plt.show()

"""* Bar chart and subplots"""

# cat = ['cat0', 'cat1']
# num = [10, 20]
# plt.figure(figsize=(10, 6))
# plt.subplot(121)
# plt.bar(cat, num)
# plt.subplot(122)
# plt.bar(cat, num, width=0.5)
# plt.suptitle('example', fontsize=20)
#plt.show()

"""# Read in an image

* using matplotlib
"""

import matplotlib.image as mpimg
img = mpimg.imread('Tutorials/Week-2/sample_img.jpg')
# plt.imshow(img)
#plt.show()

"""* using OpenCV"""

import cv2
img = cv2.imread('Tutorials/Week-2/sample_img.jpg')
#cv2.imshow('image', img)  # this cannot be used in Colab
# cv2.waitKey(3000)
# cv2.destroyAllWindows()  

# from google.colab.patches import cv2_imshow
# cv2_imshow(img)

"""* Try to fix this problem yourself"""

image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb) # note that the colour is not displayed correctly, why?
# plt.show()
# OpenCV uses BGR instead of RGB!
# How to fix it?

"""* using the Python Imaging Library (PIL)"""

# from PIL import Image  
# from IPython.display import display # to display images
# pil_im = Image.open('Tutorials/Week-2/sample_img.jpg')
# display(pil_im)

"""* cropping"""

# img_cropped = image_rgb[0:100, 50:300, :]
# plt.imshow(img_cropped)

"""* RGB to Grey (recall lecture 2)"""

img_grey = image_rgb[:, :, 0]*0.2989 + image_rgb[:, :, 1]*0.5870 + image_rgb[:, :, 2]*0.1140
img_grey = img_grey.astype(np.uint8)
cv2.imshow('image', img_grey)  # this cannot be used in Colab
cv2.waitKey(0)
cv2.destroyAllWindows()  


plt.imshow(img_grey, cmap='gray')  # using the gray colour space
plt.show()


"""# Explore other methods for image cropping and colour-greyscale conversion enabled by different python packages yourself"""