import cv2
import os
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    #step 1: convert to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #step 2: histogram equalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)
    
    #step 3: gaussian blur
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    #TODO: currently all hyper parameters are used as default, it required fine-tunning

    return blurred_image

