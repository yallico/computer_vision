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
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(12, 12))
    equalized_image = clahe.apply(gray_image)
    
    #step 3: gaussian blur
    blurred_image = cv2.GaussianBlur(equalized_image, (3, 3), 0.3)

    #TODO: currently all hyper parameters are used as default, it required fine-tunning

    return blurred_image

def process_and_save_images(image_paths, save_dir):
    for img_path in image_paths:
        processed_image = preprocess_image(img_path)
        if processed_image is not None:
            #save path
            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, processed_image)
        else:
            print(f"Skipping image: {img_path}")

def extract_and_filter_features(image_path, save_visualization=False, save_dir=None, min_size=5, max_size=50):
    #load
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image: {image_path}")
        return [], None

    #SIFT detector, need to use in grid-search
    sift = cv2.SIFT_create(
        nfeatures=0,             #number of best features to retain (0 means no limit)
        nOctaveLayers=3,         #number of layers in each octave
        contrastThreshold=0.04,  #threshold for filtering out weak features
        edgeThreshold=10         #threshold for edge detection
    )
    
    #detect
    keypoints, descriptors = sift.detectAndCompute(image, mask=None)
    
    #filter features by size
    filtered_keypoints = [kp for kp in keypoints if min_size <= kp.size <= max_size]
    
    #update descriptors
    if descriptors is not None:
        indices = [i for i, kp in enumerate(keypoints) if kp in filtered_keypoints]
        filtered_descriptors = descriptors[indices, :]
    else:
        filtered_descriptors = None
    
    #save sample image with keypoints
    if save_visualization and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        keypoint_image = cv2.drawKeypoints(
            image, filtered_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        img_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, keypoint_image)
    
    return filtered_keypoints, filtered_descriptors