import cv2
import os
from sklearn.cluster import KMeans
import numpy as np

def preprocess_image(image_path, knn):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None, None

    #step 1: convert to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #step 2: histogram equalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    equalized_image = clahe.apply(gray_image)
    
    #step 3: gaussian blur (might need to update to mediam blur, gaussian did not preserve edges well)
    #blurred_image = cv2.GaussianBlur(equalized_image, (3, 3), 0.3)
    blurred_image = cv2.medianBlur(equalized_image, 5)

    #step 4: morphological operation
    kernel = np.ones((3, 3), np.uint8)  
    morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)

    #step 5: get region of interest for feature extraction in SIFT
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = image_hsv.shape
    pixels = image_hsv.reshape(-1, 3)
    predicted_labels = knn.predict(pixels) #predict labels using knn
    mask = predicted_labels.reshape(height, width)
    binary_mask = (mask == 1).astype(np.uint8) * 255

    #step 5.1: segment green apples
    #lower_green1 = np.array([85, 0, 50])   #0130320T012914.905227_42.png & 20130320T005916.773278.Cam6_31.png
    #upper_green1 = np.array([179, 105, 255])
    #steap 5.2: segment red apples 
    # lower_red1 = np.array([130, 0, 150])   #(BD12_sup_201711_093_09_RGBhr.png) & BD11_inf_201710_081_08_RGBhr.png
    # upper_red1 = np.array([179, 255, 255])
    # lower_red2 = np.array([115, 0, 40])   # 20130320T004608.376022.Cam6_51.png & 20130320T005755.628752.Cam6_23.png
    # upper_red2 = np.array([179, 255, 255]) 
    #step 5.3: create mask
    #mask_green1 = cv2.inRange(image_hsv, lower_green1, upper_green1)
    #mask_red1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    #mask_red2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
    #mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    #mask_all = mask_red

    # Step 5.3.2: Check if the mask is valid
    if cv2.countNonZero(binary_mask) == 0:  # No mask created
        return morph_image, None

    #step 5.4: apply mask
    segment = cv2.bitwise_and(morph_image, morph_image, mask=binary_mask)
    #step 5.5: erode and dilate edges to expand non-background regions
    eroded_edges = cv2.erode(segment, np.ones((8, 8), np.uint8), iterations=1)
    dilated_edges = cv2.dilate(eroded_edges, np.ones((30, 30), np.uint8), iterations=1)
    roi_mask = cv2.threshold(dilated_edges, 1, 255, cv2.THRESH_BINARY)[1]

    #TODO: currently all hyper parameters are used as default, it required fine-tunning

    #check for ROI mask
    if cv2.countNonZero(roi_mask) == 0:
        return morph_image, None

    return morph_image, roi_mask

def process_and_save_images(image_paths, save_dir, sample_dir, sample_files, save_visualization=False):
    masks = {}

    for img_path in image_paths:
        save_visualization = img_path in sample_files #check if file is in sample
        processed_image, mask = preprocess_image(img_path)

        if processed_image is not None:
            #save path
            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, processed_image)
            masks.update({img_name: mask})
        else:
            print(f"Skipping image: {img_path}")

            #save sample image with keypoints
        if save_visualization and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if mask is None:
                img_name = os.path.basename(img_path)
                save_path = os.path.join(sample_dir, img_name)
                cv2.imwrite(save_path, cv2.imread(img_path))
                continue
            #use the mask on the greyscale image
            overlay = cv2.addWeighted(cv2.imread(img_path), 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(overlay, [contour], -1, (255, 0, 255), 2)  # Purple color in BGR

            img_name = os.path.basename(img_path)
            save_path = os.path.join(sample_dir, img_name)
            cv2.imwrite(save_path, overlay)

    return masks
    

def extract_and_filter_features(image_path, save_visualization=False, save_dir=None, min_size=20, max_size=80, mask=None):
    #load
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image: {image_path}")
        return [], None

    #SIFT detector, need to use in grid-search
    sift = cv2.SIFT_create(
        nfeatures=0,             #number of best features to retain (0 means no limit)
        nOctaveLayers=4,         #number of layers in each octave
        contrastThreshold=0.02,  #threshold for filtering out weak features
        edgeThreshold=8         #threshold for edge detection
    )
    
    #detect
    keypoints, descriptors = sift.detectAndCompute(image, mask=mask)
    
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

def detect_circles(image, dp=1, minDist=30, param1=223, param2=30, minRadius=5, maxRadius=40):
    """
    detect circles in an image using the Hough Circle Transform.

    Parameters:
    - image: grayscale input image.
    - dp: inverse ratio of the accumulator resolution to the image resolution.
    - minDist: Minimum distance between detected centers (min distance between apple centers).
    - param1: 	first method-specific parameter. In case of HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
    - param2: second method-specific parameter. In case of HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. .
    - minRadius: minimum apple radius.
    - maxRadius: maximum apple radius.

    Returns:
    - circles: detected circles as an array of (x, y, radius).
    """

    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
    else:
        circles = np.array([])

    return circles

def filter_circles_with_keypoints(circles, keypoints):
    """
    Filter circles by checking if they contain any SIFT keypoints.

    Parameters:
    - circles: Detected circles.
    - keypoints: List of SIFT keypoints.

    Returns:
    - filtered_circles: Circles that contain at least one keypoint. THis gets rid of false positives (like leafes)
    """
    filtered_circles = []

    for circle in circles:
        x, y, r = circle
        for kp in keypoints:
            kp_x, kp_y = kp.pt
            distance = np.sqrt((kp_x - x) ** 2 + (kp_y - y) ** 2)
            if distance <= r:
                filtered_circles.append(circle)
                break  

    return filtered_circles

def segment_apples(image_path, keypoints=None, save_visualization=False, save_dir=None):
    """
    Segment apples in the greyscale image using mask ROI from pre-processing.

    Returns:
    - apple_circles: List of detected apple circles.
    """
    #Step1: load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, []

    #step2: detect circles
    circles = detect_circles(image)

    #step3: filter circles with keypoints
    apple_circles = filter_circles_with_keypoints(circles, keypoints)

    #step4: sample image with detected circles
    if save_visualization and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result_image = image.copy()
        for circle in apple_circles:
            x, y, r = circle
            # Draw the outer circle
            cv2.circle(result_image, (x, y), r, (255, 0, 255), 2)
            # Draw the center of the circle
            cv2.circle(result_image, (x, y), 2, (255, 0, 255), 3)
        img_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, result_image)

    return apple_circles

def extract_mean_hsv(image_path, circle):
    """
    Extract the mean HSV values from the area within the circle.
    """
    x, y, r = circle
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    #extract the HSV image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #calculate the mean HSV values within the circle
    mean_hsv = cv2.mean(hsv_image, mask=mask)[:3]

    return np.array(mean_hsv)

def perform_kmeans_clustering(hsv_features, n_clusters=3):
    """
    Perform k-means clustering on HSV features.

    Parameters:
    - hsv_features: list of all hsv vectors.
    - n_clusters: Number of clusters (default is 3).
    """
    hsv_features = np.array(hsv_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(hsv_features[:, :2]) #only use HUE and saturation for labels
    return labels, kmeans

def assign_cluster_labels(kmeans):
    """
    Assign labels to clusters based on the mean Hue value.
    """
    cluster_labels = {}
    cluster_centers = kmeans.cluster_centers_

    for idx, center in enumerate(cluster_centers):
        hue = center[0]
        if (hue < 10) or (hue > 160):
            cluster_labels[idx] = 'red'
        elif 35 < hue < 85:
            cluster_labels[idx] = 'green'
        else:
            cluster_labels[idx] = 'undefined'

    return cluster_labels

import pandas as pd

def create_classification_dataframe(apple_labels):
    """
    create a pandas DataFrame from the classification results.
    """
    rows = []
    for image_name, apples in apple_labels.items():
        for apple in apples:
            circle, label = apple
            x, y, r = circle
            rows.append({
                'image': image_name,
                'c-x': x,
                'c-y': y,
                'radius': r,
                'label': label
            })

    df = pd.DataFrame(rows, columns=['image', 'c-x', 'c-y', 'radius', 'label'])
    return df

