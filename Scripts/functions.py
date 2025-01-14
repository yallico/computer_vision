import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from data_collection_apple_labels import classify_apple_color, red_hue_range, green_hue_range

def preprocess_image(image_path, knn, scale_factor = 0.1):
    image = cv2.imread(image_path)
    if image is None:
        print(f"error loading image: {image_path}")
        return None, None
    original_height, original_width = image.shape[:2]

    #step 1 convert to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #step 2 histogram equalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    equalized_image = clahe.apply(gray_image)
    
    #step 3 gaussian blur (might need to update to mediam blur, gaussian did not preserve edges well)
    blurred_image = cv2.medianBlur(equalized_image, 5)

    #step 4 morphological operation
    kernel = np.ones((3, 3), np.uint8)  
    morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)

    #step 5 get region of interest for feature extraction in SIFT
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height)) #resize the image to a smaller resolution
    image_hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    height, width, _ = image_hsv.shape
    pixels = image_hsv.reshape(-1, 3)
    predicted_labels = knn.predict(pixels) #predict labels using knn
    mask = predicted_labels.reshape(height, width)
    binary_mask = (mask == 1).astype(np.uint8) * 255
    if cv2.countNonZero(binary_mask) == 0: 
        return morph_image, None
    resized_mask = cv2.resize(
        binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
        )

    #step 5.4 apply mask
    segment = cv2.bitwise_and(morph_image, morph_image, mask=resized_mask)
    #step 5.5 erode and dilate edges to expand non-background regions
    eroded_edges = cv2.erode(segment, np.ones((8, 8), np.uint8), iterations=1)
    dilated_edges = cv2.dilate(eroded_edges, np.ones((30, 30), np.uint8), iterations=1)
    #TODO: currently erode/dilate is heuristic, it required fine-tunning
    roi_mask = cv2.threshold(dilated_edges, 1, 255, cv2.THRESH_BINARY)[1]

    #check for roi mask
    if cv2.countNonZero(roi_mask) == 0:
        return morph_image, None

    return morph_image, roi_mask

def process_and_save_images(image_paths, save_dir, sample_dir, sample_files, knn, save_visualization=False):
    masks = {}

    for img_path in image_paths:
        save_visualization = img_path in sample_files
        processed_image, mask = preprocess_image(img_path, knn)

        if processed_image is not None:
            #save path
            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, processed_image)
            masks.update({img_name: mask})
        else:
            print(f"skipping image: {img_path}")

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
                cv2.drawContours(overlay, [contour], -1, (255, 0, 255), 2) #visualise areas

            img_name = os.path.basename(img_path)
            save_path = os.path.join(sample_dir, img_name)
            cv2.imwrite(save_path, overlay)

    return masks
    

def extract_and_filter_features(image_path, save_visualization=False, save_dir=None, min_size=20, max_size=80, mask=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"error loading image: {image_path}")
        return [], None

    #SIFT detector hyperparamenters
    sift = cv2.SIFT_create(
        nfeatures=0,             #number of best features to retain
        nOctaveLayers=8,         #number of layers in each octave
        contrastThreshold=0.005,
        edgeThreshold=3         
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

#feature extractom
def detect_circles(image, dp=1, minDist=20, param1=220, param2=25, minRadius=5, maxRadius=40):
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

#match sift with features
def filter_circles_with_keypoints(circles, keypoints):
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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"failed to load image: {image_path}")
        return None, []
    #detect circles
    circles = detect_circles(image)
    #filter circles with keypoints
    apple_circles = filter_circles_with_keypoints(circles, keypoints)
    #sample image with detected circles
    if save_visualization and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result_image = image.copy()
        for circle in apple_circles:
            x, y, r = circle
            cv2.circle(result_image, (x, y), r, (255, 0, 255), 2)
            cv2.circle(result_image, (x, y), 2, (255, 0, 255), 3)
        img_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, result_image)

    return apple_circles

def extract_hsv(image_path, circle):
    x, y, r = circle
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    #extract the hsv image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_values = hsv_image[mask == 255]

    return hsv_values

def classify_by_hsv(hsv_features):
    labels = []
    for feature in hsv_features:
        labels.append(classify_apple_color(feature[0], red_hue_range, green_hue_range))
    return labels