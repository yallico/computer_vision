import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def process_csv_files(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not all_files:
        print("No CSV files found in the folder.")
        return
    
    dataframes = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            df["image"] = file.replace('RGBhr','RGB') #handle hr suffix
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files to process.")
        return
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['image'] = combined_df['image'].str.replace('.csv', '.png', regex=False)

    return combined_df

def calculate_rmse_counts(detections, ground_truth_counts):
    """
    calculate RMSE of apple counts per image and across the whole dataset.
    Returns:
    - per_image_rmse: dictionary {image_name: float} RMSE per image 
    - total_rmse: float, overall RMSE across all images
    """
    errors = []
    per_image_rmse = {}

    for image_name, predicted_apples in detections.items():
        predicted_count = len(predicted_apples)
        gt_count = ground_truth_counts.get(image_name.split('/')[2].replace('RGBhr','RGB'), 0)
        error = (predicted_count - gt_count)**2
        errors.append(error)

        per_image_rmse[image_name.split('/')[2].replace('RGBhr','RGB')] = np.sqrt(error)

    total_rmse = np.sqrt(np.mean(errors)) if errors else 0.0
    return per_image_rmse, total_rmse

def distance(c1, c2):
    """
    distance between two circle centers
    """
    x1, y1, _ = c1
    x2, y2, _ = c2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def match_apples_by_min_distance(predicted_circles, ground_truth_circles):
    """
    match predicted apples to ground truth apples using a greedy approach based on minimum center distance.

    Returns:
    - matches: list of tuples (pred_idx, gt_idx) indicating which predicted apple 
               matches which ground truth apple.
    - unmatched_pred: list of indices of predicted apples that couldn't be matched
    - unmatched_gt: list of indices of ground truth apples that remain unmatched
    """

    #if both lists are empty, no matches can be made
    if not predicted_circles or not ground_truth_circles:
        return [], list(range(len(predicted_circles))), list(range(len(ground_truth_circles)))

    #calculate all distances between predicted and ground truth apples
    distance_list = []
    for i, pred_circle in enumerate(predicted_circles):
        for j, gt_circle in enumerate(ground_truth_circles):
            dist = distance(pred_circle, gt_circle)
            distance_list.append(((i, j), dist))

    #sort by distance
    distance_list.sort(key=lambda x: x[1])

    matched_pred = set()
    matched_gt = set()
    matches = []

    #greedy match, go through the sorted list and pick the first available pair
    for (pred_idx, gt_idx), dist in distance_list:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            matches.append((pred_idx, gt_idx))

    unmatched_pred = [i for i in range(len(predicted_circles)) if i not in matched_pred]
    unmatched_gt = [j for j in range(len(ground_truth_circles)) if j not in matched_gt]

    return matches, unmatched_pred, unmatched_gt

def circle_iou(circle1, circle2):
    """
    calculate the intersection over union (IoU) between two circular bounding structures

    Parameters:
    - circle1: (x1, y1, r1)
    - circle2: (x2, y2, r2)

    Returns:
    - iou: float, the IoU between the two circles.
    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    #distance between circle centers
    d = distance(circle1, circle2)
    #area of each circle
    area1 = math.pi * (r1**2)
    area2 = math.pi * (r2**2)

    #if circles do not overlap return 0.0 for IoU
    if d >= (r1 + r2):
        return 0.0

    #if one circle is completely inside the other
    if d <= abs(r1 - r2):
        #IoU is the area of the smaller circle
        intersection = math.pi * (min(r1, r2)**2)
    else:
        #partial overlap case
        r1_sq = r1**2
        r2_sq = r2**2
        #angles for the segments
        alpha = math.acos((d**2 + r1_sq - r2_sq) / (2 * d * r1)) * 2
        beta = math.acos((d**2 + r2_sq - r1_sq) / (2 * d * r2)) * 2
        #area of the segment in circle1
        segment_area1 = 0.5 * r1_sq * (alpha - math.sin(alpha))
        #area of the segment in circle2
        segment_area2 = 0.5 * r2_sq * (beta - math.sin(beta))

        intersection = segment_area1 + segment_area2

    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics (precision, recall, F1) for the three labels
    """
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics_dict = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics_dict