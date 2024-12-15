import os
import random
from functions import *
from data_statistics import *
from tqdm import tqdm
import joblib

knn = joblib.load('Scripts/knn_model.pkl')

num_samples = 30  #sample for validation
image_dir = 'data-collection/images'
all_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]

#ingest annotations and combine in single df
folder_path = "data-collection/annotations/"
combined_df = process_csv_files(folder_path)
print(combined_df.describe(include='all'))

#seed shuffle for repeatability
random.seed(42)
#random.shuffle(all_images)
#do 70/30 split
#split_index = int(len(all_images) * 0.7)
#train_images = all_images[:split_index]
#test_images = all_images[split_index:]

#set directories for pre-processed images
dir = 'pre-processed'
os.makedirs(dir, exist_ok=True)

#pre-process images
vis_dir_feature = 'pre-processed/sample'
sample_images = random.sample(all_images, num_samples) #sample for visualization
roi_mask_train = process_and_save_images(all_images, dir, vis_dir_feature, sample_images, knn=knn)
sample_images = [img for img in os.listdir(vis_dir_feature) if img.endswith('.png')] #update sample paths

no_masks_train = len([x for x in roi_mask_train.values() if x is None])

print("Image Pre-processing:")
print(f"out of: {len(roi_mask_train)} images, {no_masks_train} images have no Mask/RIO")
print(f"the ratio of images with no masks is: {no_masks_train/len(roi_mask_train):.2f}")

#extract features
image_paths = [os.path.join(dir, img) for img in os.listdir(dir) if img.endswith('.png')]
vis_dir_feature = 'sample-features'
keypoints_dict = {}
all_descriptors = []

for img_path in tqdm(image_paths):
    save_vis = img_path.split('/')[1] in sample_images
    keypoints, descriptors = extract_and_filter_features(
        img_path, save_visualization=save_vis, save_dir=vis_dir_feature, mask=roi_mask_train[os.path.basename(img_path)]
    )
    keypoints_dict[img_path] = keypoints
    all_descriptors.append(descriptors)

no_keypoints = len([x for x in keypoints_dict.values() if len(x) == 0])

print("Feature Extraction:")
print(f"out of: {len(keypoints_dict)} images, {no_keypoints} images have no keypoints detected.")
print(f"the ratio of images with no keypoints is: {no_keypoints/len(keypoints_dict):.2f}")

#segment features using circle shape detection
vis_dir_segment = 'sample-segment'
detections = {}
hsv_features = [] #mean hsv values for circles
apple_indices = []  #for tracking the apple indices

for img_path in tqdm(keypoints_dict.keys()):
    save_vis = img_path.split('/')[1] in sample_images
    # Segment apples in the image
    apple_circles = segment_apples(
        img_path, keypoints=keypoints_dict[img_path], save_visualization=save_vis, save_dir=vis_dir_segment
        )

    # Extract mean HSV values for each apple
    for circle in apple_circles:
        hsv_array = extract_hsv(f"{image_dir}/{img_path.split('/')[1]}", circle)
        hsv_features.append(hsv_array)
        apple_indices.append((img_path, circle))

    # Save the results in a dict
    detections[img_path] = apple_circles

#classification: re-use algorigthm from data_collection_apple_labels.py
labels = classify_by_hsv(hsv_features)
#update apple_labels
apple_labels = {}
for idx, (image_file, circle) in enumerate(apple_indices):
    color_label = labels[idx]
    image_name = image_file.split('/')[1]
    if image_name not in apple_labels:
        apple_labels[image_name.replace("RGBhr", "RGB")] = []
    apple_labels[image_name.replace("RGBhr", "RGB")].append((circle, color_label))

#Get ground truth data
ground_truth_counts = combined_df.groupby('image')['radius'].count().to_dict()
ground_truth_apples = combined_df.groupby('image')[['c-x', 'c-y', 'radius']].apply(lambda x: x.values.tolist()).to_dict()
ground_truth_labels = combined_df.groupby('image')['label'].apply(lambda x: x.values.tolist()).to_dict()

#Metric evaluation: RSME
per_image_rmse, total_rmse = calculate_rmse_counts(detections, ground_truth_counts)
print(f"RMSE per image: {total_rmse}")   

#Metric evaluation: IoU
all_matches = {}           # {image_name: [(pred_idx, gt_idx), ...]}
all_unmatched_pred = {}    # {image_name: [list_of_pred_idx]}
all_unmatched_gt = {}      # {image_name: [list_of_gt_idx]}
iou_results = {}           # {image_name: [IoU_values]}
labels_train = []
labels_pred = []

for image_name, predicted_apples in apple_labels.items():
    predicted_circles = [p[0] for p in predicted_apples]  #extract just (x, y, r)
    gt_circles = ground_truth_apples.get(image_name, [])

    # Run matching for this image
    matches, unmatched_pred, unmatched_gt = match_apples_by_min_distance(predicted_circles, gt_circles)

    # Store results
    all_matches[image_name] = matches
    all_unmatched_pred[image_name] = unmatched_pred
    all_unmatched_gt[image_name] = unmatched_gt

    #IoU
    iou_values = []

    for (pred_idx, gt_idx) in matches:
        pred_circle = predicted_circles[pred_idx]
        gt_circle = gt_circles[gt_idx]
        #IoU
        iou_val = circle_iou(pred_circle, gt_circle)
        iou_values.append(iou_val)
        # Classification: compare predicted_label vs ground_truth_labels
        predicted_label = predicted_apples[pred_idx][1]
        true_label = ground_truth_labels[image_name][gt_idx]
        labels_train.append(true_label)
        labels_pred.append(predicted_label)
    
    iou_results[image_name] = iou_values

print(f"Mean IoU: {np.mean([np.mean(v) for v in iou_results.values() if len(v) > 0])}")

#Metric evaluation: Classification
class_metric = compute_classification_metrics(labels_pred, labels_train)

print("Done")