import os
import random
from functions import *
from data_statistics import process_csv_files
from tqdm import tqdm

num_samples = 20  #sample for validation
image_dir = 'data-collection/images'
all_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]

#seed shuffle for repeatability
random.seed(42)
random.shuffle(all_images)
#do 70/30 split
split_index = int(len(all_images) * 0.7)
train_images = all_images[:split_index]
test_images = all_images[split_index:]

#set directories for pre-processed images
train_dir = 'pre-processed/train_images'
test_dir = 'pre-processed/test_images'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#pre-process images
vis_dir_feature = 'pre-processed/sample'
sample_images = random.sample(all_images, num_samples) #sample for visualization
roi_mask_train = process_and_save_images(train_images, train_dir, vis_dir_feature, sample_images)
roi_mask_test = process_and_save_images(test_images, test_dir, vis_dir_feature, sample_images)

#extract features
image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.endswith('.png')]
vis_dir_feature = 'sample-features/'
keypoints_dict = {}
all_descriptors = []

for img_path in tqdm(image_paths):
    save_vis = img_path in sample_images
    keypoints, descriptors = extract_and_filter_features(
        img_path, save_visualization=save_vis, save_dir=vis_dir_feature, mask=roi_mask_train[os.path.basename(img_path)]
    )
    keypoints_dict[img_path] = keypoints
    all_descriptors.append(descriptors)

#segment features using circle shape detection
vis_dir_segment = 'sample-segment/'
detections = {}
hsv_features = [] #mean hsv values for circles
apple_indices = []  #for tracking the apple indices

for img_path in tqdm(keypoints_dict.keys()):
    save_vis = img_path in sample_images
    # Segment apples in the image
    apple_circles = segment_apples(img_path, keypoints=keypoints_dict[img_path], save_visualization=save_vis, save_dir=vis_dir_segment)

    # Extract mean HSV values for each apple
    for circle in apple_circles:
        mean_hsv = extract_mean_hsv(img_path, circle)
        hsv_features.append(mean_hsv)
        apple_indices.append((img_path, circle))

    # Save the results in a dict
    detections[img_path] = apple_circles

#Classification 
labels, kmeans = perform_kmeans_clustering(hsv_features, n_clusters=3) #k-means clustering
print("Cluster Centers (HSV):")
for idx, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {idx}: H={center[0]:.2f}, S={center[1]:.2f}, V={center[2]:.2f}")
cluster_labels = assign_cluster_labels(kmeans) # labels assignment
#update apple_labels
apple_labels = {}
for idx, (image_file, circle) in enumerate(apple_indices):
    cluster_idx = labels[idx]
    color_label = cluster_labels[cluster_idx]
    if image_file not in apple_labels:
        apple_labels[image_file] = []
    apple_labels[image_file].append((circle, color_label))


print("Done")