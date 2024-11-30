import os
import random
from functions import *
from tqdm import tqdm

image_dir = 'data-collection/images'
all_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]

#seed shuffle for repeatability
random.seed(42)
random.shuffle(all_images)
#do 70/30 split
split_index = int(len(all_images) * 0.7)
#split the dataset
train_images = all_images[:split_index]
test_images = all_images[split_index:]

#set directories for pre-processed images
train_dir = 'pre-processed/train_images'
test_dir = 'pre-processed/test_images'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#pre-process images
process_and_save_images(train_images, train_dir)
process_and_save_images(test_images, test_dir)

#extract features
image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.endswith('.png')]
num_samples = 10  #sample for validation
sample_images = random.sample(image_paths, num_samples)
vis_dir_feature = 'sample-images-features/'
all_keypoints = []
all_descriptors = []

for img_path in tqdm(image_paths):
    save_vis = img_path in sample_images
    keypoints, descriptors = extract_and_filter_features(
        img_path, save_visualization=save_vis, save_dir=vis_dir_feature
    )
    all_keypoints.append(keypoints)
    all_descriptors.append(descriptors)


print("Done")