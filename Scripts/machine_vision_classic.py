import os
import random

image_dir = 'data-collection/images'
all_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]

# seed shuffle for repeatability
random.seed(42)
random.shuffle(all_images)

# do 70/30 split
split_index = int(len(all_images) * 0.7)

# Split the dataset
train_images = all_images[:split_index]
test_images = all_images[split_index:]

print("Done")