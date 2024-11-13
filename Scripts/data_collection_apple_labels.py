import cv2
import matplotlib.pyplot  as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import os
import re

# paths
images_dir = './data-collection/images/'
annotations_dir = './data-collection/annotations'

red_count = 0
green_count = 0
undefined_count = 0

processed_images = []

def classify_apple_color(hue_values, red_hue_range, green_hue_range):

    red_pixels = np.sum((hue_values >= red_hue_range[0]) & (hue_values <= red_hue_range[1])) + np.sum((hue_values >= red_hue_range[2]) & (hue_values <= red_hue_range[3]))
    green_pixels = np.sum((hue_values >= green_hue_range[0]) & (hue_values <= green_hue_range[1]))
    label = 'red' if red_pixels > green_pixels or red_pixels/len(hue_values) > 0.15 else 'green'

    return label

# hue ranges for red and green for opencv
red_hue_range = [0, 25, 145, 180]
green_hue_range = [35, 75]

#mean brightness of masks
brightness_mean_values = []

# sample images for validation
sampled_files = random.sample(os.listdir(annotations_dir), 15)

# for each annotation file
for annotation_file in os.listdir(annotations_dir):
    if annotation_file.endswith('.csv'):
        # match image file using regex
        base_name = re.match(r"(.*?)(?:hr)?\.csv", annotation_file).group(1)
        image_path = None
        possible_image_paths = [
            os.path.join(images_dir, f"{base_name}.png"),
            os.path.join(images_dir, f"{base_name}hr.png")
        ]
        
        for path in possible_image_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path is None:
            print(f"Image for {annotation_file} not found. Skipping.")
            continue
        
        # load the image
        img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # load annotation data
        df = pd.read_csv(os.path.join(annotations_dir, annotation_file))
        labels = []

        for _, row in df.iterrows():
            # get co-ordinates for apple
            x_center, y_center = int(row['c-x']), int(row['c-y'])
            radius = int(row['radius'])
            
            # mask circle for the circular region
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x_center, y_center), radius, 1, -1)
            # get pixels from the circle
            apple_pixels = hsv_img[mask == 1]
            
            # get hue values
            hue_values = apple_pixels[:, 0]
            # get brightness values
            brightness_mean = np.mean(apple_pixels[:, 2])
            brightness_mean_values.append(brightness_mean)
            
            # classify apply
            if brightness_mean < 50 or brightness_mean > 223: #calculated after plot
                label = 'undefined'
                undefined_count += 1
                color = (128, 128, 128) 
            else:
                hue_values = apple_pixels[:, 0]
                label = classify_apple_color(hue_values, red_hue_range, green_hue_range)
                if label == 'red':
                    red_count += 1
                    color = (0, 0, 255)
                else:
                    green_count += 1
                    color = (0, 255, 0)

            #draw the bounding circle
            cv2.circle(img, (x_center, y_center), radius, color, 1)
            
            labels.append(label)

        df['label'] = labels
        #df.to_csv(os.path.join(annotations_dir, annotation_file), index=False) #uncomments to overwrite data
        #sample images
        if annotation_file in sampled_files:
            processed_images.append((image_path, img))

# Calculate the IQR for brightness values
q1 = np.percentile(brightness_mean_values, 25)
q3 = np.percentile(brightness_mean_values, 75)
iqr = q3 - q1

# note: we can't use IQR because its a bimodal distribution

# Plot the brightness distribution with thresholds
plt.figure(figsize=(10, 6))
sns.histplot(brightness_mean_values, bins=30, kde=True, color="skyblue", edgecolor="black")
plt.axvline(50, color='red', linestyle='--', label='Low Threshold')
plt.axvline(223, color='green', linestyle='--', label='High Threshold')
plt.xlabel('Brightness (V-component)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("./Documentation/brightness_distribution.pdf", format='pdf', bbox_inches='tight')
#plt.show()

# Output the split between red and green apples
print(f"Total red apples: {red_count}")
print(f"Total green apples: {green_count}")
print(f"Total undefined apples: {undefined_count}")
print(f"Red to green split: {round(red_count/green_count,2)}")

output_dir = "./Sample/"

# clear sample
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

os.makedirs(output_dir, exist_ok=True)
for i, (name, processed_img) in enumerate(processed_images):
    base_name = '.'.join(os.path.basename(name).split('.')[:-1])
    output_path = os.path.join(output_dir, f"{base_name}_labeled.png")
    cv2.imwrite(output_path, processed_img)
    print(f"Saved: {output_path}")

print("done")