import cv2
import matplotlib.pyplot  as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import random
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import hsv_to_rgb
import joblib

# paths
images_dir = './data-collection/images/'
annotations_dir = './data-collection/annotations'

red_count = 0
green_count = 0
undefined_count = 0

processed_images = []
brightness_mean_values = []

apple_pixels = []
non_apple_pixels = []

def classify_apple_color(hue_values, red_hue_range, green_hue_range):

    red_pixels = np.sum((hue_values >= red_hue_range[0]) & (hue_values <= red_hue_range[1])) + np.sum((hue_values >= red_hue_range[2]) & (hue_values <= red_hue_range[3]))
    green_pixels = np.sum((hue_values >= green_hue_range[0]) & (hue_values <= green_hue_range[1]))
    label = 'red' if red_pixels > green_pixels or red_pixels/len(hue_values) > 0.15 else 'green'

    return label

# hue ranges for red and green for opencv
red_hue_range = [0, 25, 145, 180]
green_hue_range = [35, 75]

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

        #mask for all apples in this image
        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for _, row in df.iterrows():
            # get co-ordinates for apple
            x_center, y_center = int(row['c-x']), int(row['c-y'])
            radius = int(row['radius'])
            
            # mask circle for the circular region
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x_center, y_center), radius, 1, -1)
            full_mask[mask == 1] = 1
            # get pixels from the circle
            apple_pixels_current = hsv_img[mask == 1]
            apple_pixels.extend(apple_pixels_current) #for K-NN
            
            # get brightness values
            brightness_mean = np.mean(apple_pixels_current[:, 2])
            brightness_mean_values.append(brightness_mean)
            
            #label apply
            if brightness_mean < 50 or brightness_mean > 223: #calculated after plot
                label = 'undefined'
                undefined_count += 1
                color = (128, 128, 128) 
            else:
                hue_values = apple_pixels_current[:, 0]
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

        #non-apple pixels for K-NN (only works with a good computer, lots of ram needed)
        # if len(non_apple_candidates[0]) > 0:
        #     non_apple_pix = hsv_img[non_apple_candidates[0], non_apple_candidates[1]]
        #     non_apple_pixels.extend(non_apple_pix)

        non_apple_candidates = np.where(full_mask == 0)
        if len(non_apple_candidates[0]) > 0:
            #limit samples per image to avoid crashing
            samples_needed = min(500, len(non_apple_candidates[0]))
            idxs = np.random.choice(len(non_apple_candidates[0]), samples_needed, replace=False)
            non_apple_pix = hsv_img[non_apple_candidates[0][idxs], non_apple_candidates[1][idxs]]
            non_apple_pixels.extend(non_apple_pix)

        #sample images
        if annotation_file in sampled_files:
            processed_images.append((image_path, img))

# Calculate the IQR for brightness values
# q1 = np.percentile(brightness_mean_values, 25)
# q3 = np.percentile(brightness_mean_values, 75)
# iqr = q3 - q1

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
#plt.savefig("./Documentation/brightness_distribution.pdf", format='pdf', bbox_inches='tight')
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

print("Finished labelling data")

######K-NN and hsv analysis

desired_ratio = 20  # non-apple:apple
non_apple_count = len(non_apple_pixels)
apple_needed = non_apple_count // desired_ratio  # integer division for simplicity
print(f"Non-apple pixels available: {non_apple_count}")
print(f"Apple pixels needed for 20:1 ratio: {apple_needed}")

# Downsample apple pixels
apple_pixels = np.array(apple_pixels)
p = np.random.permutation(len(apple_pixels))
apple_pixels = apple_pixels[p[:apple_needed]]
non_apple_pixels = np.array(non_apple_pixels)

#labels: 1 for apple, 0 for non-apple
X = np.vstack((apple_pixels, non_apple_pixels))
y = np.array([1]*len(apple_pixels) + [0]*len(non_apple_pixels))

#shuffle data to avoid bias
p = np.random.permutation(len(X))
X, y = X[p], y[p]

#fit K-NN on all data
knn = KNeighborsClassifier(n_neighbors=30, algorithm='kd_tree')
knn.fit(X, y)
#save model
joblib.dump(knn, 'Scripts/knn_model.pkl')

h_vals = np.arange(0, 181, 1)
s_vals = np.arange(0, 256, 5)
v_vals = np.arange(0, 256, 5)

H, S, V = np.meshgrid(h_vals, s_vals, v_vals, indexing='ij')
grid_points = np.column_stack((H.flatten(), S.flatten(), V.flatten()))

# Predict with K-NN
pred_labels = knn.predict(grid_points)

# Extract only the points predicted as apple
apple_points = grid_points[pred_labels == 1]
non_apple_points = grid_points[pred_labels == 0]

#normalize S and V to [0,1]
H_apple = apple_points[:,0]
S_apple = apple_points[:,1] / 255.0
V_apple = apple_points[:,2] / 255.0

H_full = H_apple * 2  
H_rad = np.deg2rad(H_full) #convert to radians
#convert to Cartesian
S_constrained = S_apple * V_apple
X = S_constrained * np.cos(H_rad)
Y = S_constrained * np.sin(H_rad)
Z = V_apple * 2

fig = plt.figure(figsize=(10,8)) 
ax = fig.add_subplot(111, projection='3d')

theta = np.linspace(0, 2*np.pi, 72)  # angular resolution
z = np.linspace(0, 2, 20)             # height resolution
Theta, Z_mesh = np.meshgrid(theta, z)      # Create 2D grid for angle and height
R = Z_mesh / 2  # Radius decreases as height decreases (cone shape)
X_mesh = R * np.cos(Theta)
Y_mesh = R * np.sin(Theta)

#plot
ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color='gray', alpha=0.1, edgecolor='black', linewidth=0.05)

# Normalize H, S, V values for conversion
H_norm = H_apple / 180.0 
hsv_col = np.column_stack((H_norm, S_apple, V_apple))
rgb_values = hsv_to_rgb(hsv_col)

# Plot the apple points inside the cone
ax.scatter(X, Y, Z, c=rgb_values, alpha=0.9, s=1)

ax.tick_params(axis='both', which='major', labelsize=9)
ax.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Brightness')

ax.view_init(elev=30, azim=135)

#plt.title('Apple Pixels (HSV) KNN Predictions')
#plt.show()

plt.savefig("./Documentation/hsv_cone_with_apple_points.pdf", format='pdf', bbox_inches='tight')
#plt.close()

print("done")
