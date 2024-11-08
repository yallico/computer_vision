import os
import shutil
from PIL import Image
import pandas as pd
import re

source_folders = ["acfr-multifruit-2016", "KFuji_RGB-DS_dataset"]
target_folder = "data-collection"
annotations_folder = os.path.join(target_folder, "annotations")
images_folder = os.path.join(target_folder, "images")

# clear temp data in folders
if os.path.exists(annotations_folder):
    shutil.rmtree(annotations_folder)
if os.path.exists(images_folder):
    shutil.rmtree(images_folder)

# re-crate target directories
os.makedirs(annotations_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

def process_acfr_annotations(filepath, target_path):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["label"])
    df.to_csv(target_path, index = False)

def process_kfuji_annotations(filepath, target_path):
    df = pd.read_csv(filepath, header=None, names=["id", "xmin", "ymin", "width", "height", "label"])
    # calculate the center x and y coordinates
    df["c-x"] = df["xmin"] + (df["width"] / 2)
    df["c-y"] = df["ymin"] + (df["height"] / 2)
    # approximate radius using the average of width and height
    df["radius"] = round((((df["width"]/2) + (df["height"]/2)) / 2), 2)
    df["c-x"] =  round(df["c-x"],2)
    df["c-y"] =  round(df["c-y"],2)
    df = df[["c-x", "c-y", "radius"]].reset_index(drop=True)
    df.to_csv(target_path, index_label="#item")

for source in source_folders:
    for root, _, files in os.walk(source):
        for file in files:
            source_path = os.path.join(root, file)
            # .csv files to annotations folder
            if file.endswith(".csv"):
                target_csv_path = os.path.join(annotations_folder, f"{os.path.splitext(file)[0]}.csv")
                if source == "acfr-multifruit-2016":
                    process_acfr_annotations(source_path, target_csv_path)
                elif source == "KFuji_RGB-DS_dataset":
                    process_kfuji_annotations(source_path, target_csv_path)
            # convert and save images as .png in images folder
            elif file.endswith((".jpg", ".png")):
                pattern = r".*RGBp\.jpg$"
                if bool(re.match(pattern, file)):
                    continue
                image = Image.open(source_path)
                # convert to .png if it's a .jpg file
                target_image_path = os.path.join(images_folder, f"{os.path.splitext(file)[0]}.png")
                image.save(target_image_path, format="PNG")

print("Data and image conversion completed successfully.")
