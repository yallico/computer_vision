import os
import shutil
from PIL import Image
import pandas as pd

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
    df = pd.read_csv(filepath, header=None, names=["id", "xmin", "ymin", "xmax", "ymax", "label"])
    # calculate the center x and y coordinates
    df["c-x"] = (df["xmin"] + df["xmax"]) / 2
    df["c-y"] = (df["ymin"] + df["ymax"]) / 2
    # calculate radius from width (assuming the bounding box width equals height)
    df["radius"] = round(((df["xmax"] - df["xmin"]) / 2),2)
    df["c-x"] =  round(df["c-x"],1)
    df["c-y"] =  round(df["c-y"],1)
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
                image = Image.open(source_path)
                # convert to .png if it's a .jpg file
                target_image_path = os.path.join(images_folder, f"{os.path.splitext(file)[0]}.png")
                image.save(target_image_path, format="PNG")

print("Data consolidation and image conversion completed successfully.")
