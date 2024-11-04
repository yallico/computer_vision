# Coursework

To access the overleaf document please use: [Overleaf](https://www.overleaf.com/8895935383kvmprmwxdckv#2317ff)

## Datasets

- [ACFR Orchard Fruit Dataset](https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/) - 1120 images with circle annotations
- [KFuji RGB-DS database](https://www.grap.udl.cat/en/publications/kfuji-rgb-ds-database/) - 967 multi-modal images (RGB, Depth, Range Corrected Intensity) with rectangular bounding boxes
- [AmodalAppleSize_RGB-D](https://research.wur.nl/en/datasets/amodalapplesizergb-d) - Roughly 6500 over two subsets, RGB-D. Includes modal masks that highlight occluded regions. *Note*: We cannot use this dataset as it is not annotated with bounding boxes/circles, instead we only have the count and the diameter of each apple.

## Data Collection

Data from the above sources was put together in a single folder available on the following [OneDrive](https://uob-my.sharepoint.com/:u:/g/personal/np23992_bristol_ac_uk/EfrxU3YRFs1OjNt0rqRQTkgBtK4NK9F-UZ9a8ZfXhYFVBA?e=ghAuhw). Note that two sub-folders *annotations* and *images*. The relationship between the annotation .csv file and the .png file is the name of the file itself.

## CSRNet Initial Checkpoints

 - Model Checkpoint 0, KFuji dataset for training: [OneDrive](https://uob-my.sharepoint.com/:u:/r/personal/pe22304_bristol_ac_uk/Documents/Machine%20Vision/CSRNet/0model.pth.tar?csf=1&web=1&e=bAdQzu)
 - Model Checkpoint 1, our dataset for training: [OneDrive](https://uob-my.sharepoint.com/:u:/r/personal/pe22304_bristol_ac_uk/Documents/Machine%20Vision/CSRNet/1model.pth.tar?csf=1&web=1&e=Paav2f)

To use the checkpoints you will need to clone the Github Repo for CSRNet as it contains the model file. If anyone is interested in doing this and require assistance please let me (Colm) know.
