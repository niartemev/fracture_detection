#Kfold cross validation

#Import libraries
from ultralytics import YOLO
import os
import glob
import shutil
from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
from IPython.display import clear_output
import time

# Paths for consolidated image and label data
CONSOLIDATED_IMAGES_PATH = '\\kfold\\images\\'
CONSOLIDATED_LABELS_PATH = 'kfold\\labels\\'

# Template for the original dataset structure
SOURCE_DATA_PATH_TEMPLATE = "\\processed\\{data_split}\\{data_type}"

# Data categories and file types to copy
dataset_splits = ["train", "valid"]
data_types = ["images/*.jpg", "labels/*.txt"]

# Create folders if they don't exist
os.makedirs(CONSOLIDATED_IMAGES_PATH, exist_ok=True)
os.makedirs(CONSOLIDATED_LABELS_PATH, exist_ok=True)

# Gather all image and label files into consolidated folders
for split in dataset_splits:
    for file_type in data_types:
        matching_files = glob.glob(SOURCE_DATA_PATH_TEMPLATE.format(data_split=split, data_type=file_type))
        for filepath in matching_files:
            if "image" in file_type:
                shutil.copy(filepath, CONSOLIDATED_IMAGES_PATH)
            else:
                shutil.copy(filepath, CONSOLIDATED_LABELS_PATH)

# Get lists of all image and label paths
all_image_paths = glob.glob(CONSOLIDATED_IMAGES_PATH + "*.jpg")
all_label_paths = glob.glob(CONSOLIDATED_LABELS_PATH + "*.txt")

# Base path to all processed data
BASE_DATASET_PATH = Path('\\working\\processed\\')
all_labels = sorted(BASE_DATASET_PATH.rglob("*labels/*.txt"))

# Load class names from YAML file
YAML_CONFIG_PATH = '\\dataset\\data.yaml'
with open(YAML_CONFIG_PATH, 'r', encoding="utf8") as file:
    class_names = yaml.safe_load(file)['names']
class_indices = list(range(len(class_names)))

# Prepare DataFrame to hold label counts per image
image_ids = [label_file.stem for label_file in all_labels]  # filenames without extension
label_count_df = pd.DataFrame([], columns=class_indices, index=image_ids)

# Parse each label file and count class occurrences
for label_file in all_labels:
    label_counter = Counter()
    with open(label_file, 'r') as lf:
        lines = lf.readlines()
    for line in lines:
        class_id = int(line.split(' ')[0])  # YOLO format: class_id x_center y_center width height
        label_counter[class_id] += 1
    label_count_df.loc[label_file.stem] = label_counter

# Replace NaNs with 0 for classes not present in certain images
label_count_df = label_count_df.fillna(0.0)

# Set number of folds for cross-validation
NUM_FOLDS = 3
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=20)
fold_indices = list(kf.split(label_count_df))

# Evaluate class distribution across folds
fold_names = [f'fold_{i+1}' for i in range(NUM_FOLDS)]
fold_class_balance_df = pd.DataFrame(index=fold_names, columns=class_indices)

for fold_number, (train_indices, val_indices) in enumerate(fold_indices, start=1):
    train_label_sum = label_count_df.iloc[train_indices].sum()
    val_label_sum = label_count_df.iloc[val_indices].sum()
    class_ratio = val_label_sum / (train_label_sum + 1E-7)  # Prevent divide-by-zero
    fold_class_balance_df.loc[f'fold_{fold_number}'] = class_ratio

# Prepare directories and config files for training
KFOLD_OUTPUT_PATH = Path('\\working\\kfinal')
if KFOLD_OUTPUT_PATH.is_dir():
    shutil.rmtree(KFOLD_OUTPUT_PATH)
os.makedirs(str(KFOLD_OUTPUT_PATH))

train_txt_list = []
val_txt_list = []
yaml_config_paths = []

# Create image path lists and YAML files for each fold
for fold_index, (train_idx, val_idx) in enumerate(fold_indices):
    train_images = [all_image_paths[i] for i in train_idx]
    val_images = [all_image_paths[i] for i in val_idx]

    train_txt_file = KFOLD_OUTPUT_PATH / f"train_{fold_index}.txt"
    val_txt_file = KFOLD_OUTPUT_PATH / f"val_{fold_index}.txt"

    with open(train_txt_file, 'w') as f:
        f.writelines(image + '\n' for image in train_images)
    with open(val_txt_file, 'w') as f:
        f.writelines(image + '\n' for image in val_images)

    train_txt_list.append(str(train_txt_file))
    val_txt_list.append(str(val_txt_file))

    yaml_output_path = KFOLD_OUTPUT_PATH / f'data_{fold_index}.yaml'
    with open(yaml_output_path, 'w') as yaml_out:
        yaml.safe_dump({
            'train': str(train_txt_file.name),
            'val': str(val_txt_file.name),
            'names': class_names
        }, yaml_out)
    yaml_config_paths.append(str(yaml_output_path))

# Training configuration
BATCH_SIZE = 16
PROJECT_NAME = 'kfold'
EPOCHS = 1

training_results = []

# Train a YOLO model for each fold
for fold_idx in range(NUM_FOLDS):
    model = YOLO('yolov8s.pt')
    print(f"Training model for fold={fold_idx} using {yaml_config_paths[fold_idx]}")
    model.train(
        data=yaml_config_paths[fold_idx],
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        epochs=EPOCHS,
        verbose=False,
        workers=8,
        device=1
    )
    result_metrics = model.metrics  # Get validation metrics
    training_results.append(result_metrics)
    clear_output()

# Collect and summarize metrics
metrics_summary = dict()
for result in training_results:
    for metric_name, value in result.results_dict.items():
        metrics_summary.setdefault(metric_name, []).append(value)

# Convert to DataFrame for analysis
metrics_df = pd.DataFrame.from_dict(metrics_summary)

# Output summary statistics for key metrics
stats_to_show = ['mean', 'std', 'min', 'max']
metrics_df.describe().loc[stats_to_show]
