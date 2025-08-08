#Image preprocessing
#Load necessary libraries

import os, cv2, shutil
from tqdm.notebook import tqdm

#final path
processed_path = "working\\processed"

#Original dataset path. Taken from - https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project
dataset_path = "dataset"

#Preprocessing function
def preprocess(image_path):
    #Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Normalize and convert back
    img = img / 255.0
    return (img * 255).astype(np.uint8)

def main():

    #Create directories for each split
    for split in["train", "valid", "test"]:
        os.makedirs(f"{processed_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{processed_path}/{split}/labels", exist_ok=True)

    #Preprocess each image and copy it over to new directory
    for split in ["train", "valid", "test"]:
        images_dir = f"{dataset_path}/{split}/images"
        labels_dir = f"{dataset_path}/{split}/labels"
        output_images_dir = f"{processed_path}/{split}/images"
        output_labels_dir = f"{processed_path}/{split}/labels"

        for img_file in tqdm(os.listdir(images_dir), desc=f"Processing {split} images"):
            img_path = os.path.join(images_dir, img_file)
            processed = preprocess(img_path)
            cv2.imwrite(os.path.join(output_images_dir, img_file), processed)

            # Copy corresponding label
            label_file = img_file.replace(".jpg", ".txt")
            if os.path.exists(os.path.join(labels_dir, label_file)):
                shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_labels_dir, label_file))
    
main()
