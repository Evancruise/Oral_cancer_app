import json
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

ROOT_DIR = "./"
ANNOT_DIR = ROOT_DIR + "all/annotations_pt"
ANNOT_MASK_DIR = ROOT_DIR + "all/annotations_mask"
ANNOT_DIR_JSON = ROOT_DIR + "all/annotations_json"
ALL_IMG_DIR = ROOT_DIR + "all/images"

ANNOT_TRAIN_MASK_DIR = ROOT_DIR + "train/annotations_mask"
ANNOT_VAL_MASK_DIR = ROOT_DIR + "val/annotations_mask"
ANNOT_TEST_MASK_DIR = ROOT_DIR + "test/annotations_mask"

TRAIN_ANNOT_DIR = ROOT_DIR + "train/annotations"
VAL_ANNOT_DIR = ROOT_DIR + "val/annotations"
TEST_ANNOT_DIR = ROOT_DIR + "test/annotations"

TRAIN_IMG_DIR = ROOT_DIR + "train/images"
VAL_IMG_DIR = ROOT_DIR + "val/images"
TEST_IMG_DIR = ROOT_DIR + "test/images"

os.makedirs(ANNOT_DIR, exist_ok=True)
os.makedirs(ANNOT_MASK_DIR, exist_ok=True)
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(ALL_IMG_DIR, exist_ok=True)

os.makedirs(ANNOT_TRAIN_MASK_DIR, exist_ok=True)
os.makedirs(ANNOT_VAL_MASK_DIR, exist_ok=True)
os.makedirs(ANNOT_TEST_MASK_DIR, exist_ok=True)

os.makedirs(TRAIN_ANNOT_DIR, exist_ok=True)
os.makedirs(VAL_ANNOT_DIR, exist_ok=True)
os.makedirs(TEST_ANNOT_DIR, exist_ok=True)

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

class_dict = {"Green": 1, "Yellow": 2, "Red": 3}

colors = {
    "Backgroud": (0, 0, 0),
    "Green": (0, 255, 0),     
    "Yellow": (255, 255, 0),      
    "Red": (255, 0, 0),
}

def save_mask(json_path, output_shape=(512, 384), save_path_jpg=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    semantic_mask = np.zeros(output_shape, dtype=np.uint8)
    overlap_map = np.zeros(output_shape, dtype=np.uint8)  # 用來記錄是否有重疊
    mask = np.zeros(output_shape, dtype=np.uint8)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for poly in data["polygons"]:

        label = poly["label"]

        points = np.array(poly["points"], dtype=np.int32)

        cv2.fillPoly(mask, [points], color=255)

        mask[mask == 255] = 1
        current_area = (mask == 1)

        overlap_pixels = (semantic_mask > 0) & current_area
        semantic_mask[overlap_pixels] = 255

        semantic_mask[current_area] = class_dict[label]
        color_mask[semantic_mask == class_dict[label]] = colors[label]
        overlap_map[current_area] += 1

    num_overlap = np.sum(overlap_map > 1)
    if num_overlap > 0:
        print(f"[WARNING] {num_overlap} overlapping pixels detected in {save_path_jpg}")

    # print(save_path_jpg, np.unique(semantic_mask))
    # Image.fromarray(semantic_mask).save(save_path_jpg)
    Image.fromarray(color_mask).save(save_path_jpg)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪除檔案或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 刪除資料夾及其內容
            # print(f"✅ Removed: {file_path}")
        except Exception as e:
            print(f"❌ Failed to delete {file_path}. Reason: {e}")

def distribute_data(dataset, all_img_dir, target_img_dir):
    for prefix_name in tqdm(dataset):
        if os.path.exists(os.path.join(all_img_dir, prefix_name + ".png")):
            shutil.copy(os.path.join(all_img_dir, prefix_name + ".png"), os.path.join(target_img_dir, prefix_name + ".png"))
        else:
            shutil.copy(os.path.join(all_img_dir, prefix_name + ".jpg"), os.path.join(target_img_dir, prefix_name + ".jpg"))

def reshuffle_datasets(train_val_ratio=0.7, trainval_test_ratio=0.2):
    train_set, val_set, test_set = [], [], []   
    num_of_all_set = len(os.listdir(ANNOT_DIR_JSON))
    num_of_train_set = int(num_of_all_set * train_val_ratio)
    print("num_of_train_set:", num_of_train_set)
    num_of_val_set = int(num_of_all_set * (trainval_test_ratio + train_val_ratio))
    print("num_of_val_set:", num_of_val_set)
    num_of_test_set = int(num_of_all_set)
    print("num_of_test_set:", num_of_test_set)

    for i, json_file in enumerate(os.listdir(ANNOT_DIR_JSON)):
        if i < num_of_train_set:
            train_set.append(json_file.split('.')[0])
        elif i >= num_of_train_set and i < num_of_val_set:
            val_set.append(json_file.split('.')[0])
        else:
            test_set.append(json_file.split('.')[0])
    
    # clear_folder(TRAIN_ANNOT_DIR)
    clear_folder(TRAIN_IMG_DIR)
    # clear_folder(VAL_ANNOT_DIR)
    clear_folder(VAL_IMG_DIR)
    # clear_folder(TEST_ANNOT_DIR)
    clear_folder(TEST_IMG_DIR)

    print("distribute training set...")
    distribute_data(train_set, ALL_IMG_DIR, TRAIN_IMG_DIR)
    distribute_data(train_set, ANNOT_MASK_DIR, ANNOT_TRAIN_MASK_DIR)
    print("distribute validation set...")
    distribute_data(val_set, ALL_IMG_DIR, VAL_IMG_DIR)
    distribute_data(val_set, ANNOT_MASK_DIR, ANNOT_VAL_MASK_DIR)
    print("distribute testing set...")
    distribute_data(test_set, ALL_IMG_DIR, TEST_IMG_DIR)
    distribute_data(test_set, ANNOT_MASK_DIR, ANNOT_TEST_MASK_DIR)

if __name__ == "__main__":
    
    for json_file in os.listdir(ANNOT_DIR_JSON):
        if json_file.split('.')[-1] == "json":
            output_size = (384, 512)  # 高 x 寬
            save_mask(ANNOT_DIR_JSON + '/' + json_file, 
                       output_shape=output_size,
                       save_path_jpg=ANNOT_MASK_DIR + '/' + json_file.split('.')[0] + ".jpg")

    reshuffle_datasets(0.7, 0.25)

        
        
