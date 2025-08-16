import json
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import shutil

ROOT_DIR = "./dataset/"
ANNOT_DIR = ROOT_DIR + "all/annotations_pt"
ANNOT_MASK_DIR = ROOT_DIR + "all/annotations_mask"
ANNOT_DIR_JSON = ROOT_DIR + "all/annotations_json"
ALL_IMG_DIR = ROOT_DIR + "all/images"

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

os.makedirs(TRAIN_ANNOT_DIR, exist_ok=True)
os.makedirs(VAL_ANNOT_DIR, exist_ok=True)
os.makedirs(TEST_ANNOT_DIR, exist_ok=True)

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪除檔案或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 刪除資料夾及其內容
            print(f"✅ Removed: {file_path}")
        except Exception as e:
            print(f"❌ Failed to delete {file_path}. Reason: {e}")

def distribute_data(dataset, target_anno_dir, target_img_dir, format='.pt'):
    if format == ".pt":
        for json_file in dataset:
            shutil.copy(os.path.join(ANNOT_DIR, json_file + format), os.path.join(target_anno_dir, json_file + format))
            if os.path.exists(os.path.join(ALL_IMG_DIR, json_file + ".png")):
                shutil.copy(os.path.join(ALL_IMG_DIR, json_file + ".png"), os.path.join(target_img_dir, json_file + ".png"))
            else:
                shutil.copy(os.path.join(ALL_IMG_DIR, json_file + ".jpg"), os.path.join(target_img_dir, json_file + ".jpg"))
    else:
        for json_file in dataset:
            shutil.copy(os.path.join(ANNOT_DIR_JSON, json_file + format), os.path.join(target_anno_dir, json_file + format))
            if os.path.exists(os.path.join(ALL_IMG_DIR, json_file + ".png")):
                shutil.copy(os.path.join(ALL_IMG_DIR, json_file + ".png"), os.path.join(target_img_dir, json_file + ".png"))
            else:
                shutil.copy(os.path.join(ALL_IMG_DIR, json_file + ".jpg"), os.path.join(target_img_dir, json_file + ".jpg"))

def reshuffle_datasets(train_val_ratio=0.7, trainval_test_ratio=0.2):
    train_set, val_set, test_set = [], [], []   
    num_of_all_set = len(os.listdir(ANNOT_DIR_JSON))
    num_of_train_set = int(num_of_all_set * train_val_ratio)
    num_of_val_set = int(num_of_all_set * (trainval_test_ratio + train_val_ratio))
    num_of_test_set = int(num_of_all_set)

    for i, json_file in enumerate(os.listdir(ANNOT_DIR_JSON)):
        if i < num_of_train_set:
            train_set.append(json_file.split('.')[0])
        elif i >= num_of_train_set and i < num_of_val_set:
            val_set.append(json_file.split('.')[0])
        else:
            test_set.append(json_file.split('.')[0])
    
    clear_folder(TRAIN_ANNOT_DIR)
    clear_folder(TRAIN_IMG_DIR)
    clear_folder(VAL_ANNOT_DIR)
    clear_folder(VAL_IMG_DIR)
    clear_folder(TEST_ANNOT_DIR)
    clear_folder(TEST_IMG_DIR)

    distribute_data(train_set, TRAIN_ANNOT_DIR, TRAIN_IMG_DIR)
    distribute_data(val_set, VAL_ANNOT_DIR, VAL_IMG_DIR)
    distribute_data(test_set, TEST_ANNOT_DIR, TEST_IMG_DIR)

if __name__ == "__main__":
    reshuffle_datasets(0.7, 0.9)

        
        
