import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
import cv2

class YoloDetectionDataset(Dataset):
    def __init__(self, image_paths, ann_paths, transform=None, img_size=640):
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx].split("\\")[-1]
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        if not self.ann_paths:
            return image, None, img_name
        
        w0, h0 = image.size
        data = torch.load(self.ann_paths[idx])

        if len(data["bboxes"]) == 0:
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0)
            targets = {"labels": labels, "bboxes": boxes}
        else:
            bboxes = data["bboxes"]
            labels = data["labels"]
            boxes_dict = []
            labels_dict = []

            for bbox, cls in zip(bboxes, labels):
                x, y, w, h = map(float, bbox)
                # scale bbox from original to resized size

                _, h1, w1 = image.shape

                x *= w1 / w0
                y *= h1 / h0
                w *= w1 / w0
                h *= h1 / h0
                boxes_dict.append([x, y, w, h])
                labels_dict.append(cls)
            
            boxes = torch.tensor(boxes_dict)
            labels = torch.tensor(labels_dict)

            targets = {"labels": labels, "bboxes": boxes}
        
        return image, targets, img_name

class SegmentationDataset(Dataset):
    def __init__(self, img_size, image_paths, ann_paths, image_transform=None, mask_transform=None):
        self.img_size = img_size
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.image_transform = image_transform  # 影像增強或tensor化
        self.mask_transform = mask_transform
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 讀圖片
        img_name = self.image_paths[idx].split("\\")[-1]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # 讀遮罩 (假設是多物件instance mask：每個物件有自己的二元mask，存在一張多通道圖或多張圖中)
        # 這裡舉例，假設 ann_paths[idx] 是多物件mask的 numpy array: shape [N, H, W]
        #masks_np = np.load(self.ann_paths[idx])  # 形狀 [N, H, W]
        #labels_np = np.load(self.ann_paths[idx].replace("masks", "labels"))  # shape [N]

        if self.image_transform:
            img = self.image_transform(img)

        # 讀取
        if not self.ann_paths:
            return img, None, img_name
        
        data = torch.load(self.ann_paths[idx])

        if len(data["masks"]) == 0:
            masks = torch.zeros(0, self.img_size[0], self.img_size[1])
            labels = torch.zeros(0)
            # masks = self.transform(masks)
            target = {"labels": labels, "masks": masks}
        else:
            masks = data["masks"]       # [N, H, W]
            labels = data["labels"]      # [N]
            mask_all = []

            for mask in masks:
                if self.mask_transform:
                    mask_all.append(self.mask_transform(mask.view(1, self.img_size[0], self.img_size[1])))
                else:
                    mask_all.append(mask)

            target = {"labels": labels, "masks": mask_all}

        return img, target, img_name

class SegmentationDataset_ema(Dataset):
    def __init__(self, img_size, resize_img_size, image_paths, ann_paths, image_transform=None, box_transform=None, mask_transform=None):
        self.img_size = img_size
        self.resize_img_size = resize_img_size
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.image_transform = image_transform  # 影像增強或tensor化
        self.mask_transform = mask_transform
        self.box_transform = box_transform
        # self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 讀圖片
        img_name = self.image_paths[idx].split("\\")[-1]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # 讀遮罩 (假設是多物件instance mask：每個物件有自己的二元mask，存在一張多通道圖或多張圖中)
        # 這裡舉例，假設 ann_paths[idx] 是多物件mask的 numpy array: shape [N, H, W]
        #masks_np = np.load(self.ann_paths[idx])  # 形狀 [N, H, W]
        #labels_np = np.load(self.ann_paths[idx].replace("masks", "labels"))  # shape [N]

        if self.image_transform:
            img = self.image_transform(img)

        # 讀取
        if not self.ann_paths:
            return img, None, img_name
        
        data = torch.load(self.ann_paths[idx])

        if len(data["masks"]) == 0:
            boxes = torch.zeros(0, 4)
            masks = torch.zeros(0, self.img_size[0], self.img_size[1]) if not self.resize_img_size else torch.zeros(0, self.resize_img_size[0], self.resize_img_size[1])
            labels = torch.zeros(0)
            # masks = self.transform(masks)
            target = {"labels": labels, "masks": masks, "boxes": boxes}
        else:
            boxes = data["bboxes"]       # [N, 4]
            masks = data["masks"]       # [N, H, W]
            labels = data["labels"]      # [N]
            box_all = []
            mask_all = []

            for box, mask in zip(boxes, masks):
                mask_pil = Image.fromarray(mask.numpy().astype("uint8"))
                if self.mask_transform:
                    mask_all.append(self.mask_transform(mask_pil))
                    # mask_all.append(self.mask_transform(mask.view(1, self.img_size[0], self.img_size[1])))
                else:
                    mask_all.append(mask_pil)
                
                #if self.box_transform:
                #    box_all.append(self.box_transform(box.view(1, 4)))
                #else:
                #    box_all.append(box)
                
                box_all.append(box)

            target = {"labels": labels, "masks": mask_all, "boxes": box_all}

        return img, target, img_name

class MultiModalSegDataset(Dataset):
    def __init__(self, img_size, image_paths, ann_paths=None, image_transform=None, label_map=None, resize_img_size=None, patch_size=16):
        self.img_size = img_size
        self.resize_img_size = resize_img_size
        self.image_paths = image_paths
        self.ann_paths = ann_paths
        self.patch_size = patch_size
        self.label_map = label_map if label_map else {"None"}
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.ann_paths[idx]

        # --- 1. Load image ---
        img = Image.open(img_path).convert("RGB")
        img = self.image_transform(img)  # [3, H, W]
        # img = img[0:1]  # use only 1 channel: [1, H, W]

        if not ann_path or not os.path.exists(ann_path):

            return {
                "name": img_path.split("\\")[-1],
                "image": img,
                "seg_mask": None,
                "rois": None,
                "mask_labels": None,
                "text_labels": None
            }

        # --- 2. Load annotations ---
        data = torch.load(ann_path)
        boxes = data.get("bboxes", torch.zeros(0, 4))   # [N, 4]

        if not self.resize_img_size:
            masks = data.get("masks", torch.zeros(0, *self.img_size))  # [N, H, W]
        else:
            masks = data.get("masks", torch.zeros(0, *self.resize_img_size))

        labels = data.get("labels", torch.zeros(0, dtype=torch.long))  # [N]

        # --- 3. Segmentation mask (optional) ---
        # 可以從所有 instance mask 合成一張 segmentation label map
        if not self.resize_img_size:
            seg_mask = torch.zeros(self.img_size, dtype=torch.long)  # shape: [H, W]
        else:
            seg_mask = torch.zeros(self.resize_img_size, dtype=torch.long)  # shape: [H, W]

        for i, mask in enumerate(masks):
            mask = torch.tensor(mask)
            
            # 保證 mask shape 和 seg_mask 一致
            if mask.shape != seg_mask.shape:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  # → [1, 1, H, W]
                    size=self.img_size if not self.resize_img_size else self.resize_img_size,
                    mode='nearest'
                ).squeeze()  # → [H, W]

            seg_mask[mask.bool()] = labels[i]

        # --- 4. RoIs ---
        rois = self.convert_boxes_to_rois(boxes, batch_idx=0)

        # --- 5. RoI mask labels ---
        masks_tensor = torch.tensor(masks, dtype=torch.float32)  # [N, H, W]

        mask_labels = torch.nn.functional.interpolate(
            masks_tensor.unsqueeze(1), size=(self.img_size[0], self.img_size[1]), mode='bilinear', align_corners=False
        ) if not self.resize_img_size else torch.nn.functional.interpolate(
            masks_tensor.unsqueeze(1), size=(self.resize_img_size[0], self.resize_img_size[1]), mode='bilinear', align_corners=False
        )

        # --- 6. Text labels (class names) ---
        if self.label_map:
            text_labels = [self.label_map[int(label.item())] for label in labels]
        else:
            text_labels = [f"class_{int(label.item())}" for label in labels]

        return {
            "name": img_path.split("\\")[-1],
            "image": img,                        # [3, 128, 128]
            "seg_mask": seg_mask,                # [128, 128]
            "rois": rois,                        # [N, 5]
            "mask_labels": mask_labels,          # [N, 1, 128, 128]
            "text_labels": text_labels           # List[str]
        }

    def convert_boxes_to_rois(self, boxes, batch_idx=0):
        rois = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            fx1, fy1, fx2, fy2 = x1 / self.patch_size, y1 / self.patch_size, x2 / self.patch_size, y2 / self.patch_size
            rois.append([batch_idx, fx1, fy1, fx2, fy2])
        return torch.tensor(rois, dtype=torch.float32)