"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from matplotlib.collections import QuadMesh
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import DataFrame
from string import ascii_uppercase
from distutils.version import LooseVersion
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import torchvision.ops as ops
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import seaborn as sns
import json
import os
from datetime import datetime

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

def compute_bbox_loss(bbox_deltas, proposals, t, num_classes=3, iou_thresh=0.5):
    # Step 1: IoU matching
    iou = box_iou(proposals, t["boxes"])  # [num_proposals, num_gt]
    max_iou, matched_gt_idx = iou.max(dim=1)  # [num_proposals]

    # Step 2: Positive samples
    positive_indices = torch.nonzero(max_iou >= iou_thresh).squeeze(1)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=bbox_deltas.device), positive_indices  # 沒有正樣本

    matched_gt = matched_gt_idx[positive_indices]  # [P]
    labels_pos = t["labels"][matched_gt]           # [P]

    # Step 3: Select bbox_deltas for matched class
    bbox_deltas = bbox_deltas.view(-1, num_classes, 4)  # [num_proposals, num_classes, 4]
    bbox_preds_pos = bbox_deltas[positive_indices, labels_pos]  # [P, 4]

    # Step 4: Ground-truth boxes
    gt_boxes_pos = t["boxes"][matched_gt]  # [P, 4]

    # Step 5: Loss
    loss = F.smooth_l1_loss(bbox_preds_pos, gt_boxes_pos)

    return loss, positive_indices

def encode_boxes(proposals, gt_boxes):
    # proposals: [N, 4], gt_boxes: [N, 4]
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gx = gt_boxes[:, 0] + 0.5 * gw
    gy = gt_boxes[:, 1] + 0.5 * gh

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)

    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(proposals, deltas):
    # proposals: [N, 4], deltas: [N, 4]
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    gx = dx * pw + px
    gy = dy * ph + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh

    return torch.stack([x1, y1, x2, y2], dim=1)

def dice_score(pred_mask, gt_mask, eps=1e-6):
    # [H, W] binary masks
    pred = pred_mask.float()
    gt = gt_mask.float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return (2. * inter + eps) / (union + eps)

def smooth_one_hot(labels, num_classes, smoothing=0.1):
    """
    labels: [N] - 整數 class id
    returns: [N, C] - soft label
    """
    confidence = 1.0 - smoothing
    label_shape = torch.Size((labels.size(0), num_classes))
    soft_labels = torch.full(label_shape, smoothing / (num_classes - 1)).to(labels.device)
    soft_labels.scatter_(1, labels.unsqueeze(1), confidence)
    return soft_labels

def apply_nms_to_proposals_with_index(proposals, scores=None, iou_threshold=0.5, top_k=100):
    """
    NMS + 回傳篩選後的 index 方便對應 logits / bbox。
    回傳: List of (filtered_boxes, keep_idx)
    """
    results = []
    for i, props in enumerate(proposals):
        if props.numel() == 0:
            results.append((props, torch.tensor([], dtype=torch.long)))
            continue

        score = scores[i] if scores is not None else torch.ones(props.size(0), device=props.device)
        keep_idx = ops.nms(props, score, iou_threshold)
        keep_idx = keep_idx[:top_k]
        results.append((props[keep_idx], keep_idx))
    return results

'''
def visualize_predictions(image, preds, class_names=None, conf_thresh=0.3, color=(0, 255, 0)):
    """
    在圖片上畫出 YOLO 模型預測的 bounding boxes。
    
    Args:
        image: 原始圖像 (H, W, 3)，格式為 np.uint8
        preds: Tensor[N, 6]，每一列為 [x1, y1, x2, y2, conf, class_id]
        class_names: List[str]，用來顯示類別名稱（若為 None，只顯示 id）
        conf_thresh: 信心分數過濾閾值
        color: 邊框顏色（預設綠色）
    Returns:
        畫好框的圖像 (np.uint8)
    """
    img = image.copy()

    for pred in preds:
        x1, y1, x2, y2, conf, cls_id = pred.tolist()
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{int(cls_id)}"
        if class_names:
            label = class_names[int(cls_id)]
        text = f"{label} {conf:.2f}"

        # Draw rectangle and label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
'''

# color = class_names[pred_labels[i]]

def visualize_predictions(
    img, pred_labels, pred_scores, gt_labels=None, 
    pred_boxes=None, gt_boxes=None, pred_masks=None, gt_masks=None, 
    class_names=None, class_color_map=None  # Optional: list like ["background", "car", "person", ...]
):
    """
    img: Tensor[C, H, W]
    pred_labels: Tensor[N_pred]
    pred_scores: Tensor[N_pred]
    pred_masks: Tensor[N_pred, H, W]
    label_names: Optional, list of strings
    """
    img = (img * 255).byte().cpu() if img.max() <= 1 else img.byte().cpu()
    img_np = img.permute(1, 2, 0).numpy()  # [H, W, C]

    # cv2.imshow("Original img", img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).copy()  # 原圖 (BGR)
    # cv2.imshow("Original img (convert COLOR_RGB2BGR)", img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    num_masks = pred_masks.shape[0]

    for i in range(num_masks):
        mask = pred_masks[i].cpu().numpy().astype(np.uint8)  # [H, W], binary
        if mask.sum() == 0:
            continue

        # 取得 class 顏色
        label_idx = int(pred_labels[i].item())
        color = class_color_map[label_idx]  # e.g., (0, 255, 0)
        colored_mask = np.zeros_like(img_np)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # 疊加透明 mask
        img_np = cv2.addWeighted(img_np, 1.0, colored_mask, 0.01, 0)

        # 計算 mask 中心點來放 label
        yx = np.argwhere(mask > 0)
        cy, cx = yx.mean(axis=0).astype(int)

        score = pred_scores[i].item()
        label_text = class_names[label_idx] if class_names and label_idx < len(class_names) else str(label_idx)
        text = f"{label_text} {score:.2f}"

        # 放文字
        cv2.putText(img_np, text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 2)

        # Optional: 畫外框輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_np, contours, -1, color, 2)

    # 顯示
    # cv2.imshow("Predicted Masks", img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_np

def convert_instance_to_class(mask, instance_to_class):
    class_mask = torch.zeros_like(mask)
    for instance_id_str, class_id in instance_to_class.items():
        instance_id = int(instance_id_str)
        class_mask[mask == instance_id] = class_id
    return class_mask

def collate_fn(batch):

    imgs = [item[0] for item in batch]      # list of images
    targets = [item[1] for item in batch]   # list of targets dict
    img_names = [item[2] for item in batch]

    # imgs 如果是PIL圖，需要先轉tensor或用 transforms
    imgs = torch.stack(imgs, dim=0)  # [B, C, H, W]
    return imgs, targets, img_names

def collate_fn_test(batch):

    imgs = [item[0] for item in batch]      # list of images
    targets = [item[1] for item in batch]   # list of targets dict
    
    # imgs 如果是PIL圖，需要先轉tensor或用 transforms
    imgs = torch.stack(imgs, dim=0)  # [B, C, H, W]
    return imgs, targets

def collate_fn_yolo(batch):
    images, targets, img_names = zip(*batch)
    images = torch.stack(images)  # [B, 3, H, W]
    return images, targets, img_names  # targets 保持為 list，因為 box 數量不一定

def collate_fn_moe(batch):
    """
    batch: List[dict], 每個 dict 來自 __getitem__
    """
    batched = {}

    # 簡單欄位直接 stack
    batched["name"] = [item["name"] for item in batch]
    batched["image"] = torch.stack([item["image"] for item in batch])
    batched["seg_mask"] = torch.stack([item["seg_mask"] for item in batch])

    # rois, mask_labels, text_labels 是可變長的 → 用 list 儲存
    batched["rois"] = [item["rois"] for item in batch]
    batched["mask_labels"] = [item["mask_labels"] for item in batch]
    batched["text_labels"] = [item["text_labels"] for item in batch]

    return batched

class PadToMultiple:
    def __init__(self, multiple=16):
        self.multiple = multiple

    def __call__(self, img):
        H, W = img.size
        pad_h = (self.multiple - H % self.multiple) % self.multiple
        pad_w = (self.multiple - W % self.multiple) % self.multiple
        # pad (left, top, right, bottom)
        padding = (0, 0, pad_w, pad_h)
        return F.pad(transforms.ToTensor()(img), padding)

def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"✅ Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return checkpoint

def decode_single_pred(raw_pred, anchors, stride, num_classes):
    B, _, H, W = raw_pred.shape
    na = anchors.shape[0]

    pred = raw_pred.view(B, na, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
    # pred: [B, na, H, W, 5+num_classes]

    # 建格點
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).to(raw_pred.device)  # [H, W, 2]

    # bbox中心點
    pred_xy = (pred[..., 0:2].sigmoid() + grid) * stride
    # bbox寬高
    pred_wh = (pred[..., 2:4].exp() * anchors.view(1, na, 1, 1, 2))

    boxes = torch.cat([pred_xy, pred_wh], dim=-1).view(B, -1, 4)  # [B, S, 4]
    obj = pred[..., 4].view(B, -1, 1)
    cls = pred[..., 5:].view(B, -1, num_classes)

    return {"boxes": boxes, "obj": obj, "cls": cls}

def decode_preds(pred, anchors, stride):
    """
    pred: [B, S, 5+C] (tx, ty, tw, th, obj, cls_logits...)
    anchors: [na, 2]
    stride: int

    return:
        boxes: [B, S, 4] (x_center, y_center, w, h) scaled to 原圖尺度
    """
    B, S, _ = pred.shape
    na = anchors.shape[0]
    grid_size = int((S // na) ** 0.5)

    device = pred.device

    # 預測解包
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]

    # 產生 grid
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    grid_x = grid_x.to(device).float()
    grid_y = grid_y.to(device).float()
    grid = torch.stack((grid_x, grid_y), 2).view(-1, 2)  # [grid_size^2, 2]

    # 擴展 grid 到所有 anchors
    grid = grid.repeat(na, 1)  # [S, 2]

    # anchors 重複匹配到 S 個格點
    anchor_tensor = anchors.repeat(grid_size * grid_size, 1).to(device)  # [S, 2]

    # 位置 (x,y) = sigmoid(tx, ty) + grid，再乘 stride 還原到原圖尺度
    x = (torch.sigmoid(tx) + grid[:, 0]) * stride
    y = (torch.sigmoid(ty) + grid[:, 1]) * stride

    # 寬高 w,h = anchor * exp(tw, th)
    w = anchor_tensor[:, 0] * torch.exp(tw)
    h = anchor_tensor[:, 1] * torch.exp(th)

    boxes = torch.stack([x, y, w, h], dim=-1)  # [B, S, 4] (center_x, center_y, w, h)

    return boxes

def draw_predictions(image, preds, class_names=None, conf_thresh=0.3):
    """
    Args:
        image (np.ndarray): BGR 圖片
        preds (Tensor): [N, 6] -> x1, y1, x2, y2, conf, class_id
        class_names (list): 類別名稱清單 (可選)
    """
    img = image.copy()
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds

    for x1, y1, x2, y2, conf, cls_id in preds:
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        text = f"{label} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img

def visualize_confusion_matrix(cm, class_names=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion matrix")
    plt.show()

def model_info(model, trainable_parameters=None):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_parameters)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        print(f"Trainable: {name} => {param.shape}")

def convert_boxes_to_rois(boxes, img_size, patch_size=16, batch_idx=0):
    """
    將原圖 bounding boxes 轉換為 RoIAlign 所需格式
    Args:
        boxes: List of boxes in original image size. Each box: (x1, y1, x2, y2)
        img_size: Tuple (H, W), e.g., (128, 128)
        patch_size: Size of patch embedding. Default = 16
        batch_idx: batch index (default 0)
    Returns:
        Tensor of shape (N, 5), each row is [batch_idx, x1, y1, x2, y2] in feature map units
    """
    rois = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Convert from original image scale → feature map scale
        fx1, fy1, fx2, fy2 = x1 / patch_size, y1 / patch_size, x2 / patch_size, y2 / patch_size
        rois.append([batch_idx, fx1, fy1, fx2, fy2])
    return torch.tensor(rois, dtype=torch.float)

def visualize_segmentation_with_confidence(
    seg_logits,                          # [1, C, H, W]
    output_path="output_overlay.png",
    class_names=None,                   # e.g., ['tumor', 'normal', 'inflammation']
    class_colors=None,                  # e.g., [(255, 0, 0), (0,255,0), (0,0,255)]
    confidence_threshold=0.5,
    background_class=0
):
    seg_logits = seg_logits[0]  # [C, H, W]
    C, H, W = seg_logits.shape
    probs = F.softmax(seg_logits, dim=0)  # [C, H, W]
    pred_mask = torch.argmax(probs, dim=0)  # [H, W]

    if class_colors is None:
        # 隨機顏色給每類
        np.random.seed(42)
        class_colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(C)]

    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    conf_scores = {}

    for cls_id in range(C):
        if cls_id == background_class:
            continue  # 不畫背景

        mask = (pred_mask == cls_id)
        class_prob = probs[cls_id]

        if mask.sum() > 0:
            confidence = class_prob[mask].mean().item()
            conf_scores[cls_id] = confidence

            if confidence < confidence_threshold:
                continue  # 跳過低信心類別

            # 畫顏色
            overlay[mask.cpu().numpy()] = class_colors[cls_id]

            # 浮水印
            ys, xs = torch.where(mask)
            cx, cy = int(xs.float().mean().item()), int(ys.float().mean().item())
            label = f"{class_names[cls_id] if class_names else 'Class '+str(cls_id)}: {confidence:.2f}"

            cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            conf_scores[cls_id] = 0.0

    cv2.imwrite(output_path, overlay)
    print(f"Saved overlay to {output_path}")
    return overlay, conf_scores

def overlay_mask(image_tensor, mask_tensor, confidence_map=None, alpha=0.5, class_color=None):
    """
    Overlay mask with optional confidence on the original image
    """
    image_np = image_tensor.squeeze().cpu().numpy()
    mask_np = mask_tensor.cpu().numpy()
    overlay = np.stack([image_np] * 3, axis=-1) * 255
    overlay = overlay.astype(np.uint8)

    color_mask = np.zeros_like(overlay)
    h, w = mask_np.shape

    for cls in range(1, len(class_color)):
        class_mask = mask_np == cls
        if confidence_map is not None:
            conf = confidence_map[cls]  # scalar for class
        else:
            conf = 1.0
        color = np.array(class_color[cls]) * conf
        for c in range(3):
            color_mask[..., c] += class_mask * color[c]

    blended = (1 - alpha) * overlay + alpha * color_mask
    blended = blended.astype(np.uint8)
    return Image.fromarray(blended)

def export_prediction(image_tensor, mask_pred, confidence, save_dir, 
                      patient_id="Unknown", image_id="Unknown", 
                      model_name="Model", model_version="v1.0", class_color_map=None, name=None, alpha=0.1):
    # os.makedirs(save_dir, exist_ok=True)

    # Convert and save overlay image
    # overlay_img = overlay_mask(image_tensor, mask_pred, confidence, class_color_map)

    image_np = image_tensor.squeeze().cpu().numpy()
    mask_np = mask_pred.cpu().numpy()
    overlay = np.array(image_np) * 255
    overlay = overlay.astype(np.uint8)
    overlay = overlay.transpose(1, 2, 0)

    color_mask = np.zeros_like(overlay)
    for cls in range(1, len(class_color_map)):
        class_mask = mask_np == cls

        if confidence is not None:
            conf = confidence[cls]  # scalar for class
            color = (np.array(class_color_map[cls]) * conf.item()).astype(np.uint8)
        else:
            color = np.array(class_color_map[cls]).astype(np.uint8)

        for c in range(3):
            color_mask[..., c] += class_mask * color[c]

    blended = (1 - alpha) * overlay + alpha * color_mask
    cv2.imwrite(os.path.join(save_dir, f"{name[0]}"), cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGB2BGR))  # OpenCV 用 BGR

    # Calculate bbox & area per class
    predictions = []
    for cls in range(1, len(class_color_map)):
        class_mask = (mask_pred == cls)
        if class_mask.sum() == 0:
            continue

        coords = torch.nonzero(class_mask)
        y1, x1 = coords.min(0).values.tolist()
        y2, x2 = coords.max(0).values.tolist()
        bbox = [x1, y1, x2, y2]
        area = int(class_mask.sum().item())
        centroid = coords.float().mean(0).tolist()[::-1]  # [x, y]

        predictions.append({
            "class_id": cls,
            "class_name": f"Class_{cls}",
            "confidence_score": float(confidence[cls]),
            "area_pixels": area,
            "bbox": bbox,
            "centroid": [round(c, 2) for c in centroid]
        })

    # Compose result
    result = {
        "patient_id": patient_id,
        "image_id": image_id,
        "inference_datetime": datetime.utcnow().isoformat() + "Z",
        "model": {
            "name": model_name,
            "version": model_version
        },
        "predictions": predictions,
        "summary": {
            "num_classes_detected": len(predictions),
            "total_area": sum(p["area_pixels"] for p in predictions)
        }
    }

    json_path = os.path.join(save_dir, f"{image_id}_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    return json_path

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps
    
def compute_matches_per_class(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks, class_id,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    
    gt_boxes=gt_boxes[np.where(gt_class_ids==class_id)]
    gt_masks=gt_masks[:,:,gt_class_ids==class_id]
    pred_boxes=pred_boxes[np.where(pred_class_ids==class_id)]
    pred_scores=pred_scores[np.where(pred_class_ids==class_id)]
    pred_masks=pred_masks[:,:,pred_class_ids==class_id]
    pred_class_ids=np.delete(pred_class_ids,np.where(pred_class_ids!=class_id))
    gt_class_ids=np.delete(gt_class_ids,np.where(gt_class_ids!=class_id))

    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    
    '''
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]
    '''
    
    #print('gt_masks:')
    #print(np.array(gt_masks).shape)
    #print('pred_masks:')
    #print(np.array(pred_masks).shape)

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break
    
    return gt_match, pred_match, overlaps
    
def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

#function 1 to be added to your utils.py
def get_iou(a, b, epsilon=1e-5):
    """ 
    Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.
    Args: 
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero
    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def plot_confusion_matrix_from_data(y_test, predictions, backbone_name, model_weight_file, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[36,36], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
        return the tp, fp and fn
    """

    #data
    if(not columns):
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:max(len(np.unique(y_test)),len(np.unique(predictions)))]]
    
    y_test = np.array(y_test) # np.append(np.array(y_test), 1)
    predictions = np.array(predictions) # np.append(np.array(predictions), 1)
    print('y_test:', y_test)
    print('predictions:', predictions)
    #confusion matrix 
    confm = confusion_matrix(y_test, predictions)
    num_classes = len(columns)
    
    #compute tp fn fp 
    
    fp=[0]*num_classes
    fn=[0]*num_classes
    tp=[0]*num_classes
    for i in range(confm.shape[0]):
        fp[i]+=np.sum(confm[i])-np.diag(confm)[i]
        fn[i]+=np.sum(np.transpose(confm)[i])-np.diag(confm)[i]
        for j in range(confm.shape[1]):
            if i==j:
                tp[i]+=confm[i][j]
    
    #plot
    print('confm:', confm)
    print('columns', columns)
    df_cm = DataFrame(confm, index=[columns], columns=[columns])

    #pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, 
    #    pred_val_axis=pred_val_axis, lw=lw, fmt=fmt)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g') # font size
    
    plt.title('Confusion matrix (per lesion)')
    plt.savefig('confusion_matrix_maskrcnn_per_lesion_' + str(backbone_name) + '_' + model_weight_file + '.png')
    #plt.show()
    plt.close() 

    return tp, fp, fn

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]
    
    #print('gt_masks:')
    #print(np.array(gt_masks).shape)
    #print('pred_masks:')
    #print(np.array(pred_masks).shape)

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    # ious = np.zeros([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break
    
    #print('pred_match:', pred_match)
    #print('gt_match:', gt_match)
    #print('pred_class_ids:', pred_class_ids)
    #print('gt_class_ids:', gt_class_ids)
    
    return gt_match, pred_match, overlaps

def compute_ap_pre_class(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks, class_id,
               iou_threshold=0.5, score_threshold=0.5):
    mAPs = np.zeros((len(gt_class_ids), 1))
    
    gt_match, pred_match, overlaps = compute_matches_per_class(
    gt_boxes, gt_class_ids, gt_masks,
    pred_boxes, pred_class_ids, pred_scores, pred_masks, class_id, 
    iou_threshold, score_threshold)
        
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
        
    for j in range(len(precisions) - 2, -1, -1):
        precisions[j] = np.maximum(precisions[j], precisions[j + 1])
        
    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    
    return mAP, precisions, recalls, overlaps
    
def gt_pred_lists(gt_class_ids, gt_bboxes, pred_class_ids, pred_bboxes, iou_tresh = 0.5):

    """ 
        Given a list of ground truth and predicted classes and their boxes, 
        this function associates the predicted classes to their gt classes using a given Iou (Iou>= 0.5 for example) and returns 
        two normalized lists of len = N containing the gt and predicted classes, 
        filling the non-predicted and miss-predicted classes by the background class (index 0).
        Args    :
            gt_class_ids   :    list of gt classes of size N1
            pred_class_ids :    list of predicted classes of size N2
            gt_bboxes      :    list of gt boxes [N1, (x1, y1, x2, y2)]
            pred_bboxes    :    list of pred boxes [N2, (x1, y1, x2, y2)]
            
        Returns : 
            gt             :    list of size N
            pred           :    list of size N 
    """

    #dict containing the state of each gt and predicted class (0 : not associated to any other class, 1 : associated to a class)
    gt_class_ids_ = {'state' : [0*i for i in range(len(gt_class_ids))], "gt_class_ids":list(gt_class_ids)}
    pred_class_ids_ = {'state' : [0*i for i in range(len(pred_class_ids))], "pred_class_ids":list(pred_class_ids)}

    #the two lists to be returned
    pred=[]
    gt=[]

    for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
        for j, pred_class in enumerate(pred_class_ids_['pred_class_ids']): 
            #check if the gt object is overlapping with a predicted object
            if get_iou(gt_bboxes[i], pred_bboxes[j])>=iou_tresh:
                #change the state of the gt and predicted class when an overlapping is found
                gt_class_ids_['state'][i] = 1
                pred_class_ids_['state'][j] = 1
                #gt.append(gt_class)
                #pred.append(pred_class)
                
                #chack if the overlapping objects are from the same class
                if (gt_class == pred_class):
                	gt.append(gt_class)
                	pred.append(pred_class)
                #if the overlapping objects are not from the same class 
                else : 
                    gt.append(gt_class)
                    pred.append(pred_class)
                
    #look for objects that are not predicted (gt objects that dont exists in pred objects)
    for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
        if gt_class_ids_['state'][i] == 0:
            gt.append(gt_class)
            pred.append(0)
            #match_id += 1
    #look for objects that are mispredicted (pred objects that dont exists in gt objects)
    for j, pred_class in enumerate(pred_class_ids_["pred_class_ids"]):
        if pred_class_ids_['state'][j] == 0:
            gt.append(0)
            pred.append(pred_class)
    return gt, pred
    

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    
    segmentation = resize(mask, (y2 - y1, x2 - x1))
    segmentation = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    
    full_segmentation = np.zeros(image_shape[:2], dtype=np.uint8)
    full_segmentation[y1:y2, x1:x2] = segmentation
    
    return full_mask, full_segmentation
    
############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP

def plot_confusion_matrix_from_data_per_image(y_test, predictions, backbone_name, model_weight_file, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[36,36], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
        return the tp, fp and fn
    """

    #data
    if(not columns):
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:max(len(np.unique(y_test)),len(np.unique(predictions)))]]
    
    y_test = np.array(y_test)
    predictions = np.array(predictions)
    print('y_test:', y_test)
    print('predictions:', predictions)
    #confusion matrix 
    confm = confusion_matrix(y_test, predictions)
    num_classes = len(columns)
    
    #compute tp fn fp 
    
    fp=[0]*num_classes
    fn=[0]*num_classes
    tp=[0]*num_classes
    for i in range(confm.shape[0]):
        fp[i]+=np.sum(confm[i])-np.diag(confm)[i]
        fn[i]+=np.sum(np.transpose(confm)[i])-np.diag(confm)[i]
        for j in range(confm.shape[1]):
            if i==j:
                tp[i]+=confm[i][j]
    
    #plot
    print('confm:', confm)
    print('columns', columns)
    df_cm = DataFrame(confm, index=[columns], columns=[columns])

    #pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, 
    #    pred_val_axis=pred_val_axis, lw=lw, fmt=fmt)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g') # font size
    
    plt.title('Confusion matrix (per image)')
    plt.savefig('confusion_matrix_maskrcnn_per_image_' + backbone_name + '_' + model_weight_file + '.png')
    #plt.show()
    plt.close() 

    return tp, fp, fn

def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)
