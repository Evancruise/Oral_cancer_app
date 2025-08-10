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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, OneCycleLR
import shutil
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

def compute_iou_matrix(pred_masks, gt_masks, threshold=0.5):
    import torch.nn.functional as F

    P = pred_masks.size(0)
    G = gt_masks.size(0)

    if P == 0 or G == 0:
        return torch.zeros((P, G), dtype=torch.float32)

    # 判斷維度，決定 resize size
    if pred_masks.dim() == 3:
        # pred_masks: (P, H, W)
        spatial_size = pred_masks.shape[1:]
    elif pred_masks.dim() == 4:
        # pred_masks: (P, C, H, W)
        spatial_size = pred_masks.shape[2:]
    else:
        raise ValueError(f"Unsupported pred_masks dims: {pred_masks.shape}")

    if gt_masks.dim() == 3:
        gt_spatial_size = gt_masks.shape[1:]
    elif gt_masks.dim() == 4:
        gt_spatial_size = gt_masks.shape[2:]
    else:
        raise ValueError(f"Unsupported gt_masks dims: {gt_masks.shape}")

    if spatial_size != gt_spatial_size:
        # 轉成 (N, C, H, W) 格式，並resize，再壓回原格式
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)  # (G, 1, H, W)
        gt_masks = F.interpolate(gt_masks, size=spatial_size, mode='nearest')
        if gt_masks.size(1) == 1:
            gt_masks = gt_masks.squeeze(1)  # 回 (G, H, W)

    # 下面繼續計算 IoU
    pred_bin = (pred_masks > threshold).float()
    gt_bin = (gt_masks > threshold).float()

    pred_flat = pred_bin.view(P, -1)
    gt_flat = gt_bin.view(G, -1)

    intersection = torch.matmul(pred_flat, gt_flat.t())
    pred_area = pred_flat.sum(dim=1).unsqueeze(1)
    gt_area = gt_flat.sum(dim=1).unsqueeze(0)
    union = pred_area + gt_area - intersection
    return intersection / torch.clamp(union, min=1e-6)

def best_matching(iou_matrix):
    """GT 與 Pred 的最佳匹配"""
    matched_preds = []
    matched_ious = []
    used_preds = set()
    for gt_idx in range(iou_matrix.shape[1]):
        ious_for_gt = iou_matrix[:, gt_idx]
        if ious_for_gt.numel() == 0:  # 無 pred
            matched_preds.append(None)
            matched_ious.append(0.0)
            continue
        sorted_iou, sorted_idx = torch.sort(ious_for_gt, descending=True)
        for pred_idx in sorted_idx:
            if pred_idx.item() not in used_preds:
                matched_preds.append(pred_idx.item())
                matched_ious.append(sorted_iou[0].item())
                used_preds.add(pred_idx.item())
                break
    return matched_preds, matched_ious

def nms(pred_masks, scores, iou_threshold=0.5):
    """基於 IoU 的 NMS（防呆版）"""
    keep = []

    # 如果沒有任何預測，直接回傳空
    if pred_masks.size(0) == 0 or scores.numel() == 0:
        return keep

    # 保證是 1 維索引
    idxs = scores.argsort(descending=True).view(-1)

    while idxs.numel() > 0:
        keep.append(idxs[0].item())
        if idxs.numel() == 1:
            break
        ref_mask = pred_masks[idxs[0]].unsqueeze(0)
        others = pred_masks[idxs[1:]]
        iou_matrix = compute_iou_matrix(others, ref_mask)
        ious = iou_matrix.squeeze(1)
        idxs = idxs[1:][ious <= iou_threshold]
    return keep

def compute_ap(recalls, precisions):
    """計算 AP (VOC 2010+ integral 方法)"""
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for i in range(precisions.size(0) - 1, 0, -1):
        precisions[i - 1] = torch.maximum(precisions[i - 1], precisions[i])
    idx = (recalls[1:] != recalls[:-1]).nonzero().squeeze()
    ap = torch.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return ap.item()

# compute_map(pred_masks_list, gt_masks_list, scores_list, pred_labels_list, gt_labels_list, iou_threshold=0.5)
def compute_map(all_preds, all_gts, all_scores, all_pred_labels, all_gt_labels, iou_threshold=0.5):
    """計算 mAP（多類別，batch-safe）"""
    if len(all_gts) == 0:
        return 0.0, []

    classes = torch.unique(torch.cat(all_gt_labels)) if len(all_gt_labels) > 0 else torch.tensor([])
    aps = []
    for cls in classes:
        preds_cls = []
        gts_cls = []
        scores_cls = []

        # 收集該類別的所有 batch
        for preds, gts, scores, plabels, glabels in zip(all_preds, all_gts, all_scores, all_pred_labels, all_gt_labels):
            pred_idx = (plabels == cls)
            gt_idx = (glabels == cls)
            preds_cls.append(preds[pred_idx])
            gts_cls.append(gts[gt_idx])
            scores_cls.append(scores[pred_idx])

        preds_cls = torch.cat(preds_cls) if preds_cls else torch.empty((0,))
        gts_cls = torch.cat(gts_cls) if gts_cls else torch.empty((0,))
        scores_cls = torch.cat(scores_cls) if scores_cls else torch.empty((0,))

        if preds_cls.size(0) == 0 or gts_cls.size(0) == 0:
            aps.append(0.0)
            continue

        order = scores_cls.argsort(descending=True)
        preds_cls = preds_cls[order]
        scores_cls = scores_cls[order]

        iou_matrix = compute_iou_matrix(preds_cls, gts_cls)
        detected_gt = set()
        tp = torch.zeros(len(preds_cls))
        fp = torch.zeros(len(preds_cls))

        for i in range(len(preds_cls)):
            max_iou, max_gt_idx = iou_matrix[i].max(dim=0)
            if max_iou >= iou_threshold and max_gt_idx.item() not in detected_gt:
                tp[i] = 1
                detected_gt.add(max_gt_idx.item())
            else:
                fp[i] = 1

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / len(gts_cls)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps.append(compute_ap(recalls, precisions))

    mAP = sum(aps) / len(aps) if aps else 0.0
    return mAP, aps

def segmentation_eval_batch(pred_masks_list, gt_masks_list, scores_list, pred_labels_list, gt_labels_list, threshold=0.5, nms_iou_threshold=0.5):
    """Batch-safe Segmentation 評估"""
    all_results = []
    for preds, gts, scores, plabels, glabels in zip(pred_masks_list, gt_masks_list, scores_list, pred_labels_list, gt_labels_list):
        iou_matrix = compute_iou_matrix(preds, gts, threshold)

        if iou_matrix.numel() == 0:
            miou = 0.0
            all_results.append({
                "iou_matrix": None,
                "miou": miou,
                "best_match": {"gt_to_pred": None, "ious": None}
                # "nms_keep": None
            })

        else:
            miou = iou_matrix.max(dim=1)[0].mean().item()

            best_match_preds, best_match_ious = best_matching(iou_matrix)
            # keep_indices = nms(preds, scores, nms_iou_threshold)

            all_results.append({
                "iou_matrix": iou_matrix,
                "miou": miou,
                "best_match": {"gt_to_pred": best_match_preds, "ious": best_match_ious}
                # "nms_keep": keep_indices
            })

    mAP, aps = compute_map(pred_masks_list, gt_masks_list, scores_list, pred_labels_list, gt_labels_list, iou_threshold=0.5)

    return {
        "per_image": all_results,
        "mAP": mAP,
        "AP_per_class": aps
    }

def masks_to_polygons(mask_batch, threshold=0.5):
    """
    mask_batch: [N, H, W] 二值 mask (float, 0~1)
    回傳 List[List[np.ndarray]]，每個 mask 轉成多個 polygons (ndarray Nx2)
    """
    polygons_list = []
    for mask in mask_batch:
        # 二值化
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # 找輪廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # 篩選小面積雜訊
                polygons.append(cnt.reshape(-1, 2))
        polygons_list.append(polygons)
    return polygons_list

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def plot_pr_curve(y_true, y_score, num_classes):
    # binarize true labels
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_score = np.array(y_score)
    y_true_bin = np.array(y_true_bin)

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"Class {i}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def build_rpn_targets(anchors, gt_boxes_batch, positive_iou_threshold=0.7, negative_iou_threshold=0.3, batch_size=256):
    """
    產生 RPN training targets，支援 batch input。
    
    Args:
        anchors: (N,4) Tensor，所有 anchors（通常不分 batch）
        gt_boxes_batch: List of tensor，每個 tensor shape (M_i, 4)，batch 裡各圖的 gt_boxes
        positive_iou_threshold: IoU 大於此為正樣本
        negative_iou_threshold: IoU 小於此為負樣本
        batch_size: 每張圖最多採樣 anchor 數量

    Returns:
        rpn_matches: (B, N) tensor，每個 anchor 標記為 1 (正樣本), -1 (負樣本), 0 (忽略)
        rpn_bbox_targets: (B, N, 4) tensor，正樣本對應的 bbox 偏移量，其他為 0
    """

    def bbox_iou(boxes1, boxes2):
        if boxes2.numel() == 0:
            # 如果 gt_boxes 是空，回傳全 0
            return torch.zeros((boxes1.size(0), 0), dtype=torch.float32)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 左上角 max
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角 min
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, None] + area2 - inter + 1e-6)
        return iou

    def encode_bbox(anchors, gt_boxes):
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

        dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
        dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
        dw = torch.log(gt_widths / anchor_widths)
        dh = torch.log(gt_heights / anchor_heights)

        return torch.stack([dx, dy, dw, dh], dim=1)

    B = len(gt_boxes_batch)
    N = anchors.size(0)

    rpn_matches = torch.zeros((B, N), dtype=torch.int8)  # 1, -1, 0
    rpn_bbox_targets = torch.zeros((B, N, 4), dtype=torch.float32)

    for b in range(B):
        gt_boxes = gt_boxes_batch[b]
        iou = bbox_iou(anchors, gt_boxes)  # [N, M]

        if gt_boxes.numel() == 0:
            # 沒有 gt，全部設為忽略 0
            continue

        max_iou, argmax_iou = iou.max(dim=1)  # 每個 anchor 找 max IoU 與對應 gt idx

        # 標記正負樣本
        rpn_matches[b][max_iou >= positive_iou_threshold] = 1
        rpn_matches[b][max_iou < negative_iou_threshold] = -1

        # 確保每個 gt_box 至少有一個 anchor 正樣本
        max_iou_per_gt, argmax_iou_per_gt = iou.max(dim=0)
        rpn_matches[b][argmax_iou_per_gt] = 1

        # 計算 bbox regression targets
        positive_indices = torch.where(rpn_matches[b] == 1)[0]
        if positive_indices.numel() > 0:
            assigned_gt = argmax_iou[positive_indices]
            rpn_bbox_targets[b, positive_indices] = encode_bbox(
                anchors[positive_indices], gt_boxes[assigned_gt]
            )

        # 以下可以根據需要做 subsample，保持正負樣本數量比例
        positive_count = int(batch_size * 0.5)
        positive_indices = positive_indices[torch.randperm(len(positive_indices))[:positive_count]]
        negative_indices = torch.where(rpn_matches[b] == -1)[0]
        negative_count = batch_size - len(positive_indices)
        negative_indices = negative_indices[torch.randperm(len(negative_indices))[:negative_count]]

        rpn_matches[b].zero_()
        rpn_matches[b][positive_indices] = 1
        rpn_matches[b][negative_indices] = -1

    return rpn_matches, rpn_bbox_targets

def rpn_bbox_loss_fn(rpn_bbox, rpn_match, rpn_bbox_targets):
    """
    Args:
        rpn_bbox:         [B, num_anchors, 4]
        rpn_match:        [B, num_anchors]
        rpn_bbox_targets: [B, num_anchors, 4]
    """
    pos_indices = torch.where(rpn_match == 1)
    pred = rpn_bbox[pos_indices]
    target = rpn_bbox_targets[pos_indices]

    loss = F.smooth_l1_loss(pred, target, reduction='mean')
    return loss

def rpn_class_loss_fn(rpn_class_logits, rpn_match):
    """
    Args:
        rpn_class_logits: [B, num_anchors, 2]
        rpn_match:        [B, num_anchors]  1=positive, -1=negative, 0=neutral
    """
    # 過濾掉中性 anchor（label == 0）
    pos_neg_indices = torch.where(rpn_match != 0)

    target = (rpn_match[pos_neg_indices] == 1).long()
    logits = rpn_class_logits[pos_neg_indices]

    loss = F.cross_entropy(logits, target)
    return loss

def mask_rcnn_loss_fn(mask_logits, mask_targets, mask_labels, original_size):
    """
    mask_logits: [B, N, num_classes, 28, 28] 或展平的 [total_rois, num_classes, 28, 28]
    mask_targets: [B, N, 28, 28] 0/1 mask
    mask_labels:  [B, N] (對應正樣本的 class_id, 負樣本填 -1)

    回傳:
        loss: scalar
    """

    # 如果是展平格式 [total_rois, num_classes, 28, 28]
    if mask_logits.dim() == 4:
        # 假設 mask_targets 與 mask_labels 已是 batch 格式
        B, N = mask_targets.shape[0], mask_targets.shape[1]
        mask_logits = mask_logits.view(B, N, mask_logits.size(1), original_size[0], original_size[1])

    # 找出正樣本 (class_id >= 0)
    positive_idx = mask_labels >= 0  # [B, N]

    if positive_idx.sum() == 0:
        return torch.tensor(0.0)

    # 取正樣本的 class_id 與 logits
    positive_classes = mask_labels[positive_idx]  # [num_pos]
    positive_logits = mask_logits[positive_idx, positive_classes, :, :]  # [num_pos, 28, 28]
    positive_targets = mask_targets[positive_idx]  # [num_pos, 28, 28]

    # Binary cross entropy
    loss = F.binary_cross_entropy_with_logits(positive_logits, positive_targets.float())

    return loss

def mask_loss_fn(pred_mask_logits, target_masks, target_labels):
    """
    pred_mask_logits: [B, num_classes, H, W]
    target_masks: [B, H, W]
    target_labels: [B]
    """
    B, num_classes, H, W = pred_mask_logits.shape
    pred_masks = pred_mask_logits[torch.arange(B), target_labels]  # [B, H, W]
    return F.binary_cross_entropy_with_logits(pred_masks, target_masks.float())

def cascade_rcnn_loss(class_logits, bbox_deltas, gt_labels, gt_deltas):
    """
    Args:
        class_logits: [B*N, num_classes]
        bbox_deltas:  [B*N, num_classes * 4]
        gt_labels:    [B, N]
        gt_deltas:    [B, N, 4]
    """
    B, N = gt_labels.shape

    # 把 gt_labels 展平成一維
    gt_labels_flat = gt_labels.view(-1)        # [B*N]

    # Classification Loss
    cls_loss = F.cross_entropy(class_logits, gt_labels_flat)

    # Regression Loss：只考慮正樣本
    pos_indices = torch.where(gt_labels_flat > 0)[0]
    if len(pos_indices) > 0:
        # bbox_deltas shape [B*N, num_classes * 4] → reshape成 [B*N, num_classes, 4]
        bbox_deltas = bbox_deltas.view(-1, class_logits.size(-1), 4)
        # 選出正樣本預測的 bbox delta
        pred_deltas = bbox_deltas[pos_indices, gt_labels_flat[pos_indices]]

        # 把 gt_deltas 也展平成 [B*N, 4]
        gt_deltas_flat = gt_deltas.view(-1, 4)
        reg_loss = F.smooth_l1_loss(pred_deltas, gt_deltas_flat[pos_indices])
    else:
        reg_loss = torch.tensor(0.0, device=class_logits.device)

    return cls_loss, reg_loss

def box_to_center(boxes):
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h

def encode_box(proposals, gt_boxes, eps=1e-6):
    """
    Transform GT boxes into regression deltas w.r.t. proposals
    proposals, gt_boxes: [N, 4]
    """
    px, py, pw, ph = box_to_center(proposals)
    gx, gy, gw, gh = box_to_center(gt_boxes)

    dx = (gx - px) / (pw + eps)
    dy = (gy - py) / (ph + eps)
    dw = torch.log((gw + eps) / (pw + eps))
    dh = torch.log((gh + eps) / (ph + eps))

    return torch.stack([dx, dy, dw, dh], dim=-1)

def compute_iou(boxes1, boxes2):
    """
    boxes1: [N, 4], boxes2: [M, 4]
    return: IoU [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None]

def apply_box_deltas(boxes, deltas):
    """
    對 anchors boxes 應用回歸偏移，計算 proposals boxes
    boxes: [N,4], anchor boxes格式(x1,y1,x2,y2)
    deltas: [N,4], 格式(dx,dy,dw,dh)
    return: proposals boxes [N,4]
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes

def delete_files_in_folder(dir=None):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader):
    
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
    elif optimizer_type == "adamW":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    else: # Default mode: adam
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
    if scheduler_mode == "cosineanneal":
        scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
    elif scheduler_mode == "stepLR":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_mode == "MultistepLR":
        scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    elif scheduler_mode == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_mode == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_mode == "onecycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=10)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
    
    return optimizer, scheduler

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

def decode_boxes(proposals, bbox_deltas, labels):
    """
    proposals: (N, 4) 原始候選框 [x1, y1, x2, y2]
    bbox_deltas: (N, num_classes, 4) 每個類別的偏移量
    labels: (N,) 對應的類別索引（要用哪一組偏移量）

    回傳:
        decoded_boxes: (N, 4) 解碼後的 [x1, y1, x2, y2]
    """
    # 取出對應 label 的 bbox 偏移量
    deltas = bbox_deltas[torch.arange(bbox_deltas.size(0)), labels]  # (N, 4)

    # anchor 轉換成中心點座標 + 寬高
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    # 防止除以 0
    widths = torch.clamp(widths, min=1e-6)
    heights = torch.clamp(heights, min=1e-6)

    # 解碼公式
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # 中心點轉回左上右下格式
    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    return decoded_boxes

def compute_ap(recall, precision):
    """計算單一 IoU threshold 的 AP"""
    # 在 recall 前後補點 (0,0) -> (1,0)
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    # 保證 precision 是 non-increasing
    for i in range(precision.size(0) - 1, 0, -1):
        precision[i-1] = torch.maximum(precision[i-1], precision[i])

    # 找出 recall 改變的點
    idx = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return ap.item()

def compute_miou(pred_masks, gt_masks, threshold=0.5):
    """計算 mIoU (多預測 vs 多 GT)"""
    pred_bin = (pred_masks > threshold).float()  # (P, H, W)
    gt_bin = (gt_masks > threshold).float()      # (G, H, W)

    ious = []
    for pred in pred_bin:
        # 計算 pred 與每個 gt 的 IoU
        intersection = (pred.unsqueeze(0) * gt_bin).sum(dim=(1, 2))
        union = pred.sum() + gt_bin.sum(dim=(1, 2)) - intersection
        iou = intersection / torch.clamp(union, min=1e-6)
        ious.append(iou.max())  # 取最大值 (匹配度最高的 GT)

    return torch.stack(ious).mean().item()

def best_match(iou_matrix):
    """
    iou_matrix: (P, G)
    回傳: 每個 GT 對應的最佳 Pred 索引，以及對應 IoU
    """
    matched_preds = []
    matched_ious = []

    used_preds = set()

    for gt_idx in range(iou_matrix.shape[1]):  # 遍歷每個 GT
        ious_for_gt = iou_matrix[:, gt_idx]  # 取該 GT 對所有 Pred 的 IoU
        # 找 IoU 最大且尚未被配對的 Pred
        sorted_iou, sorted_idx = torch.sort(ious_for_gt, descending=True)
        for pred_idx in sorted_idx:
            if pred_idx.item() not in used_preds:
                matched_preds.append(pred_idx.item())
                matched_ious.append(sorted_iou[0].item())
                used_preds.add(pred_idx.item())
                break

    return matched_preds, matched_ious

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

def apply_nms_to_proposals_with_index(proposals, scores=None, iou_threshold=0.5, top_k=5):
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

def delete_files_in_folder(dir=None):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

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

def collate_fn_cascade(batch):
    images = torch.stack([b['images'] for b in batch])
    image_meta = [b['image_meta'] for b in batch]
    gt_boxes = [b['gt_boxes'] for b in batch]
    gt_labels = [b['gt_labels'] for b in batch]
    gt_masks = [b['gt_masks'] for b in batch]

    # 統一轉 tensor，即使空的
    gt_labels = [torch.tensor(lbl, dtype=torch.long) if len(lbl) > 0 else torch.zeros((0,), dtype=torch.long)
                 for lbl in gt_labels]
    gt_boxes = [torch.tensor(box, dtype=torch.float32) if len(box) > 0 else torch.zeros((0, 4), dtype=torch.float32)
                for box in gt_boxes]
    
    return {
        'images': images,
        'image_meta': image_meta,
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels,
        'gt_masks': gt_masks
    }

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

