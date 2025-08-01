import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import imageio.v2 as imageio
import torch.nn.functional as F
import torchvision.ops as ops
from sklearn.metrics import precision_score, recall_score, accuracy_score, jaccard_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.ticker import MaxNLocator
from torchvision.ops import box_iou
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json

from model_training.loss import yolov9_loss_fn_with_anchors, detection_loss
from model_training.utils import draw_predictions, dice_score, visualize_predictions, \
                 visualize_confusion_matrix, \
                 export_prediction

def train_yolo(model, epoch, total_epochs, train_loader, optimizer, device, num_classes, progress_path=None):
    model.train()
    total_loss = 0
    training_status = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": 0,
        "total_steps": 0,
        "loss": 0,
        "obj_loss": 0,
        "cls_loss": 0,
        "box_loss": 0,
        "accuracy": 0,
        "mask_loss": 0,
        "seg_ce_loss": 0,
        "clip_seg_loss": 0,
        "finished": False,
        "cancel": False
    }

    for i, (imgs, targets) in enumerate(train_loader):

        if progress_path:
            with open(progress_path) as f:
                training_status = json.load(f)
                if training_status["cancel"] == True:
                    return total_loss / (i+1), training_status

        imgs = imgs.to(device)
        # targets = [t.to(device) for t in targets]
        preds = model(imgs)  # 假設多層 [P3, P4, P5] outputs
        
        loss, loss_dict = yolov9_loss_fn_with_anchors(
            preds=preds,
            targets=targets,         # List[Tensor], 每張圖: [N, 5]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        training_status["epoch"] = epoch+1
        training_status["step"] = i+1
        training_status["total_steps"] = len(train_loader)
        training_status["loss"] = loss.item()
        training_status["obj_loss"] = loss_dict['obj_loss']
        training_status["cls_loss"] = loss_dict['cls_loss']
        training_status["box_loss"] = loss_dict['box_loss']

        # if progress_path:
        #     with open(progress_path, "w") as f:
        #         json.dump(training_status, f, indent=2)

        print(f"Step {i+1} / {len(train_loader)} - loss: {loss.item()}, obj_loss: {loss_dict['obj_loss']:.4f}, cls_loss: {loss_dict['cls_loss']:.4f}, box_loss: {loss_dict['box_loss']:.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Train loss: {avg_loss:.4f}")
    return avg_loss, training_status

@torch.no_grad
def evaluate_yolo(model, dataloader, device, num_classes, iou_threshold=0.5, conf_thresh=0.3):
    model.eval()
    all_preds = []
    all_gts = []

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        outputs = model(imgs)  # 解碼後的格式
        B = imgs.size(0)

        for b in range(B):
            boxes_list, scores_list = [], []
            for level in outputs:
                boxes = level["boxes"][b]      # [S, 4]
                obj = level["obj"][b].squeeze()  # [S]
                cls = level["cls"][b].sigmoid()  # [S, C]
                score, cls_idx = cls.max(dim=1)  # 類別與分數
                conf = obj * score
                mask = conf > conf_thresh

                boxes = boxes[mask]
                conf = conf[mask]
                cls_idx = cls_idx[mask]

                if boxes.numel() > 0:
                    boxes_list.append(boxes)
                    scores_list.append(torch.stack([conf, cls_idx.float()], dim=1))

            if boxes_list:
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
                keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold)
                final_boxes = all_boxes[keep]
                final_classes = all_scores[keep][:, 1]
                preds = torch.cat([final_boxes, final_classes.unsqueeze(1)], dim=1)  # [x1, y1, x2, y2, cls]
            else:
                preds = torch.empty((0, 5))

            all_preds.append(preds.cpu())
            all_gts.append(targets[b])

    # 統計 per-class TP/FP/FN
    TP_per_class = defaultdict(int)
    FP_per_class = defaultdict(int)
    FN_per_class = defaultdict(int)
    all_gt_labels = []
    all_pred_labels = []

    for pred, gt in zip(all_preds, all_gts):
        pred_boxes = pred[:, :4]
        pred_labels = pred[:, 4] if pred.shape[0] > 0 else torch.empty(0)
        print("gt:", gt)
        gt_boxes = gt["bboxes"]
        gt_labels = gt["labels"]

        if gt_labels.numel() == 0:
            for cls in pred_labels.tolist():
                FP_per_class[int(cls)] += 1
            continue

        if pred_labels.numel() == 0:
            for cls in gt_labels.tolist():
                FN_per_class[int(cls)] += 1
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for i in range(ious.shape[0]):
            max_iou, j = ious[i].max(dim=0)
            if max_iou >= iou_threshold and j.item() not in matched_gt and pred_labels[i] == gt_labels[j]:
                TP_per_class[int(pred_labels[i])] += 1
                matched_gt.add(j.item())
                matched_pred.add(i)

        # 剩下的 pred 是 FP
        for i in range(len(pred_labels)):
            if i not in matched_pred:
                FP_per_class[int(pred_labels[i])] += 1

        # 剩下的 gt 是 FN
        for j in range(len(gt_labels)):
            if j not in matched_gt:
                FN_per_class[int(gt_labels[j])] += 1

        all_gt_labels.extend(gt_labels.cpu().tolist())
        all_pred_labels.extend(pred_labels.int().cpu().tolist())

    # 統計指標
    class_precision = {}
    class_recall = {}
    class_iou = {}

    for c in range(num_classes):
        TP = TP_per_class[c]
        FP = FP_per_class[c]
        FN = FN_per_class[c]
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)

        class_precision[c] = precision
        class_recall[c] = recall
        class_iou[c] = iou

    mean_precision = np.mean(list(class_precision.values()))
    mean_recall = np.mean(list(class_recall.values()))
    mean_iou = np.mean(list(class_iou.values()))

    print(f"[Eval] mPrec: {mean_precision:.4f}, mRecall: {mean_recall:.4f}, mIoU: {mean_iou:.4f}")

    # 繪製 Confusion Matrix
    cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # plt.show()

    # 條狀圖
    def plot_bar(metric_dict, title):
        keys = list(metric_dict.keys())
        vals = list(metric_dict.values())
        plt.figure(figsize=(10, 4))
        sns.barplot(x=keys, y=vals)
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        # plt.show()

    # plot_bar(class_precision, "Per-class Precision")
    # plot_bar(class_recall, "Per-class Recall")
    # plot_bar(class_iou, "Per-class IoU")

    return {"class_precision": class_precision, 
            "class_recall": class_recall, 
            "class_iou": class_iou, 
            "mean_precision": mean_precision, 
            "mean_recall": mean_recall, 
            "mean_iou": mean_iou}

@torch.no_grad
def test_yolo(model, dataloader, device, class_names, iou_threshold=0.5, conf_thresh=0.3, save_dir=None, show_image=True):
    
    model.eval()

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        outputs = model(imgs)  # 已解碼格式

        for b in range(B):
            boxes_list, scores_list = [], []
            for level in outputs:
                boxes = level["boxes"][b]      # [S, 5]
                obj = level["obj"][b].squeeze()  # [S]
                cls = level["cls"][b].sigmoid()  # [S, C]
                score, cls_idx = cls.max(dim=1)  # 取出最高機率的類別及其分數
                conf = obj * score               # obj * cls 確信度
                mask = conf > conf_thresh
                boxes = boxes[mask]
                conf = conf[mask]
                cls_idx = cls_idx[mask]
            if boxes_list:
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
                
                # NMS (可替換成更進階的 per-class NMS)
                keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold)
                final_boxes = all_boxes[keep]
                final_classes = all_scores[keep][:, 1]
                preds = torch.cat([final_boxes, final_classes.unsqueeze(1)], dim=1)  # [x1, y1, x2, y2, class]
            else:
                preds = torch.empty((0, 5))

        img_np = imgs[b].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, C]
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_with_boxes = draw_predictions(img_bgr, preds, class_names)

        if show_image:
            cv2.imshow("Prediction", img_with_boxes)
            cv2.waitKey(0)

@torch.no_grad
def inference_yolo(model, dataloader, device, class_names, iou_threshold=0.5, conf_thresh=0.3, save_dir=None, show_image=True):
    
    model.eval()
    filename_list = []

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        outputs = model(imgs)  # 已解碼格式

        for b in range(B):
            boxes_list, scores_list = [], []
            for level in outputs:
                boxes = level["boxes"][b]      # [S, 5]
                obj = level["obj"][b].squeeze()  # [S]
                cls = level["cls"][b].sigmoid()  # [S, C]
                score, cls_idx = cls.max(dim=1)  # 取出最高機率的類別及其分數
                conf = obj * score               # obj * cls 確信度
                mask = conf > conf_thresh
                boxes = boxes[mask]
                conf = conf[mask]
                cls_idx = cls_idx[mask]
            if boxes_list:
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
                
                # NMS (可替換成更進階的 per-class NMS)
                keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold)
                final_boxes = all_boxes[keep]
                final_classes = all_scores[keep][:, 1]
                preds = torch.cat([final_boxes, final_classes.unsqueeze(1)], dim=1)  # [x1, y1, x2, y2, class]
            else:
                preds = torch.empty((0, 5))

        img_np = imgs[b].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, C]
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_with_boxes = draw_predictions(img_bgr, preds, class_names)

        if show_image:
            cv2.imshow("Prediction", img_with_boxes)
            cv2.waitKey(0)
        
        if save_dir:
            filename = f"val_img{b}_{i}.png"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, img_with_boxes)
            filename_list.append(filename)
    
    return filename_list

def train_seg(model, epoch, total_epochs, train_loader, optimizer, scheduler, loss_fn, device, progress_path=None):
    
    model.train()
    total_loss = 0
    training_status = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": 0,
        "total_steps": 0,
        "loss": 0,
        "obj_loss": 0,
        "cls_loss": 0,
        "box_loss": 0,
        "accuracy": 0,
        "mask_loss": 0,
        "seg_ce_loss": 0,
        "clip_seg_loss": 0,
        "finished": False,
        "cancel": False
    }

    for i, (imgs, targets) in enumerate(train_loader):

        if progress_path:
            with open(progress_path) as f:
                training_status = json.load(f)
                if training_status["cancel"] == True:
                    return total_loss / (i+1), training_status

        imgs = imgs.to(device)
        outputs = model(imgs)

        optimizer.zero_grad()
        loss, loss_ce, loss_dice = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Dice: {dice_loss.item():.4f})")
        training_status["epoch"] = epoch+1
        training_status["step"] = i+1
        training_status["total_steps"] = len(train_loader)
        training_status["loss"] = loss.item()
        total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Step {i+1}/{len(train_loader)} - loss: {loss.item():.4f} | loss_ce: {loss_ce:.4f} | loss_dice: {loss_dice:.4f}")

        #if progress_path:
        #    with open(progress_path, "w") as f:
        #        json.dump(training_status, f, indent=2)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, training_status

@torch.no_grad()
def evaluate_seg(model, dataloader, loss_fn, device, num_classes, save_dir=None, class_names=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_gt = []
    all_pred = []

    for idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

        for i in range(len(imgs)):
            pred_mask = outputs["sem_seg"][i]  # [C, H, W]
            pred_label = torch.argmax(pred_mask, dim=0)  # [H, W]

            gt_masks_all = targets[i]["masks"]
            gt_labels_all = targets[i]["labels"]

            gt_masks = []
            gt_labels = []
            for gt_mask, gt_label in zip(gt_masks_all, gt_labels_all):
                gt_masks.append(gt_mask)
                gt_labels.append(gt_label)

            gt_masks = torch.stack(gt_masks, dim=0)
            gt_labels = torch.tensor(gt_labels)

            gt_seg_map = torch.zeros_like(pred_label)
            for inst_mask, cls in zip(gt_masks, gt_labels):
                gt_seg_map[inst_mask.bool()] = cls.item()

            gt_np = gt_seg_map.cpu().numpy().flatten()
            pred_np = pred_label.cpu().numpy().flatten()

            all_gt.append(gt_np)
            all_pred.append(pred_np)

            # 可選：儲存預測圖
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                imageio.imwrite(f"{save_dir}/eval_sample_{idx}_{i}.png", pred_label.cpu().numpy().astype("uint8"))

    avg_loss = total_loss / num_batches
    print(f"[Validation] Avg Loss: {avg_loss:.4f}")

    # === 評估指標統計 ===
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    precision_per_class = precision_score(all_gt, all_pred, labels=range(num_classes), average=None, zero_division=0)
    recall_per_class = recall_score(all_gt, all_pred, labels=range(num_classes), average=None, zero_division=0)
    iou_per_class = jaccard_score(all_gt, all_pred, labels=range(num_classes), average=None, zero_division=0)

    mean_precision = precision_per_class.mean()
    mean_recall = recall_per_class.mean()
    mean_iou = iou_per_class.mean()

    print(f"\nPer-class metrics:")
    for i in range(num_classes):
        cls_name = class_names[i] if class_names else f"Class {i}"
        print(f"{cls_name:<12}: Precision={precision_per_class[i]:.4f}  Recall={recall_per_class[i]:.4f}  IoU={iou_per_class[i]:.4f}")

    print("\nMacro Averages:")
    print(f"  Precision : {mean_precision:.4f}")
    print(f"  Recall    : {mean_recall:.4f}")
    print(f"  mIoU      : {mean_iou:.4f}")

    # === 混淆矩陣 ===
    cm = confusion_matrix(all_gt, all_pred, labels=range(num_classes))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", xticklabels=class_names or range(num_classes), yticklabels=class_names or range(num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "val_confusion_matrix.png"))
    plt.show()

    # === 條狀圖 ===
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_per_class, width, label="Precision")
    plt.bar(x, recall_per_class, width, label="Recall")
    plt.bar(x + width, iou_per_class, width, label="IoU")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Validation Per-Class Metrics")
    plt.xticks(x, class_names or [f"Class {i}" for i in x])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "val_per_class_metrics.png"))
    plt.show()

    return {
        "avg_loss": avg_loss,
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
        "per_class_iou": iou_per_class,
        "macro_precision": mean_precision,
        "macro_recall": mean_recall,
        "macro_iou": mean_iou,
        "confusion_matrix": cm,
    }

@torch.no_grad()
def test_seg(model, dataloader, device, num_classes, save_dir=None, class_names=None):
    model.eval()

    all_gt = []
    all_pred = []

    for idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        outputs = model(imgs)

        for i in range(len(imgs)):
            pred_mask = outputs["sem_seg"][i]  # [C, H, W]
            pred_label = torch.argmax(pred_mask, dim=0)  # [H, W]

            gt_masks_all = targets[i]["masks"]
            gt_labels_all = targets[i]["labels"]

            gt_masks = []
            gt_labels = []
            for gt_mask, gt_label in zip(gt_masks_all, gt_labels_all):
                gt_masks.append(gt_mask)
                gt_labels.append(gt_label)

            gt_masks = torch.stack(gt_masks, dim=0)
            gt_labels = torch.tensor(gt_labels)

            gt_seg_map = torch.zeros_like(pred_label)
            for inst_mask, cls in zip(gt_masks, gt_labels):
                gt_seg_map[inst_mask.bool()] = cls.item()

            gt_np = gt_seg_map.cpu().numpy().flatten()
            pred_np = pred_label.cpu().numpy().flatten()

            all_gt.append(gt_np)
            all_pred.append(pred_np)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                imageio.imwrite(f"{save_dir}/sample_{idx}_{i}.png", pred_label.cpu().numpy().astype("uint8"))

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    precision_per_class = precision_score(all_gt, all_pred, labels=range(num_classes), average=None, zero_division=0)
    recall_per_class = recall_score(all_gt, all_pred, labels=range(num_classes), average=None, zero_division=0)
    iou_per_class = jaccard_score(all_gt, all_pred, labels=range(num_classes), average=None, zero_division=0)

    mean_precision = precision_per_class.mean()
    mean_recall = recall_per_class.mean()
    mean_iou = iou_per_class.mean()

    print(f"[Test] Inference done.\n")
    print("Per-class metrics:")
    for i in range(num_classes):
        cls_name = class_names[i] if class_names else f"Class {i}"
        print(f"{cls_name:<12}: Precision={precision_per_class[i]:.4f}  Recall={recall_per_class[i]:.4f}  IoU={iou_per_class[i]:.4f}")

    print("\nMacro Averages:")
    print(f"  Precision : {mean_precision:.4f}")
    print(f"  Recall    : {mean_recall:.4f}")
    print(f"  mIoU      : {mean_iou:.4f}")

    # 混淆矩陣
    cm = confusion_matrix(all_gt, all_pred, labels=range(num_classes))

    # 繪製熱力圖
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names or range(num_classes), yticklabels=class_names or range(num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

    # 條狀圖: precision / recall / iou
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_per_class, width, label="Precision")
    plt.bar(x, recall_per_class, width, label="Recall")
    plt.bar(x + width, iou_per_class, width, label="IoU")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Per-Class Metrics")
    plt.xticks(x, class_names or [f"Class {i}" for i in x])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "per_class_metrics.png"))
    plt.show()

    return {
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
        "per_class_iou": iou_per_class,
        "macro_precision": mean_precision,
        "macro_recall": mean_recall,
        "macro_iou": mean_iou,
        "confusion_matrix": cm,
    }

@torch.no_grad()
def inference_seg(model, dataloader, device, class_color_map=None, original_dir=None, save_dir=None, progress_path=None):
    model.eval()
    filename_list = []
    inference_status = {
        "current_index": 1,
        "total_num_imgs": len(dataloader),
        "cancel": False,
        "success": False
    }

    for idx, (imgs, _) in enumerate(dataloader):

        if progress_path:
            with open(progress_path) as f:
                inference_status = json.load(f)
                if inference_status["cancel"] == True:
                    return filename_list
                
        inference_status["current_index"] = idx+1
        inference_status["total_num_imgs"] = len(dataloader)
        #if progress_path:
        #    with open(progress_path, "w") as f:
        #        json.dump(inference_status, f, indent=2)

        imgs = imgs.to(device)
        outputs = model(imgs)  # dict: pred_logits, pred_masks, sem_seg...

        for i in range(len(imgs)):
            pred_mask = outputs["sem_seg"][i]  # [C, H, W]
            pred_label = torch.argmax(pred_mask, dim=0).cpu().numpy()  # [H, W]

            # 可視化用的彩色 mask
            color_mask = np.zeros((pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8)

            for class_id in np.unique(pred_label):
                if class_color_map and class_id in class_color_map:
                    color = class_color_map[class_id]
                else:
                    color = (0, 255, 0)  # 預設顏色：綠色
                color_mask[pred_label == class_id] = color

            # 轉回 numpy 原圖（tensor [C,H,W] → [H,W,C]）
            img_np = imgs[i].detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))

            overlay = cv2.addWeighted(img_np, 1.0, color_mask, 0.4, 0)

            if save_dir is not None:
                filename = f"sample_{idx}_{i}.jpg"
                filename_list.append(filename)
                cv2.imwrite(os.path.join(original_dir, f"{filename}"), img_np)
                cv2.imwrite(os.path.join(save_dir, f"{filename}"), overlay)

    inference_status["success"] = True
    #if progress_path:
    #    with open(progress_path, "w") as f:
    #        json.dump(inference_status, f, indent=2)
                
    print(f"[Test] Inference done. {len(filename_list)} results.")
    return filename_list

def train_maskrcnn_ema(model, epoch, total_epochs, ema, train_loader, optimizer, device, batch_size, num_classes, writer=None, progress_path=None):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    total_mask_loss = 0
    total_obj_loss = 0
    training_status = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": 0,
        "total_steps": 0,
        "loss": 0,
        "obj_loss": 0,
        "cls_loss": 0,
        "box_loss": 0,
        "accuracy": 0,
        "mask_loss": 0,
        "seg_ce_loss": 0,
        "clip_seg_loss": 0,
        "finished": False,
        "cancel": False
    }

    for i, (imgs, targets) in enumerate(train_loader):

        if progress_path:
            with open(progress_path) as f:
                training_status = json.load(f)
                if training_status["cancel"] == True:
                    return total_loss / (i+1), training_status

        # proposals = generate_proposals(batch_size).to(device)
        # model output
        cls_logits, bbox_deltas, mask_logits, obj_logits, batch_indices, rpn_losses, final_proposals = model(imgs, targets)

        # loss 計算
        loss, loss_dict = detection_loss(
            cls_logits, bbox_deltas, mask_logits, obj_logits,
            targets, batch_indices=batch_indices,
            rpn_losses=rpn_losses,
            final_proposals=final_proposals,
            num_classes=num_classes
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新 EMA
        ema.update(model)
        total_loss += loss.item()
        total_cls_loss += loss_dict['cls_loss']
        total_bbox_loss += loss_dict['bbox_loss']
        total_mask_loss += loss_dict['mask_loss']
        total_obj_loss += loss_dict['obj_loss']

        training_status["epoch"] = epoch+1
        training_status["step"] = i+1
        training_status["total_steps"] = len(train_loader)
        training_status["loss"] = loss.item()
        training_status['cls_loss'] = loss_dict['cls_loss']
        training_status['box_loss'] = loss_dict['bbox_loss']
        training_status['mask_loss'] = loss_dict['mask_loss']
        training_status['obj_loss'] = loss_dict['obj_loss']

        # if progress_path:
        #    with open(progress_path, "w") as f:
        #        json.dump(training_status, f, indent=2)
        
        print(f"[Epoch {epoch+1}] Step {i+1}/{len(train_loader)} - cls_loss: {loss_dict['cls_loss']:.4f}, bbox_loss: {loss_dict['bbox_loss']:.4f}, mask_loss: {loss_dict['mask_loss']:.4f}, obj_loss: {loss_dict['obj_loss']:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_bbox_loss = total_bbox_loss / len(train_loader)
    avg_mask_loss = total_mask_loss / len(train_loader)
    avg_obj_loss = total_obj_loss / len(train_loader)
    print(f"[Train] Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # === SummaryWriter log ===
    if writer:
        writer.add_scalar("train/maskrcnn/total_loss", avg_loss, epoch)
        writer.add_scalar("train/maskrcnn/cls_loss", avg_cls_loss, epoch)
        writer.add_scalar("train/maskrcnn/bbox_loss", avg_bbox_loss, epoch)
        writer.add_scalar("train/maskrcnn/mask_loss", avg_mask_loss, epoch)
        writer.add_scalar("train/maskrcnn/obj_loss", avg_obj_loss, epoch)

    return avg_loss, training_status

def evaluate_maskrcnn_ema(model, dataloader, device, epoch, num_classes, class_names, class_color_map, save_dir=None, writer=None):
    model.eval()
    total_cls_loss = 0
    total_bbox_loss = 0
    total_mask_loss = 0
    total_obj_loss = 0
    total_loss = 0
    count = 0

    iou_list = []
    dice_list = []

    map_metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for b, (imgs, targets) in enumerate(dataloader):
            print("imgs:", imgs)
            print("targets:", targets)

            imgs = imgs.to(device)
            # forward
            cls_logits, bbox_deltas, mask_logits, obj_logits, batch_indices, rpn_losses, final_proposals = model(imgs, targets)

            # 計算 loss（僅評估用，不會 backward）
            loss, loss_dict = detection_loss(
                cls_logits, bbox_deltas, mask_logits, obj_logits,
                targets, batch_indices=batch_indices,
                rpn_losses=rpn_losses,
                final_proposals=final_proposals,
                num_classes=num_classes
            )

            total_loss += loss.item()
            total_cls_loss += loss_dict['cls_loss']
            total_bbox_loss += loss_dict['bbox_loss']
            total_mask_loss += loss_dict['mask_loss']
            total_obj_loss += loss_dict['obj_loss']
            count += 1

            # === Metrics for each image in batch
            for i in range(len(imgs)):
                pred_boxes = final_proposals[i]  # [N, 4]
                roi_mask = batch_indices == i
                cls_probs = F.softmax(cls_logits[roi_mask], dim=1)
                pred_scores, pred_labels = cls_probs.max(dim=1)

                gt_boxes = targets[i]["boxes"]
                gt_labels = targets[i]["labels"]

                if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                    # --- IoU ---
                    iou = box_iou(pred_boxes, gt_boxes)
                    iou_list.append(iou.max(dim=1)[0].mean().item())

                    # --- mAP ---
                    preds = [{
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels
                    }]
                    gts = [{
                        "boxes": gt_boxes,
                        "labels": gt_labels
                    }]
                    map_metric.update(preds, gts)

                # --- Dice score (only if mask exists) ---
                if mask_logits.size(0) > 0 and targets[i]["masks"].size(0) > 0:
                    pred_masks = mask_logits[roi_mask].argmax(dim=1)  # [N, H, W]
                    gt_masks = targets[i]["masks"]  # [M, H, W]
                    for j in range(min(pred_masks.shape[0], gt_masks.shape[0])):
                        dice = dice_score(pred_masks[j], gt_masks[j])
                        dice_list.append(dice.item())
                
                    if b < 1 and i < 2:
                        img = imgs[i].detach().cpu()
                        vis_img = visualize_predictions(
                            img, pred_labels, pred_scores, gt_labels, pred_boxes, gt_boxes,
                            pred_masks=pred_masks[:len(gt_boxes)],
                            gt_masks=gt_masks[:len(gt_boxes)],
                            class_names=class_names,
                            class_color_map=class_color_map
                        )
                        
                        if save_dir:
                            save_path = os.path.join(save_dir, f"val_img{b}_{i}.png")
                            cv2.imwrite(save_path, vis_img)

                        if writer:
                            img_tensor = torch.from_numpy(np.array(vis_img)).permute(2, 0, 1).float() / 255.0
                            writer.add_image(f"val/vis_img_{b}_{i}", img_tensor, global_step=epoch)

    avg_loss = total_loss / count
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    mean_dice = sum(dice_list) / len(dice_list) if dice_list else 0.0
    map_results = map_metric.compute()

    print(f"[Validation] loss: {avg_loss:.4f}, cls: {total_cls_loss/count:.4f}, bbox: {total_bbox_loss/count:.4f}, mask: {total_mask_loss/count:.4f}, obj: {total_obj_loss/count:.4f}")
    print(f" - Mean IoU: {mean_iou:.4f}")
    print(f" - Dice Score: {mean_dice:.4f}")
    print(f" - mAP@IoU=0.5: {map_results['map_50']:.4f}")
    print(f" - mAP@IoU=0.5:0.95: {map_results['map']:.4f}")

    if writer:
        writer.add_scalar("val/maskrcnn/loss", avg_loss, epoch)
        writer.add_scalar("val/maskrcnn/cls_loss", total_cls_loss / count, epoch)
        writer.add_scalar("val/maskrcnn/bbox_loss", total_bbox_loss / count, epoch)
        writer.add_scalar("val/maskrcnn/mask_loss", total_mask_loss / count, epoch)
        writer.add_scalar("val/maskrcnn/obj_loss", total_obj_loss / count, epoch)
        writer.add_scalar("val/maskrcnn/mIoU", mean_iou, epoch)
        writer.add_scalar("val/maskrcnn/Dice", mean_dice, epoch)
        writer.add_scalar("val/maskrcnn/mAP_50", map_results["map_50"], epoch)
        writer.add_scalar("val/maskrcnn/mAP", map_results["map"], epoch)
    
    return avg_loss

def inference_maskrcnn_ema(model, dataloader, device, class_names=None, class_color_map=None, visualize=False, vis_save_dir=None, conf_thresh=0.5):
    model.eval()
    filename_list = []

    with torch.no_grad():
        for b, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            cls_logits, bbox_deltas, mask_logits, obj_logits, batch_indices, _, final_proposals = model(imgs)

            for i in range(len(imgs)):
                roi_mask = batch_indices == i

                cls_probs = F.softmax(cls_logits[roi_mask], dim=1)
                pred_scores, pred_labels = cls_probs.max(dim=1)

                # 過濾低信心預測
                keep = pred_scores > conf_thresh
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                pred_masks_logits = mask_logits[roi_mask][keep]  # [N, num_classes, H, W]
                pred_masks = pred_masks_logits.argmax(dim=1)     # [N, H, W]

                # 把 tensor image 轉回 numpy 原圖
                img_np = imgs[i].detach().cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_np = np.transpose(img_np, (1, 2, 0))  # [C,H,W] → [H,W,C]

                overlay = img_np.copy()

                for j, mask in enumerate(pred_masks):
                    mask = mask.cpu().numpy().astype(np.uint8)
                    label_id = pred_labels[j].item()
                    class_name = class_names[label_id] if class_names else str(label_id)

                    # mask 顏色
                    color = class_color_map[class_name] if class_color_map and class_name in class_color_map else (0, 255, 0)

                    # 建立三通道 mask（用來 overlay）
                    color_mask = np.zeros_like(overlay, dtype=np.uint8)
                    color_mask[mask == 1] = color  # 只對該 mask 填色

                    # 疊加透明度
                    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)

                # 儲存結果
                filename = f"result_b{b}_img{i}.jpg"
                save_path = os.path.join(vis_save_dir, filename)
                cv2.imwrite(save_path, overlay)
                filename_list.append(filename)

    return filename_list

# ---------------------- Training Pipeline ---------------------- #
def train_segmentation_model_moe(model, epoch, total_epochs, train_loader, optimizer, criterion, scheduler, device, writer=None, progress_path=None):
    model.train()
    total_loss = 0.0
    total_seg_ce_loss = 0.0
    total_mask_loss = 0.0
    total_clip_loss = 0.0
    training_status = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": 0,
        "total_steps": 0,
        "loss": 0,
        "obj_loss": 0,
        "cls_loss": 0,
        "box_loss": 0,
        "accuracy": 0,
        "mask_loss": 0,
        "seg_ce_loss": 0,
        "clip_seg_loss": 0,
        "finished": False,
        "cancel": False
    }
    
    for i, batch in enumerate(train_loader):
        
        if progress_path:
            with open(progress_path) as f:
                training_status = json.load(f)
                if training_status["cancel"] == True:
                    return total_loss / (i+1), training_status

        image = batch["image"]               # [B, 1, 128, 128]
        seg_gt = batch["seg_mask"]         # [B, 128, 128]
        rois = batch["rois"]                 # [N, 5]
        text_labels = batch["text_labels"]   # List[str]
        mask_labels = batch["mask_labels"]   # [N, 1, 7, 7]

        # 假設 rois 是 list of Tensors, 每個 Tensor 是 [N_i, 4]
        if isinstance(rois, list):
            batched_rois = []
            for boxes in rois:
                if boxes.numel() == 0:
                    continue  # 跳過空的
                assert boxes.shape[1] == 5, "Each ROI must be in shape [N, 5]: [batch_idx, x1, y1, x2, y2]"
                batched_rois.append(boxes)
            if len(batched_rois) == 0:
                rois = None
            else:
                rois = torch.cat(batched_rois, dim=0)  # [K, 5]

        image = image.to(device)
        seg_feat, seg_logits, seg_out_sim, masks, similarity, text_embeds, report = model(image, rois=rois, text_labels=text_labels, mask_labels=mask_labels)

        optimizer.zero_grad()

        loss, loss_dict = criterion(
            seg_feat=seg_feat,
            seg_logits=seg_logits,
            seg_gt=seg_gt,
            text_embeds=text_embeds,
            mask_logits=masks,
            mask_labels=mask_labels,
            similarity=similarity,
            clip_targets=None
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        seg_ce_loss = loss_dict["seg_ce_loss"]
        mask_loss = loss_dict["mask_loss"]
        clip_loss = loss_dict["clip_seg_loss"]

        total_seg_ce_loss += seg_ce_loss
        total_mask_loss += mask_loss
        total_clip_loss += clip_loss

        print(f"[Epoch {epoch+1}] Step {i+1}/{len(train_loader)} "
            f"Loss: {loss.item():.4f} | "
            f"seg_ce_loss: {seg_ce_loss:.4f} | "
            f"mask_loss: {mask_loss:.4f} | "
            f"clip_seg_loss: {clip_loss:.4f}")

        training_status["epoch"] = epoch+1
        training_status["step"] = i+1
        training_status["total_steps"] = len(train_loader)
        training_status["loss"] = loss.item()
        training_status["seg_ce_loss"] = seg_ce_loss
        training_status["mask_loss"] = mask_loss
        training_status["clip_loss"] = clip_loss
        
        # if progress_path:
        #    with open(progress_path, "w") as f:
        #         json.dump(training_status, f, indent=2)

    avg_loss = total_loss / len(train_loader)
    avg_seg_ce_loss = total_seg_ce_loss / len(train_loader)
    avg_mask_loss = total_mask_loss / len(train_loader)
    avg_clip_loss = total_clip_loss / len(train_loader)

    print(f"Average Training Loss: {avg_loss:.4f}")

    # === SummaryWriter log ===
    if writer:
        writer.add_scalar("train/unetr_moe/avg_loss", avg_loss, epoch)
        writer.add_scalar("train/unetr_moe/avg_seg_ce_loss", avg_seg_ce_loss, epoch)
        writer.add_scalar("train/unetr_moe/avg_mask_loss", avg_mask_loss, epoch)
        writer.add_scalar("train/unetr_moe/avg_clip_loss", avg_clip_loss, epoch)

    return avg_loss, training_status

# ---------------------- Validation Pipeline ---------------------- #
def validate_segmentation_model_moe(model, val_loader, criterion, scheduler, device, epoch, writer=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"]               # [B, 1, 128, 128]
            seg_gt = batch["seg_mask"]         # [B, 128, 128]
            rois = batch["rois"]                 # [N, 5]
            text_labels = batch["text_labels"]   # List[str]
            mask_labels = batch["mask_labels"]   # [N, 1, 7, 7]

            # 假設 rois 是 list of Tensors, 每個 Tensor 是 [N_i, 4]
            if isinstance(rois, list):
                batched_rois = []
                for boxes in rois:
                    if boxes.numel() == 0:
                        continue  # 跳過空的
                    assert boxes.shape[1] == 5, "Each ROI must be in shape [N, 5]: [batch_idx, x1, y1, x2, y2]"
                    batched_rois.append(boxes)
                if len(batched_rois) == 0:
                    rois = None
                else:
                    rois = torch.cat(batched_rois, dim=0)  # [K, 5]

            # === Forward
            seg_feat, seg_logits, seg_out_sim, masks, similarity, text_embeds, report = model(
                image, rois=rois, text_labels=text_labels, mask_labels=mask_labels, generate_report=False
            )

            # === Loss 計算
            loss, loss_dict = criterion(
                seg_feat=seg_feat,
                seg_logits=seg_logits,
                seg_gt=seg_gt,
                text_embeds=text_embeds,
                mask_logits=masks,
                mask_labels=mask_labels,
                similarity=similarity,
                clip_targets=None
            )

            total_loss += loss.item()
            total_samples += image.size(0)

            # === 計算 accuracy / precision / IoU
            pred_mask = torch.argmax(seg_logits, dim=1)          # [B, H, W]
            pred_mask = pred_mask.cpu().numpy().flatten()
            gt_mask = seg_gt.cpu().numpy().flatten()

            all_preds.extend(pred_mask)
            all_gts.extend(gt_mask)

            correct = (pred_mask == gt_mask).sum()
            total_correct += correct

    # === 最終指標
    avg_loss = total_loss / len(val_loader)
    acc = total_correct / (len(all_gts) + 1e-6)
    precision = precision_score(all_gts, all_preds, average='macro', zero_division=0)
    iou = jaccard_score(all_gts, all_preds, average='macro', zero_division=0)

    if writer:
        writer.add_scalar("val/unetr_moe/avg_loss", avg_loss, epoch)
        writer.add_scalar("val/unetr_moe/acc", acc, epoch)
        writer.add_scalar("val/unetr_moe/precision", precision, epoch)
        writer.add_scalar("val/unetr_moe/iou", iou, epoch)

    print(f"\n[Validation] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | IoU: {iou:.4f}\n")
    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_precision": precision,
        "val_iou": iou
    }

def evaluate_segmentation_model_moe(model, val_loader, criterion, device, num_classes=3, class_names=None, visualize_cm=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            seg_gt = batch["seg_mask"].to(device)
            rois = batch.get("rois", None)
            text_labels = batch.get("text_labels", None)
            mask_labels = batch.get("mask_labels", None)

            seg_feat, seg_logits, masks, similarity, text_embeds, _ = model(
                image, rois=rois, text_labels=text_labels, mask_labels=mask_labels, generate_report=False
            )

            # ==== Loss ====
            loss, loss_dict = criterion(
                seg_feat=seg_feat,
                seg_logits=seg_logits,
                seg_gt=seg_gt,
                mask_logits=masks,
                mask_labels=mask_labels,
                text_embeds=text_embeds,
                similarity=similarity
            )

            total_loss += loss.item()
            total_samples += image.size(0)

            # ==== Metrices ====
            pred = torch.argmax(seg_logits, dim=1)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_gts.extend(seg_gt.cpu().numpy().flatten())

            correct = (pred == seg_gt).sum().item()
            total_correct += correct
        
    avg_loss = total_loss / len(val_loader)
    acc = total_correct / (len(all_gts) + 1e-6)
    precision = precision_score(all_gts, all_preds, average='macro', zero_division=0)
    iou = jaccard_score(all_gts, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_gts, all_preds, labels=list(range(num_classes)))

    print(f"[Evaluation] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Precision {precision:.4f} | IoU: {iou:.4f}")

    if visualize_cm:
        visualize_confusion_matrix(cm, class_names)
    
    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_precision": precision,
        "val_iou": iou,
        "confusion_matrix": cm
    }

def inference_segmentation_mode_moe(model, test_loader, device, class_color_map, class_names=None, save_dir=None):
    """
    class_color: list of RGB tuples (0-255) for each class
    """
    filename_list = []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)              # [B, C, H, W]
            batch_size = image.shape[0]

            seg_feat, seg_logits, *_ = model(image)        # seg_logits: [B, C, H, W]

            pred_masks = torch.argmax(seg_logits, dim=1)   # [B, H, W]
            conf_map = F.softmax(seg_logits, dim=1)        # [B, C, H, W]

            for i in range(batch_size):
                img_tensor = image[i]                      # [C, H, W]
                pred_mask = pred_masks[i]                 # [H, W]
                class_confidence = conf_map[i].mean(dim=(1, 2))  # [C] 平均 confidence per class

                pid = batch.get("pid", [f"unknown_{i}"])[i]
                image_id = batch.get("image_id", [f"{i}"])[i]
                filename_list.append(batch["name"])

                # export overlay and metadata json
                export_prediction(
                    image_tensor=img_tensor,
                    mask_pred=pred_mask,
                    confidence=class_confidence,
                    save_dir=save_dir,
                    patient_id=pid,
                    image_id=image_id,
                    class_color_map=class_color_map,
                    name=batch["name"]
                )

    print(f"Inference completed. Results saved to: {save_dir}")
    return filename_list

