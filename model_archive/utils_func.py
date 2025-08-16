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
import shutil

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

