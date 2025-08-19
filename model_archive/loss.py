import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou
from model_archive.utils_func import encode_boxes, smooth_one_hot
# from utils import encode_boxes, smooth_one_hot
from scipy.optimize import linear_sum_assignment

# =======================
# object detection model
# =======================

def bbox_iou(box1, box2, eps=1e-7):
    """
    計算兩組 boxes 的 IoU
    box1: [N, 4] (x1,y1,x2,y2)
    box2: [M, 4] (x1,y1,x2,y2)
    return: [N, M] IoU矩陣
    """
    # 計算交集座標
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])  # [N, M]
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 計算面積
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)  # [N]
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)  # [M]

    union_area = area1[:, None] + area2 - inter_area + eps

    iou = inter_area / union_area
    return iou  # shape: [N, M]

def xywh_to_xyxy(boxes):
    """
    boxes: [N, 4], (cx, cy, w, h)
    return: [N, 4], (x1, y1, x2, y2)
    """
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal loss for binary classification
    inputs: sigmoid logits (already sigmoid applied)
    targets: 0 or 1
    """
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.where(targets==1, inputs, 1 - inputs)
    loss = BCE_loss * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_t = torch.where(targets==1, alpha * torch.ones_like(targets), (1 - alpha) * torch.ones_like(targets))
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def xywh_to_xyxy(x):
    # x: [..., 4] (cx, cy, w, h)
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def yolov9_loss_fn_with_anchors(preds, targets, iou_thresh=0.5):
    """
    preds: List[Dict] from model.forward (每層含 boxes, obj, cls, stride, anchors)
    targets: List[Tensor], 每張圖的 ground truth box: (x1, y1, x2, y2, class_id)
    """

    device = preds[0]["boxes"].device
    total_obj_loss = torch.tensor(0., device=device)
    total_cls_loss = torch.tensor(0., device=device)
    total_box_loss = torch.tensor(0., device=device)

    batch_size = len(targets)

    for b in range(batch_size):
        gt = targets[b]  # [N, 5]
        if gt["labels"].numel() == 0:
            # 沒有目標時 obj 預測都要是 0
            for pred in preds:
                total_obj_loss += F.binary_cross_entropy_with_logits(
                    pred["obj"][b], torch.zeros_like(pred["obj"][b])
                )
            continue

        for level_idx, pred in enumerate(preds):
            boxes_pred = pred["boxes"][b]   # [S, 5], xywh 格式
            obj_pred = pred["obj"][b].squeeze(-1)  # [S]
            cls_pred = pred["cls"][b]       # [S, C]
            stride = pred["stride"]
            anchors = pred["anchors"]       # [na, 2]

            S, C = cls_pred.shape
            num_anchors = anchors.shape[0]
            grid_size = int((S // num_anchors) ** 0.5)

            # 將 boxes_pred 從 (cx,cy,w,h) 轉成 (x1,y1,x2,y2)
            boxes_pred_xyxy = xywh_to_xyxy(boxes_pred)

            obj_target = torch.zeros_like(obj_pred)
            cls_target = torch.zeros_like(cls_pred)
            matched = torch.zeros(S, dtype=torch.bool, device=device)

            gt_boxes = gt["bboxes"][:, :4]
            gt_classes = gt["labels"].long()

            for i in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[i]
                gt_cls = gt_classes[i]

                gt_wh = gt_box[2:] - gt_box[:2]
                gt_xy = (gt_box[:2] + gt_box[2:]) / 2
                gt_scaled = gt_xy / stride

                ratios = gt_wh[None] / anchors
                ratios = torch.max(ratios, 1. / ratios).max(1)[0]
                anchor_mask = ratios < 4.0

                if anchor_mask.sum() == 0:
                    continue

                for a_idx in torch.where(anchor_mask)[0]:
                    cx, cy = int(gt_scaled[0]), int(gt_scaled[1])
                    if cx >= grid_size or cy >= grid_size:
                        continue
                    idx = a_idx * grid_size * grid_size + cy * grid_size + cx

                    if idx >= S:
                        continue
                    
                    iou = bbox_iou(boxes_pred_xyxy[idx:idx+1], gt_box.unsqueeze(0)).squeeze()
                    if iou > iou_thresh:
                        obj_target[idx] = 1.0
                        cls_target[idx, gt_cls] = 1.0
                        total_box_loss += 1. - iou
                        matched[idx] = True

            total_obj_loss += F.binary_cross_entropy_with_logits(obj_pred, obj_target)
            total_cls_loss += F.binary_cross_entropy_with_logits(cls_pred, cls_target)

    total_loss = total_obj_loss + total_cls_loss + total_box_loss

    return total_loss, {
        "obj_loss": total_obj_loss.item(),
        "cls_loss": total_cls_loss.item(),
        "box_loss": total_box_loss.item()
    }

# ==================
# Segmentation Model (DINOv2)
# ==================
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        # inputs: (B, 1, H, W) after sigmoid
        # targets: (B, 1, H, W) binary mask
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
        return 1 - dice.mean()

class Mask2FormerLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, outputs, targets):

        """
        outputs: dict with keys 'pred_logits' (B, Q, C+1), 'pred_masks' (B, Q, H, W)
        targets: list of dicts, each with keys 'labels' (N,), 'masks' (N, H, W) [binary mask per object]
        """
        pred_logits = outputs['pred_logits']  # [B, Q, C+1] (2, 100, 4)
        pred_masks = outputs['pred_masks']    # [B, Q, H, W] (2, 100, 512, 512)
        B, Q, C_plus1 = pred_logits.shape
        total_loss = 0
        total_loss_ce = 0
        total_loss_dice = 0

        for i in range(B):
            tgt_label_all = targets[i]['labels']           # [N]
            tgt_masks_all = targets[i]['masks']     # [N, H, W]

            N = len(tgt_label_all)
            if N == 0:
                # 若無目標，全部當no-object計算CE即可
                no_object_label = torch.full((Q,), C_plus1 - 1, dtype=torch.long, device=pred_logits.device)
                loss_ce = self.ce(pred_logits[i], no_object_label)
                total_loss += loss_ce
                continue
            elif N == 1:
                tgt_labels = torch.tensor(tgt_label_all)
                tgt_masks = torch.tensor(*tgt_masks_all)
            else:
                tgt_masks = []
                tgt_labels = []
                for tgt_mask, tgt_label in zip(tgt_masks_all, tgt_label_all):
                    tgt_masks.append(tgt_mask)
                    tgt_labels.append(tgt_label)
            
                tgt_masks = torch.cat(tgt_masks, dim=0) # [N, H, W] (3, 512, 512)
                
            tgt_labels = torch.tensor(tgt_labels)

            # 計算mask概率
            pred_prob = pred_masks[i].sigmoid()  # [Q, H, W] (100, 512, 512)

            # Flatten for cost計算
            pred_flat = pred_prob.view(Q, -1)  # [Q, HW] (100, 512*512)
            tgt_flat = tgt_masks.view(N, -1)   # [N, HW] (3, 512*512)         

            # 確保 target 是 0/1
            tgt_flat = (tgt_flat > 0).float()

            # 保證型別正確
            pred_flat = pred_flat.float()
            tgt_flat = tgt_flat.float()

            # mask cost (BCE)
            cost_mask = F.binary_cross_entropy(
                pred_flat.unsqueeze(1).expand(-1, N, -1),
                tgt_flat.unsqueeze(0).expand(Q, -1, -1),
                reduction='none'
            ).mean(-1)  # [Q, N] (100, 3)

            # class cost
            # 只用ground truth labels，不用no-object類別
            pred_cls_prob = pred_logits[i].softmax(dim=-1)  # [Q, C+1] (100, 4)
            tgt_labels_exp = tgt_labels.unsqueeze(0).expand(Q, -1)  # [Q, N]

            # cost_class = -pred_cls_prob[:, :-1].gather(1, tgt_labels_exp)  # 負的正確class概率，shape [Q, N]
            cost_class = -pred_cls_prob.gather(1, tgt_labels_exp)  # [Q, N]

            # 總成本矩陣
            cost_matrix = cost_mask + cost_class

            # Hungarian matching
            cost_matrix_cpu = cost_matrix.detach().cpu()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)

            # 匹配預測與目標
            matched_pred_logits = pred_logits[i][row_ind]          # [matched, C+1]
            matched_pred_masks = pred_prob[row_ind]                # [matched, H, W]
            matched_tgt_labels = tgt_labels[col_ind]               # [matched]
            matched_tgt_masks = tgt_masks[col_ind]                 # [matched, H, W]

            # 計算分類損失
            loss_ce = self.ce(matched_pred_logits, matched_tgt_labels)

            # 計算 mask loss：Dice + BCE
            loss_dice = self.dice(matched_pred_masks.unsqueeze(1), matched_tgt_masks.unsqueeze(1))

            matched_tgt_masks = (matched_tgt_masks > 0).float()

            # 保證型別正確
            matched_pred_masks = matched_pred_masks.float()
            matched_tgt_masks = matched_tgt_masks.float()

            loss_bce = F.binary_cross_entropy(
                matched_pred_masks,
                matched_tgt_masks,
                reduction='mean'
            )

            loss_mask = self.dice_weight * loss_dice + self.bce_weight * loss_bce

            total_loss += self.ce_weight * loss_ce + loss_mask
            total_loss_ce += loss_ce
            total_loss_dice += loss_dice

        return total_loss / B, total_loss_ce / B, total_loss_dice / B
    
# ==================
# Segmentation Model (Mask R-CNN + EMA + swin-transformer)
# ==================

def giou_loss(pred_boxes, target_boxes):
    """
    pred_boxes, target_boxes: [N, 4] in (x1, y1, x2, y2)
    """
    giou = generalized_box_iou(pred_boxes, target_boxes)
    loss = 1.0 - giou.diag()  # 只要對應 pair 的 diagonal
    return loss.mean()

def detection_loss(cls_logits, bbox_deltas, mask_logits, obj_logits,
                   targets, batch_indices=None, rpn_losses=None,
                   final_proposals=None, num_classes=3, iou_threshold=0.5):

    if batch_indices is None and final_proposals is not None:
        batch_indices = torch.cat([
            torch.full((len(p),), i, dtype=torch.long)
            for i, p in enumerate(final_proposals)
        ], dim=0).to(cls_logits.device)

    # 預處理 targets（跳過 None）
    processed_targets = []
    for t in targets:
        if t is None:
            processed_targets.append(None)
            continue
        if isinstance(t.get("labels", []), list):
            t["labels"] = torch.stack(t["labels"]) if len(t["labels"]) > 0 and isinstance(t["labels"][0], torch.Tensor) else torch.tensor(t["labels"])
        if isinstance(t.get("boxes", []), list):
            t["boxes"] = torch.stack(t["boxes"]).float() if len(t["boxes"]) > 0 else torch.empty((0, 4))
        if isinstance(t.get("masks", []), list):
            t["masks"] = torch.stack(t["masks"]).float() if len(t["masks"]) > 0 else torch.empty((0, 224, 224))
        processed_targets.append(t)

    total_cls_loss = 0
    total_bbox_loss = 0
    total_mask_loss = 0
    count = 0

    for i, t in enumerate(processed_targets):
        if t is None or t["labels"].numel() == 0:
            continue

        roi_mask = (batch_indices == i)
        if roi_mask.sum() == 0:
            continue

        cls_logit = cls_logits[roi_mask]
        bbox_delta = bbox_deltas[roi_mask]
        proposal = final_proposals[i]
        gt_box = t["boxes"]
        gt_label = t["labels"]

        iou = box_iou(proposal, gt_box)
        iou_max, matched_idx = iou.max(dim=1)
        keep = iou_max >= iou_threshold
        if keep.sum() == 0:
            continue

        matched_idx = matched_idx[keep]
        matched_label = gt_label[matched_idx]
        matched_gt_boxes = gt_box[matched_idx]
        proposal = proposal[keep]
        cls_logit = cls_logit[keep]
        bbox_delta = bbox_delta[keep]

        cls_loss = F.cross_entropy(cls_logit, matched_label)
        bbox_delta = bbox_delta.view(-1, num_classes, 4)
        bbox_deltas_class = bbox_delta[torch.arange(bbox_delta.size(0)), matched_label]
        bbox_target = encode_boxes(proposal, matched_gt_boxes)
        bbox_loss = F.smooth_l1_loss(bbox_deltas_class, bbox_target)

        if mask_logits.size(0) > 0 and t["masks"].size(0) > 0:
            masks_this = mask_logits[roi_mask]
            mask_pred = torch.stack([
                masks_this[j, matched_label[j]] for j in range(len(matched_label))
            ])
            mask_gt = t["masks"][matched_idx].float()
            if mask_gt.dim() == 4:
                mask_gt = mask_gt.squeeze(1)
            mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)
        else:
            mask_loss = torch.tensor(0.0, device=cls_logits.device)

        total_cls_loss += cls_loss
        total_bbox_loss += bbox_loss
        total_mask_loss += mask_loss
        count += 1

    # Objectness loss 支援 None target
    has_object = torch.tensor([
        1.0 if t is not None and t.get("labels", torch.tensor([])).numel() > 0 else 0.0
        for t in targets
    ], device=obj_logits.device)
    obj_loss = F.binary_cross_entropy_with_logits(obj_logits.squeeze(), has_object)

    if count > 0:
        total_loss = (total_cls_loss + total_bbox_loss + total_mask_loss) / count + obj_loss
    else:
        total_loss = obj_loss

    return total_loss, {
        "cls_loss": total_cls_loss / count if count > 0 else total_cls_loss,
        "bbox_loss": total_bbox_loss / count if count > 0 else total_bbox_loss,
        "mask_loss": total_mask_loss / count if count > 0 else total_mask_loss,
        "obj_loss": obj_loss
    }

# ---------------------- Joint CLIP + LLM Loss ---------------------- #
class CLIP_MultiTaskLoss(nn.Module):
    def __init__(self, lambda_seg_clip=1.0, lambda_mask=1.0, lambda_clip_roi=1.0):
        super().__init__()
        self.lambda_seg_clip = lambda_seg_clip
        self.lambda_mask = lambda_mask
        self.lambda_clip_roi = lambda_clip_roi

    def _compute_mask_loss(self, mask_logits, mask_labels, device=None):
        """
        Compute multi-class segmentation loss over RoI masks.

        Args:
            mask_logits: Tensor [N, C, H, W] or list of such tensors
            mask_labels: List[Tensor], each of shape [N_i, 1, H, W], float with 0./1. values (binary masks per class)
            device: (Optional) torch.device

        Returns:
            mask_loss: torch scalar loss
        """
        # Handle device
        if device is None:
            device = mask_logits.device if isinstance(mask_logits, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- [1] 過濾空 masks ---
        mask_labels = [m for m in mask_labels if m.numel() > 0]
        if len(mask_labels) == 0:
            return torch.tensor(0.0, device=device)

        # --- [2] 將 [N, 1, H, W] → [N, H, W]，並合併成 [N_total, H, W] ---
        mask_labels = [m.squeeze(1).long().to(device) for m in mask_labels]
        mask_labels = torch.cat(mask_labels, dim=0)  # [N_total, H, W]

        # --- [3] 確保 logits 是 tensor 且 shape 正確 ---
        if isinstance(mask_logits, list):
            mask_logits = [m for m in mask_logits if m.numel() > 0]
            mask_logits = torch.cat(mask_logits, dim=0)  # [N_total, C, H, W]

        mask_logits = mask_logits.to(device)

        # --- [4] Cross entropy over masks ---
        mask_loss = F.cross_entropy(mask_logits, mask_labels)

        return mask_loss

    def forward(self, seg_feat, seg_logits, seg_gt, text_embeds,
                mask_logits=None, mask_labels=None,
                similarity=None, clip_targets=None):
        
        # === [1] 傳統 segmentation loss ===
        B, C, H, W = seg_logits.shape
        seg_logits_flat = seg_logits.permute(0, 2, 3, 1).reshape(-1, C)
        seg_gt_flat = seg_gt.view(-1)
        seg_ce_loss = F.cross_entropy(seg_logits_flat, seg_gt_flat)

        # === [2] CLIP-style Segmentation loss ===
        seg_feat_flat = seg_feat.permute(0, 2, 3, 1).reshape(-1, seg_feat.shape[1])  # [BHW, 512]
        seg_feat_norm = F.normalize(seg_feat_flat, dim=-1)
        if text_embeds is not None:
            text_embeds_norm = F.normalize(text_embeds, dim=-1)                         # [T, 512]
            clip_logits = torch.matmul(seg_feat_norm, text_embeds_norm.T)              # [BHW, T]
            clip_seg_loss = F.cross_entropy(clip_logits, seg_gt_flat)
        else:
            clip_seg_loss = 0

        # === [3] Mask Head Loss ===
        mask_loss = self._compute_mask_loss(mask_logits, mask_labels, device=seg_feat.device)

        # === [4] CLIP ROI Similarity loss ===
        clip_roi_loss = torch.tensor(0.0, device=seg_feat.device)
        if similarity is not None and clip_targets is not None:
            clip_roi_loss = F.cross_entropy(similarity, clip_targets)
        else:
            clip_roi_loss = 0
            
        # === [5] Total ===
        total_loss = (
            seg_ce_loss +
            self.lambda_seg_clip * clip_seg_loss +
            self.lambda_mask * mask_loss +
            self.lambda_clip_roi * clip_roi_loss
        )

        return total_loss, {
            "seg_ce_loss": seg_ce_loss,
            "clip_seg_loss": clip_seg_loss,
            "mask_loss": mask_loss,
            "clip_roi_loss": clip_roi_loss
        }
