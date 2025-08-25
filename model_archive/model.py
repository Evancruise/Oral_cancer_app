import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch import Tensor
import openai
import os
import json
import torch.nn.functional as F
# from dinov2.models.vision_transformer import vit_large
import torchvision
from torchvision.models.detection.image_list import ImageList
from transformers import SwinModel, SwinConfig, CLIPTextModel, CLIPVisionModel, CLIPProcessor, CLIPTokenizer, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
from timm import create_model
import copy
from torchvision.ops import roi_align
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.rpn import AnchorGenerator
from model_archive.utils_func import apply_nms_to_proposals_with_index
# from peft import get_peft_model, LoraConfig, TaskType
# from peft.tuners.lora import LoraModel
# from monai.networks.blocks import UnetrBasicBlock
# from einops import rearrange
from PIL import Image
# import torchvision.transforms as T
import torchvision.ops as ops
from torchvision.ops import RoIAlign, MultiScaleRoIAlign, box_iou
import itertools

import torchvision.models as models
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size

from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple

class RPN(nn.Module):
    def __init__(self, in_channels, anchors_per_location):
        super().__init__()
        self.shared_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.rpn_class_logits = nn.Conv2d(512, anchors_per_location * 2, kernel_size=1)
        self.rpn_probs = nn.Softmax(dim=2)  # 在最後一個類別維度做softmax

        self.rpn_bbox = nn.Conv2d(512, anchors_per_location * 4, kernel_size=1)
        self.anchors_per_location = anchors_per_location

    def forward(self, x):
        batch_size = x.shape[0]
        shared = self.relu(self.shared_conv(x))

        rpn_class_logits = self.rpn_class_logits(shared)
        # 調整維度，轉成 (batch, num_anchors, 2)
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1).contiguous()
        rpn_class_logits = rpn_class_logits.view(batch_size, -1, 2)

        rpn_probs = self.rpn_probs(rpn_class_logits)

        rpn_bbox = self.rpn_bbox(shared)
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1).contiguous()
        rpn_bbox = rpn_bbox.view(batch_size, -1, 4)

        return rpn_class_logits, rpn_probs, rpn_bbox
    
class ResNetBackboneWithFPN(nn.Module):
    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # 利用ResNet層輸出C2,C3,C4,C5特徵圖 (layer1-layer4對應)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        # FPN 預設輸入channels
        in_channels_list = [
            256,  # layer1輸出channel
            512,  # layer2輸出channel
            1024, # layer3輸出channel
            2048, # layer4輸出channel
        ]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

    def forward(self, x):
        c1 = self.stem(x)       # 下采樣一次
        c2 = self.layer1(c1)    # stride 4
        c3 = self.layer2(c2)    # stride 8
        c4 = self.layer3(c3)    # stride 16
        c5 = self.layer4(c4)    # stride 32

        features = OrderedDict()
        features['0'] = c2
        features['1'] = c3
        features['2'] = c4
        features['3'] = c5

        fpn_outs = self.fpn(features)
        # fpn_outs 是dict，key是'0','1','2','3'，value是對應fpn feature map
        return fpn_outs

class AnchorGenerator:
    def __init__(self, sizes, ratios, feature_stride):
        self.sizes = sizes
        self.ratios = ratios
        self.stride = feature_stride

    def generate_anchors(self, feature_map_size, image_size):
        """
        feature_map_size: (H, W)
        image_size: (H_img, W_img)
        回傳 [N, 4] anchors: [x1, y1, x2, y2]
        """
        anchors = []
        fm_h, fm_w = feature_map_size
        for y in range(fm_h):
            for x in range(fm_w):
                center_x = x * self.stride + self.stride // 2
                center_y = y * self.stride + self.stride // 2
                for size in self.sizes:
                    for ratio in self.ratios:
                        w = size * (ratio ** 0.5)
                        h = size / (ratio ** 0.5)
                        x1 = center_x - w / 2
                        y1 = center_y - h / 2
                        x2 = center_x + w / 2
                        y2 = center_y + h / 2
                        anchors.append([x1, y1, x2, y2])
        return torch.tensor(anchors, dtype=torch.float32)
    
class MaskRCNNBackboneWithRPN(nn.Module):

    def __init__(self, top_down_pyramid_size=256, rpn_anchor_ratios=[0.5, 1, 2], rpn_anchor_scales=[32, 64, 128], fpn_downscale_ratio=16):
        super().__init__()
        self.backbone_fpn = ResNetBackboneWithFPN(out_channels=top_down_pyramid_size)
        self.rpn = RPN(in_channels=top_down_pyramid_size,
                       anchors_per_location=len(rpn_anchor_ratios))
        
        self.anchor_generator = AnchorGenerator(
            sizes=rpn_anchor_scales,
            ratios=rpn_anchor_ratios,
            feature_stride=fpn_downscale_ratio  # e.g. 16
        )
    
    def get_anchors(self, feature_map_shape, image_shape):
        """
        feature_map_shape: [B, C, H, W] 取其 H, W 作為輸出特徵圖大小
        image_shape: [B, 3, H, W] or (B, 3, H, W)
        回傳 anchors: [N, 4]
        """
        _, _, feat_h, feat_w = feature_map_shape
        anchors = self.anchor_generator.generate_anchors((feat_h, feat_w), image_shape[2:])
        return anchors

    def forward(self, images):
        """
        images: tensor (batch, C, H, W)
        返回:
          fpn_features: dict of feature maps (P2~P5)
          rpn_outputs: rpn_class_logits, rpn_probs, rpn_bbox
        """
        fpn_features = self.backbone_fpn(images)

        '''
        # RPN 輸入是每個 FPN 特徵圖，通常對每層都會做 RPN，然後合併結果
        rpn_class_logits_all = []
        rpn_probs_all = []
        rpn_bbox_all = []

        for level_name in ['0', '1', '2', '3']:
            feat = fpn_features[level_name]  # 每個feature map
            rpn_class_logits, rpn_probs, rpn_bbox = self.rpn(feat)
            rpn_class_logits_all.append(rpn_class_logits)
            rpn_probs_all.append(rpn_probs)
            rpn_bbox_all.append(rpn_bbox)

        # Concatenate來自不同尺度的結果，shape (batch, sum_anchors, ...)
        rpn_class_logits_all = torch.cat(rpn_class_logits_all, dim=1)
        rpn_probs_all = torch.cat(rpn_probs_all, dim=1)
        rpn_bbox_all = torch.cat(rpn_bbox_all, dim=1)

        return fpn_features, (rpn_class_logits_all, rpn_probs_all, rpn_bbox_all)
        '''
        
        return fpn_features

class ProposalLayer(nn.Module):
    def __init__(self, proposal_count=5, nms_threshold=0.7, min_size=16, image_shape=(512, 512)):
        """
        proposal_count: 產生的proposal數量上限 (如2000)
        nms_threshold: NMS閾值 (如0.7)
        min_size: proposals最小寬高限制，去除過小框
        image_shape: tuple (H, W) 圖片大小，將proposal clip在圖片內
        """
        super().__init__()
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.min_size = min_size
        self.image_shape = image_shape

        # (rpn_probs, rpn_bbox, image_shape)
    def forward(self, anchors, rpn_class_probs, rpn_bbox_deltas):
        """
        anchors: [num_anchors, 4] anchor boxes
        rpn_class_probs: [batch, num_anchors, 2] 前景分數和背景分數，取前景分數 [:, :, 1]
        rpn_bbox_deltas: [batch, num_anchors, 4]

        返回:
          proposals: [batch, proposal_count, 4]
        """

        batch_size = rpn_class_probs.shape[0]
        proposals_batch = []

        for i in range(batch_size):
            scores = rpn_class_probs[i, :, 1]  # 前景分數
            deltas = rpn_bbox_deltas[i]
            proposals = apply_box_deltas(anchors, deltas)

            # clip proposals 讓box stay在image邊界內
            # proposals[:, 0] = proposals[:, 0].clamp(min=0, max=self.image_shape[1] - 1)
            # proposals[:, 1] = proposals[:, 1].clamp(min=0, max=self.image_shape[0] - 1)
            # proposals[:, 2] = proposals[:, 2].clamp(min=0, max=self.image_shape[1] - 1)
            # proposals[:, 3] = proposals[:, 3].clamp(min=0, max=self.image_shape[0] - 1)

            proposals = torch.stack([
                proposals[:, 0].clamp(min=0, max=self.image_shape[1] - 1),
                proposals[:, 1].clamp(min=0, max=self.image_shape[0] - 1),
                proposals[:, 2].clamp(min=0, max=self.image_shape[1] - 1),
                proposals[:, 3].clamp(min=0, max=self.image_shape[0] - 1)
            ], dim=1)

            # 移除太小的box
            ws = proposals[:, 2] - proposals[:, 0]
            hs = proposals[:, 3] - proposals[:, 1]
            keep = (ws >= self.min_size) & (hs >= self.min_size)
            proposals = proposals[keep]
            scores = scores[keep]

            # 根據前景分數排序，取前 pre_nms_topN 個
            pre_nms_topN = min(6000, proposals.shape[0])  # 一般會設6000
            scores, order = scores.sort(descending=True)
            scores = scores[:pre_nms_topN]
            proposals = proposals[order[:pre_nms_topN]]

            # NMS過濾
            keep_idx = ops.nms(proposals, scores, self.nms_threshold)
            keep_idx = keep_idx[:self.proposal_count]

            proposals = proposals[keep_idx]

            # 補足proposal數量，不足則用0補齊
            if proposals.shape[0] < self.proposal_count:
                padding = torch.zeros((self.proposal_count - proposals.shape[0], 4), device=proposals.device)
                proposals = torch.cat([proposals, padding], dim=0)

            proposals_batch.append(proposals.unsqueeze(0))

        proposals_batch = torch.cat(proposals_batch, dim=0)
        return proposals_batch  # shape [batch, proposal_count, 4]

class CascadeHeadStage(nn.Module):
    def __init__(self, in_channels, pool_size, num_classes, fc_dim):
        super().__init__()
        self.pool_size = pool_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

        self.class_logits = nn.Linear(fc_dim, num_classes)
        self.bbox_pred = nn.Linear(fc_dim, num_classes * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, roi_feat):
        x = roi_feat.flatten(start_dim=1)  # shape: (N, C*pool_size*pool_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        class_logits = self.class_logits(x)
        bbox_deltas = self.bbox_pred(x)

        return class_logits, bbox_deltas

class CascadeRCNNHead(nn.Module):
    def __init__(self, pool_size, num_classes=3, fc_dim=1024, stages=3):
        super().__init__()
        self.stages = stages
        # self.roi_align = RoIAlign(output_size=pool_size, spatial_scale=1/16.0, sampling_ratio=2, aligned=True)
        self.multi_roi_align_fpn = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # 對應 FPN 層
            output_size=7,
            sampling_ratio=2
        )
        
        self.heads = nn.ModuleList([
            CascadeHeadStage(in_channels=256, pool_size=pool_size, num_classes=num_classes, fc_dim=fc_dim)
            for _ in range(stages)
        ])

    def ensure_boxes_list(self, boxes, batch_size=None, device=None, dtype=torch.float32):
        """
        將 boxes 轉成 MultiScaleRoIAlign 所需的 List[Tensor] 格式。
        支援輸入：
        - List[Tensor] (直接回傳)
        - Tensor[K,5] (batch_idx + x1,y1,x2,y2)
        - Tensor[K,4] (視為 single-image，需提供 batch_size=1 或手動包成 list)
        回傳: list_of_boxes, batch_size
        """
        # already a list-like of tensors
        if isinstance(boxes, (list, tuple)):
            return [b for b in boxes], (batch_size if batch_size is not None else len(boxes))

        # Tensor case
        if isinstance(boxes, torch.Tensor):
            if boxes.dim() == 2 and boxes.size(1) == 5:
                # format [K,5] -> split by batch_idx
                if batch_size is None:
                    batch_size = int(boxes[:,0].max().item()) + 1 if boxes.numel() > 0 else 1
                boxes_list = []
                for i in range(batch_size):
                    m = boxes[:,0] == i
                    b_i = boxes[m, 1:]
                    boxes_list.append(b_i)
                return boxes_list, batch_size

            elif boxes.dim() == 2 and boxes.size(1) == 4:
                # treat as single-image boxes
                if batch_size is None or batch_size == 1:
                    return [boxes], 1
                else:
                    raise ValueError("Received Tensor[N,4] but batch_size > 1. Provide List[Tensor] per image or Tensor[K,5] with batch indices.")

        raise TypeError("Unsupported boxes format. Provide List[Tensor] or Tensor[K,5] or Tensor[N,4].")

    def forward(self, proposals, feature_maps, image_meta=None):
        """
        proposals: [B, N, 4], bbox 格式是 (x1, y1, x2, y2)
        feature_maps: FPN P2~P5 合併後的 feature map [B, C, H, W]
        """

        # 將 proposals 攤平為 list，每個 box 加上 batch index
        batch_size, num_proposals, _ = proposals.shape
        roi_feats_all = []
        refined_proposals = proposals

        image_shapes = [meta['original_shape'] for meta in image_meta]

        all_class_logits, all_bbox_deltas, all_refined = [], [], []

        for stage, head in enumerate(self.heads):
            rois_with_batch_idx = []
            for b in range(batch_size):
                boxes = refined_proposals[b]
                batch_inds = torch.full((boxes.shape[0], 1), b, dtype=torch.float32)
                rois_with_batch_idx.append(torch.cat([batch_inds, boxes], dim=1))  # shape: [N, 5]
            rois = torch.cat(rois_with_batch_idx, dim=0)  # [B*N, 5]

            boxes_list, bs = self.ensure_boxes_list(rois, batch_size=batch_size)

            pooled_feats = self.multi_roi_align_fpn(feature_maps, boxes_list, image_shapes=image_shapes)

            # Pass through head
            class_logits, bbox_deltas = head(pooled_feats)

            # 儲存結果
            all_class_logits.append(class_logits)
            all_bbox_deltas.append(bbox_deltas)

            # Refine boxes
            class_ids = class_logits.argmax(dim=1)
            box_deltas = bbox_deltas.view(-1, self.heads[stage].num_classes, 4)
            selected_deltas = box_deltas[torch.arange(box_deltas.shape[0]), class_ids]

            # 重新計算 proposal boxes
            refined = apply_box_deltas(rois[:, 1:], selected_deltas)
            refined = torch.clamp(refined, min=0.0)  # 可再加 clip 到 image_shapes
            refined_proposals = refined.view(batch_size, num_proposals, 4)
            all_refined.append(refined_proposals)

        return all_class_logits, all_bbox_deltas, all_refined

class DetectionTargetLayer(nn.Module):
    def __init__(self, stage_iou_thresh, mask_size=28):
        super().__init__()
        self.iou_threshold = stage_iou_thresh  # e.g. 0.5, 0.6, 0.7
        self.mask_size = mask_size

    def forward(self, proposals, gt_boxes, gt_labels, gt_masks):
        batch_size, num_proposals, _ = proposals.shape
        device = proposals.device

        mask_h, mask_w = self.mask_size[0], self.mask_size[1]
        mask_dtype = gt_masks[0].dtype if len(gt_masks) > 0 and gt_masks[0].numel() > 0 else torch.float32

        matched_labels = torch.zeros((batch_size, num_proposals), dtype=torch.long, device=device)
        matched_deltas = torch.zeros((batch_size, num_proposals, 4), dtype=torch.float32, device=device)
        matched_proposals = proposals.clone()
        matched_gt_boxes = torch.zeros((batch_size, num_proposals, 4), dtype=torch.float32, device=device)
        matched_gt_masks = torch.zeros((batch_size, num_proposals, mask_h, mask_w), dtype=mask_dtype, device=device)

        for b in range(batch_size):
            props = proposals[b]         # [N, 4]
            gt_box = gt_boxes[b]         # [M, 4]
            gt_cls = gt_labels[b]        # [M]

            if len(gt_masks) <= b or gt_masks[b].numel() == 0:
                continue

            gt_mask = gt_masks[b]        # [M, H, W]

            ious = box_iou(props, gt_box)  # [N, M]
            max_iou, max_ids = ious.max(dim=1)

            positive_idx = torch.nonzero(max_iou >= self.iou_threshold).squeeze(1)
            if positive_idx.numel() == 0:
                continue

            matched_labels[b][positive_idx] = gt_cls[max_ids[positive_idx]]
            matched_deltas[b][positive_idx] = encode_box(props[positive_idx], gt_box[max_ids[positive_idx]])
            matched_gt_boxes[b][positive_idx] = gt_box[max_ids[positive_idx]]

            for pi, gi in zip(positive_idx, max_ids[positive_idx]):
                # gt_mask shape: [M, H, W]
                m = gt_mask[gi].unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

                # Proposal box (x1,y1,x2,y2)
                x1, y1, x2, y2 = props[pi].round().int()
                # clamp 保證不超出 mask 尺寸 (W,H)
                x1 = x1.clamp(0, gt_mask.shape[2]-1)
                x2 = x2.clamp(0, gt_mask.shape[2]-1)
                y1 = y1.clamp(0, gt_mask.shape[1]-1)
                y2 = y2.clamp(0, gt_mask.shape[1]-1)

                if x2 <= x1 or y2 <= y1:
                    crop = torch.zeros((1,1,mask_h,mask_w), dtype=mask_dtype, device=device)
                else:
                    crop = m[:, :, y1:y2+1, x1:x2+1]  # 裁切 mask
                    crop = F.interpolate(crop, size=(mask_h, mask_w), mode='bilinear', align_corners=False)

                matched_gt_masks[b, pi] = crop.squeeze(0).squeeze(0)

        return matched_proposals, matched_labels, matched_deltas, matched_gt_boxes, matched_gt_masks

class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.predictor = nn.Conv2d(256, num_classes, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x shape: [B*N, C, 14, 14]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.deconv(x))
        return self.predictor(x)  # [B*N, num_classes, 28, 28]
    
class CascadeRCNN(nn.Module):
    def __init__(self,
                 num_classes=3,
                 cascade_iou_threshold=[0.5, 0.6, 0.7], 
                 rpn_anchor_ratios=[0.5, 1.0, 2.0], 
                 fpn_out_channels=256, 
                 num_stages=3,
                 fpn_downscale_ratio=16,
                 pool_size=7,
                 fc_dim=1024,
                 img_size=(384,512)):
        
        super().__init__()
        self.num_stages = num_stages
        self.rpn_anchor_ratios = rpn_anchor_ratios
        self.img_size = img_size

        # 每層 stride（對應 FPN 層）
        self.fpn_strides = [4, 8, 16, 32]
        # 每層的 scale 設定
        self.fpn_scales = [
            [8],   # P2 → 小物件
            [16],  # P3 → 中小物件
            [32],  # P4 → 中大物件
            [64]   # P5 → 大物件
        ]

        # Backbone + FPN (輸出單一融合 feature_map)
        self.backbone_fpn = MaskRCNNBackboneWithRPN()

        # RPN
        self.rpn = RPN(
            in_channels=fpn_out_channels,
            anchors_per_location=len(rpn_anchor_ratios)
        )

        # Proposal
        self.proposal_layer = ProposalLayer(image_shape=self.img_size)

        # Mask Head
        self.fpn_downscale_ratio = fpn_downscale_ratio
        self.mask_roi_align = RoIAlign(
            output_size=(14, 14), 
            spatial_scale=1.0 / fpn_downscale_ratio, 
            sampling_ratio=2, 
            aligned=True)
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # upsample to 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)  # [B*N, num_classes, 28, 28]
        )

        '''
        self.mask_head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 14->28
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 28->56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 56->112
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_classes, kernel_size=1),  # [B*N, num_classes, 112, 112]
        )
        '''

        # Cascade Head
        self.cascade_head = CascadeRCNNHead(
            pool_size=pool_size,
            num_classes=num_classes,
            fc_dim=fc_dim,
            stages=3
        )

        # 每階段 Target Layer
        self.target_layers = nn.ModuleList([
            DetectionTargetLayer(stage_iou_thresh, self.img_size)
            for stage_iou_thresh in cascade_iou_threshold  # e.g., [0.5, 0.6, 0.7]
        ])

    def decode_final_boxes(self, final_proposals, final_bbox_deltas, final_logits):
        scores = F.softmax(final_logits, dim=1)
        labels = scores.argmax(dim=1)
        scores_max = scores.max(dim=1).values

        boxes = decode_boxes(
            final_proposals.view(-1, 4),
            final_bbox_deltas.view(-1, final_logits.shape[-1] * 4),
            labels
        )

        return boxes, labels, scores_max

    def generate_base_anchors(self, base_size=16, ratios=[0.5, 1.0, 2.0], scales=[8, 16, 32]):
        """
        生成一組 base anchors，中心在 (0, 0)
        base_size: 單位 anchor 的基準邊長
        ratios: 不同比例（寬/高）
        scales: 不同大小的放縮比例
        """
        anchors = []
        for ratio, scale in itertools.product(ratios, scales):
            area = (base_size * scale) ** 2
            area = torch.tensor(area, dtype=torch.float32)   # 確保是 Tensor
            w = torch.sqrt(area / ratio)
            h = w * ratio
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors)  # shape = [A, 4]

    def shift_anchors(self, feature_height, feature_width, stride, base_anchors):
        """
        將 base anchors 平移到整個 feature map 上
        """
        shift_x = torch.arange(0, feature_width) * stride
        shift_y = torch.arange(0, feature_height) * stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing="xy")

        shifts = torch.stack([
            shift_x.reshape(-1),
            shift_y.reshape(-1),
            shift_x.reshape(-1),
            shift_y.reshape(-1)
        ], dim=1)

        # 加上 base_anchors → shape = [K*A, 4]
        A = base_anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (
            base_anchors.reshape((1, A, 4)) +
            shifts.reshape((K, 1, 4))
        )
        return all_anchors.reshape((-1, 4))

    def generate_anchors_for_feature_map(self, feature_map_shape, stride, ratios, scales):
        """
        給定 feature map 形狀與參數，生成對應的 anchors
        """
        H, W = feature_map_shape
        base_anchors = self.generate_base_anchors(base_size=stride, ratios=ratios, scales=scales)
        anchors = self.shift_anchors(H, W, stride, base_anchors)
        return anchors

    def generate_fpn_anchors(self, feature_maps, strides, ratios, scales_per_level):
        """
        為多層 FPN 生成 anchors
        feature_maps: dict or OrderedDict, 每層特徵圖，例如 {'0': P2, '1': P3, ...}
        strides: list, 每層的下採樣比例，例如 [4, 8, 16, 32]
        ratios: list, anchor 寬高比
        scales_per_level: list of list, 每層的 scale 設定
        return: Tensor [總anchor數, 4]
        """
        all_anchors = []
        for idx, (name, feat) in enumerate(feature_maps.items()):
            _, _, H, W = feat.shape
            stride = strides[idx]
            scales = scales_per_level[idx]

            base_anchors = self.generate_base_anchors(base_size=stride, ratios=ratios, scales=scales)
            anchors = self.shift_anchors(H, W, stride, base_anchors)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)

    def forward(self, images, image_meta, gt_boxes=None, gt_labels=None, gt_masks=None, mode='train'):
        """
        mode: 'train', 'val', 'test'
        """
        batch_size, _, _, _ = images.shape

        # === 1. Backbone FPN ===
        feature_maps = self.backbone_fpn(images)

        # === 2. Anchors 生成 ===
        anchors = self.generate_fpn_anchors(
            feature_maps=feature_maps,
            strides=self.fpn_strides,
            ratios=self.rpn_anchor_ratios,
            scales_per_level=self.fpn_scales
        )

        # === 3. RPN 預測 ===
        rpn_class_logits_list, rpn_probs_list, rpn_bbox_list = [], [], []
        for _, feat in feature_maps.items():
            rpn_class_logits, rpn_probs, rpn_bbox = self.rpn(feat)
            rpn_class_logits_list.append(rpn_class_logits)
            rpn_probs_list.append(rpn_probs)
            rpn_bbox_list.append(rpn_bbox)

        # 多層合併
        rpn_class_logits = torch.cat(rpn_class_logits_list, dim=1)
        rpn_probs = torch.cat(rpn_probs_list, dim=1)
        rpn_bbox = torch.cat(rpn_bbox_list, dim=1)

        # === 4. Proposal Layer ===
        proposals = self.proposal_layer(anchors, rpn_probs, rpn_bbox)  # [B, N, 4]

        # ========== TRAIN 模式 ==========
        if mode == 'train':
            assert gt_boxes is not None and gt_labels is not None, "Train mode requires ground truth data"
            losses = {}

            # ROIAlign 批次 index
            num_props = proposals.shape[1]
            proposals_with_batch_idx = torch.cat([
                torch.arange(batch_size).view(-1, 1).repeat(1, num_props).view(-1, 1).float(),
                proposals.view(-1, 4)
            ], dim=1)

            # Padding gt_boxes
            max_num_boxes = max([boxes.shape[0] for boxes in gt_boxes])
            gt_padded_boxes = torch.zeros((batch_size, max_num_boxes, 4), dtype=torch.float32)
            for i, boxes in enumerate(gt_boxes):
                if boxes.numel() > 0:
                    gt_padded_boxes[i, :boxes.shape[0]] = boxes
            gt_padded_anchors = torch.tensor(anchors, dtype=torch.float32)

            # RPN targets
            rpn_match, rpn_bbox_targets = build_rpn_targets(gt_padded_anchors, gt_padded_boxes)
            rpn_target = rpn_match.clone()
            rpn_target[rpn_target == 0] = -1
            rpn_target[rpn_target == -1] = 0
            rpn_target[rpn_target == 1] = 1

            # RPN Loss
            losses['rpn_class_loss'] = F.cross_entropy(
                rpn_class_logits.view(-1, 2),
                rpn_target.view(-1).long(),
                ignore_index=-1
            )

            positive_indices = torch.nonzero(rpn_match == 1, as_tuple=False)
            if positive_indices.numel() > 0:
                flat_indices = positive_indices[:,0] * rpn_bbox.size(1) + positive_indices[:,1]
                losses['rpn_bbox_loss'] = F.smooth_l1_loss(
                    rpn_bbox.view(-1, 4)[flat_indices],
                    rpn_bbox_targets.view(-1, 4)[flat_indices]
                )
            else:
                losses['rpn_bbox_loss'] = torch.tensor(0.0)

            # Cascade Head + Mask
            refined_proposals = proposals
            all_class_logits, all_bbox_deltas, all_refined = self.cascade_head(proposals, feature_maps, image_meta)

            total_cls_loss, total_bbox_loss = 0.0, 0.0
            for stage in range(self.num_stages):

                matched_props, matched_labels, matched_deltas, matched_gt_boxes, matched_gt_masks = \
                    self.target_layers[stage](refined_proposals, gt_boxes, gt_labels, gt_masks)

                class_logits = all_class_logits[stage]
                bbox_deltas = all_bbox_deltas[stage]

                cls_loss, bbox_loss = cascade_rcnn_loss(
                    class_logits,
                    bbox_deltas,
                    matched_labels,
                    matched_deltas
                )
                total_cls_loss += cls_loss
                total_bbox_loss += bbox_loss
                refined_proposals = all_refined[stage]

                if stage == self.num_stages - 1:
                    roi_feats = self.mask_roi_align(feature_maps['2'], proposals_with_batch_idx)
                    mask_logits = self.mask_head(roi_feats)
                    mask_logits = F.interpolate(mask_logits, size=self.img_size, mode='bilinear', align_corners=False)
                    losses['mask_loss'] = mask_rcnn_loss_fn(mask_logits, matched_gt_masks, matched_labels, self.img_size)

            losses['cascade_cls_loss'] = total_cls_loss
            losses['cascade_bbox_loss'] = total_bbox_loss
            return losses

        # ========== VAL / TEST 模式（輸出預測） ==========
        elif mode in ['val', 'test']:
            all_class_logits, all_bbox_deltas, all_refined = self.cascade_head(proposals, feature_maps, image_meta)
            final_stage_logits = all_class_logits[-1]
            final_stage_bbox = all_bbox_deltas[-1]
            final_proposals = all_refined[-1]

            num_props = final_proposals.shape[1]
            proposals_with_batch_idx = torch.cat([
                torch.arange(batch_size).view(-1, 1).repeat(1, num_props).view(-1, 1).float(),
                final_proposals.view(-1, 4)
            ], dim=1)

            scores = F.softmax(final_stage_logits.view(-1, final_stage_logits.size(-1)), dim=1)
            labels = scores.argmax(dim=1)
            scores_max = scores.max(dim=1).values

            boxes = decode_boxes(
                final_proposals.view(-1, 4),
                final_stage_bbox.view(-1, final_stage_logits.size(-1), 4),
                labels
            )

            roi_feats = self.mask_roi_align(feature_maps['2'], proposals_with_batch_idx)

            mask_logits = self.mask_head(roi_feats)
            mask_logits = F.interpolate(mask_logits, size=self.img_size, mode='bilinear', align_corners=False)

            mask_probs = torch.sigmoid(mask_logits)
            masks = mask_probs[torch.arange(mask_probs.size(0)), labels]

            # 使用 interpolate 逐張放大 mask 回原圖大小 (因為 masks 是扁平 B*N，需要先根據 batch_index分組)
            num_props = final_proposals.shape[1]

            masks_resized = []

            for b in range(batch_size):
                # 找該 batch 的所有 proposal mask idx
                idx_start = b * num_props
                idx_end = (b + 1) * num_props

                masks_b = masks[idx_start:idx_end]  # [N_props, H_out, W_out]

                masks_b_resized = F.interpolate(
                    masks_b.unsqueeze(1),  # [N_props, 1, H_out, W_out]
                    size=self.img_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # [N_props, Hi, Wi]

                masks_resized.append(masks_b_resized)

            return boxes, labels, scores_max, masks_resized
    
# ------------------------------
# Yolov9 Model (m-4 backbone)
# ------------------------------
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()) for _ in range(n)
        ])
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        return self.conv2(x)

# ----------------------
# YOLOv9 Backbone
# ----------------------
class YOLOv9Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU()
        )
        self.c2f1 = C2f(128, 128, 1)
        self.down1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.c2f2 = C2f(256, 256, 2)
        self.down2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.c2f3 = C2f(256, 256, 3)
        self.down3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem(x)
        p3 = self.c2f1(x)
        p4 = self.c2f2(self.down1(p3))
        p5 = self.c2f3(self.down2(p4))
        p6 = self.down3(p5)
        return p3, p4, p5, p6

# ----------------------
# PAN Neck
# ----------------------
class PANNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce0 = nn.Conv2d(512, 256, 1)
        self.reduce1 = nn.Conv2d(256, 128, 1)

        self.top_down0 = nn.Conv2d(512, 256, 3, 1, 1)  # p4 + p5_up (256 + 256)
        self.top_down1 = nn.Conv2d(256, 128, 3, 1, 1)  # p3 + p4_up (128 + 128)

        self.down0 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down1 = nn.Conv2d(256, 256, 3, 2, 1)

        self.bottom_up0 = nn.Conv2d(256 + 128, 256, 3, 1, 1)
        self.bottom_up1 = nn.Conv2d(256 + 512, 512, 3, 1, 1)

    def forward(self, p3, p4, p5, p6):
        p5_td = self.reduce0(p6)
        p5_up = F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest")
        p4_td = self.top_down0(torch.cat([p5_up, p4], dim=1))

        p4_td_red = self.reduce1(p4_td)
        p4_up = F.interpolate(p4_td_red, size=p3.shape[-2:], mode="nearest")
        p3_out = self.top_down1(torch.cat([p4_up, p3], dim=1))

        p4_down = self.down0(p3_out)
        p4_out = self.bottom_up0(torch.cat([p4_down, p4_td], dim=1))

        p5_down = self.down1(p4_out)
        p6_up = F.interpolate(p6, size=p5_down.shape[-2:], mode="nearest")  # <--- 修正這裡
        p5_out = self.bottom_up1(torch.cat([p5_down, p6_up], dim=1))

        return p3_out, p4_out, p5_out

# ----------------------
# YOLOv9-M4 with Anchor-aware Prediction
# ----------------------
class YOLOv9_M4(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 3
        self.strides = [8, 16, 32]
        self.anchors = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32),
            torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32),
            torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32),
        ]

        self.backbone = YOLOv9Backbone()
        self.neck = PANNeck()

        self.detect_head = nn.ModuleList([
            nn.Conv2d(128, self.num_anchors * (num_classes + 5), 1),
            nn.Conv2d(256, self.num_anchors * (num_classes + 5), 1),
            nn.Conv2d(512, self.num_anchors * (num_classes + 5), 1),
        ])

    def forward(self, x):
        p3, p4, p5, p6 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5, p6)

        outputs = []
        for i, (feat, head, stride, anchors) in enumerate(zip([p3, p4, p5], self.detect_head, self.strides, self.anchors)):

            B, _, H, W = feat.shape
            na = self.num_anchors
            out = head(feat)
            out = out.view(B, na, self.num_classes + 5, H, W).permute(0, 1, 3, 4, 2).contiguous()

            # decode
            boxes, obj, cls = decode_outputs(out, anchors, stride, self.num_classes)

            outputs.append({
                "boxes": boxes,
                "obj": obj,
                "cls": cls,
                "stride": stride,
                "anchors": anchors.to(x.device)
            })

        return outputs

# ------------------------------
# NMS + decode
# ------------------------------
def decode_outputs(out, anchors, stride, num_classes):
    """
    out: [B, na, H, W, 5+num_classes]
    anchors: Tensor[na, 2]
    return: boxes [B, S, 4], obj [B, S, 1], cls [B, S, C]
    """
    B, na, H, W, _ = out.shape
    device = out.device
    C = num_classes

    # 建立 grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).to(device)  # [H, W, 2]
    grid = grid.view(1, 1, H, W, 2)

    # expand anchors: [1, na, 1, 1, 2]
    anchors = anchors.view(1, na, 1, 1, 2).to(device)

    # sigmoid 解碼
    xy = (out[..., 0:2].sigmoid() + grid) * stride        # center
    wh = (out[..., 2:4].exp() * anchors)                  # size
    obj = out[..., 4:5].sigmoid()
    cls = out[..., 5:].sigmoid()

    boxes = torch.cat([xy, wh], dim=-1)                   # [B, na, H, W, 4]
    boxes = boxes.view(B, -1, 4)
    obj = obj.view(B, -1, 1)
    cls = cls.view(B, -1, C)

    return boxes, obj, cls

# ------------------------------
# NMS helper
# ------------------------------
def nms_postprocess(cls_logits, obj_logits, boxes, conf_thresh=0.25, iou_thresh=0.5):
    B, C, H, W = cls_logits.shape
    cls_scores = torch.sigmoid(cls_logits)
    obj_scores = torch.sigmoid(obj_logits)
    scores = cls_scores * obj_scores  # [B, C, H, W]

    boxes = boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, HW, 4]
    scores = scores.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, HW, C]

    results = []
    for i in range(B):
        scores_i, labels = scores[i].max(dim=1)
        mask = scores_i > conf_thresh
        boxes_i = boxes[i][mask]
        scores_i = scores_i[mask]
        labels_i = labels[mask]

        keep = torch.ops.torchvision.nms(boxes_i, scores_i, iou_thresh)
        results.append({"boxes": boxes_i[keep], "scores": scores_i[keep], "labels": labels_i[keep]})
    return results
    
# ------------------------------
# Segmentation Model (Token-Only, variable input)
# ------------------------------
class PromptEmbedding(nn.Module):
    def __init__(self, prompt_length, embed_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_length, embed_dim))

    def forward(self, batch_size):
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)  # [B, P, C]
    
class DINOv2TokenSegmentation(nn.Module):
    def __init__(self, num_classes, num_queries=100, prompt_length=5):
        super().__init__()
        # self.backbone = vit_large()
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Swin backbone
        config = SwinConfig(output_hidden_states=True)
        self.backbone = SwinModel(config)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.hidden_dim = self.backbone.config.hidden_size  # 1024 for Swin-L
        self.prompt = PromptEmbedding(prompt_length, self.hidden_dim)

        # Learnable query tokens
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Output heads
        self.class_head = nn.Linear(self.hidden_dim, num_classes + 1)  # +1 for 'no object' class
        self.mask_embed_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        B, _, H, W = x.shape

        # Backbone
        with torch.no_grad():
            swin_outputs = self.backbone(x, output_hidden_states=True)
            patch_tokens = swin_outputs.last_hidden_state  # [B, N, C]

        # Prompt + tokens
        prompt_tokens = self.prompt(B)  # [B, P, C]
        tokens = torch.cat([prompt_tokens, patch_tokens], dim=1)  # [B, P+N, C]
        tokens = tokens.permute(1, 0, 2)  # [P+N, B, C]

        # Queries
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]
        decoder_output = self.transformer_decoder(queries, tokens).permute(1, 0, 2)  # [B, Q, C]

        # Class & mask heads
        class_logits = self.class_head(decoder_output)   # [B, Q, num_classes+1]
        mask_embed = self.mask_embed_head(decoder_output)

        # Reshape patch tokens for mask prediction
        patch_tokens_count = patch_tokens.shape[1]
        patch_hw = int(patch_tokens_count ** 0.5)
        while patch_hw > 0 and patch_hw * (patch_tokens_count // patch_hw) != patch_tokens_count:
            patch_hw -= 1
        patch_h = patch_hw
        patch_w = patch_tokens_count // patch_hw

        src = patch_tokens.permute(0, 2, 1).reshape(B, self.hidden_dim, patch_h, patch_w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, src)
        mask_pred = F.interpolate(mask_pred, size=(H, W), mode="bilinear", align_corners=False)

        # Semantic segmentation logits
        seg_logits = class_logits.softmax(dim=-1)[..., :-1].permute(0, 2, 1) @ mask_pred.flatten(2)
        seg_logits = seg_logits.view(B, self.num_classes, H, W)

        # === Confidence scores ===
        class_probs = class_logits.softmax(dim=-1)               # [B, Q, num_classes+1]
        pred_probs, pred_classes = class_probs[..., :-1].max(-1) # [B, Q], [B, Q]
        seg_probs = seg_logits.softmax(dim=1)                    # [B, num_classes, H, W]

        return {
            "pred_logits": class_logits,     # raw logits
            "pred_probs": class_probs,       # per-query probabilities
            "pred_classes": pred_classes,    # predicted class index (no "no object")
            "pred_conf": pred_probs,         # confidence score per query
            "pred_masks": mask_pred,         # query masks
            "sem_seg": seg_logits,           # segmentation logits
            "sem_seg_probs": seg_probs       # segmentation probabilities
        }

# ------------------------------
# Mask2former Model
# ------------------------------

# FPN module
class FPN(nn.Module):
    """
    Feature Pyramid Network 將 C2~C5 轉為 P2~P5。
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_conv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats):
        # top-down：從高層向低層傳遞
        results = [self.lateral[-1](feats[-1])]
        for i in range(len(feats)-2, -1, -1):
            prev = F.interpolate(results[0], scale_factor=2, mode="nearest")
            lateral = self.lateral[i](feats[i])
            fused = lateral + prev
            results.insert(0, fused)
        return [conv(r) for conv, r in zip(self.fpn_conv, results)]
    
# RPN module
class DummyRPN(nn.Module):
    """
    簡化版的 RPN 模型（用於產生 proposal）
    """
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.obj_logits = nn.Conv2d(256, num_anchors, 1)
        self.bbox_deltas = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, feats):
        # 接收 P2~P5，依序產生 objectness 與 bbox 預測
        rpn_outs = []
        for f in feats:
            x = F.relu(self.conv(f))
            obj = self.obj_logits(x)
            bbox = self.bbox_deltas(x)
            rpn_outs.append((obj, bbox))
        return rpn_outs

# ROI align modules
def generate_proposals(batch_size=2, num_boxes=10):
    proposals = []
    for b in range(batch_size):
        boxes = torch.rand(num_boxes, 4)
        boxes = boxes * 224
        boxes[:, 2:] = boxes[:, :2] + torch.abs(boxes[:, 2:] - boxes[:, :2])
        batch_idx = torch.full((num_boxes, 1), b)
        proposals.append(torch.cat([batch_idx, boxes], dim=1))  # [N, 5]
    return torch.cat(proposals, dim=0)

class ROIHeads(nn.Module):
    def __init__(self, in_channels=256, num_classes=3, img_size=(224, 224), pool_size=7):
        super().__init__()
        self.pool_size = pool_size
        self.img_size = img_size
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, feats, proposals):
        """
        feats: List[Tensor], multi-level FPN features (P2~P5)
        proposals: List[Tensor], [N, 5] format (batch_idx, x1, y1, x2, y2)
        image_shapes: List[(H, W)]
        """
        # 假設都使用 P2 特徵圖做 RoIAlign
        feat = feats[0]
        pooled = roi_align(
            feat, proposals,
            output_size=self.pool_size,
            spatial_scale=1.0,  # 因為 P2 尺寸與輸入不一樣，需改成 1/4 若下採樣過
            aligned=True
        )  # [K, C, 7, 7]

        x = pooled.view(pooled.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        mask_logits = self.mask_head(pooled)
        mask_logits_upsampled = F.interpolate(mask_logits, size=self.img_size, mode='bilinear', align_corners=False)
        mask_logits = torch.sigmoid(mask_logits_upsampled)

        return cls_logits, bbox_deltas, mask_logits
    
class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.ema_model.eval()

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1. - self.decay) * param.data

def apply_lora_to_swin(swin_backbone: nn.Module, r: int = 8, alpha: int = 16):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],  # 目標是 attention 層的 qkv
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    lora_model = get_peft_model(swin_backbone, config)
    return lora_model

# Swin-transformer backbone module
class SwinBackbone(nn.Module): 
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=pretrained, features_only=True)
        # self.backbone = apply_lora_to_swin(self.backbone)
        # for name, param in self.backbone.named_parameters():
        #     if "lora_" not in name:
        #         param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)  # List[Tensor]
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return feats
    
class MaskRCNNSwin(nn.Module):
    def __init__(self, num_classes=3, img_size=(224, 224)):
        super().__init__()
        self.backbone = SwinBackbone()
        self.fpn = FPN([128, 256, 512, 1024], 256)
        self.img_size = img_size

        anchor_sizes = ((32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        self.rpn = RegionProposalNetwork(
            anchor_generator=AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios),
            head=RPNHead(in_channels=256, num_anchors=3),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 1000, "testing": 300},
            nms_thresh=0.7
        )

        self.roi_heads = ROIHeads(256, num_classes, img_size)
        self.objectness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def convert_per_image_targets(self, per_image, img_size):
        result = []
        for item in per_image:
            if item == None:
                result.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0, img_size[0], img_size[1]), dtype=torch.float32),
                })
            else:
                boxes = item["bboxes"]
                labels = item["labels"]
                masks = item["masks"]

                if isinstance(boxes, list):
                    boxes = torch.stack(boxes).float()
                if isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.int64)
                if isinstance(masks, list):
                    masks = torch.stack(masks).float()

                result.append({
                    "boxes": boxes if boxes.numel() > 0 else torch.zeros((0, 4), dtype=torch.float32),
                    "labels": labels if labels.numel() > 0 else torch.zeros((0,), dtype=torch.int64),
                    "masks": masks if masks.numel() > 0 else torch.zeros((0, img_size[0], img_size[1]), dtype=torch.float32),
                })
        return result

    def forward(self, x, targets=None):
        device = x.device
        batch_size = x.size(0)

        # Backbone + FPN
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        fpn_feats_dict = OrderedDict({f"fpn_{i}": f for i, f in enumerate(fpn_feats)})

        # Make ImageList
        image_sizes = [tuple(x.shape[-2:])] * batch_size
        image_list = ImageList(x, image_sizes)

        if targets or (not any(t is None for t in targets)):
            targets = self.convert_per_image_targets(targets, self.img_size)
            proposals, rpn_losses = self.rpn(image_list, fpn_feats_dict, targets)
        else:
            proposals, _ = self.rpn(image_list, fpn_feats_dict)
            rpn_losses = None

        # --- Apply NMS and Keep Index ---
        nms_results = apply_nms_to_proposals_with_index(proposals, scores=None)
        final_proposals = []
        # final_keep_indices = []
        offset = 0

        for i, (boxes, keep_idx) in enumerate(nms_results):
            final_proposals.append(boxes)
            # final_keep_indices.append(keep_idx + offset)
            offset += proposals[i].shape[0]

        # all_keep_indices = torch.cat(final_keep_indices, dim=0)

        # ROI feature heads
        cls_logits, bbox_deltas, mask_logits = self.roi_heads(fpn_feats, final_proposals)

        # Objectness Head
        obj_logits = self.objectness_head(fpn_feats[0])

        # Generate batch_indices (for loss)
        batch_indices = torch.cat([
            torch.full((len(p),), i, dtype=torch.long)
            for i, p in enumerate(final_proposals)
        ], dim=0).to(device)

        return cls_logits, bbox_deltas, mask_logits, obj_logits, batch_indices, rpn_losses, final_proposals

# ---------------------- MoE Feed-Forward Block ---------------------- #
class MoEFFN(nn.Module):
    def __init__(self, dim, num_experts=4, hidden_dim=2048, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        B, N, D = x.shape
        gate_scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = topk_indices[..., i]
            expert_weight = topk_scores[..., i].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                mask = (idx == expert_id)
                if mask.any():
                    x_expert = x[mask]
                    y_expert = self.experts[expert_id](x_expert)
                    out[mask] += expert_weight[mask] * y_expert
        return out

# ---------------------- Transformer Block with MoE ---------------------- #
class MoETransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe_ffn = MoEFFN(dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.moe_ffn(self.norm2(x))
        return x

# ---------------------- ICD-10 or SNOMED Mapping ---------------------- #
MEDICAL_KNOWLEDGE_BASE = {
    "green lesion": {
        "icd_code": "K13.7",
        "condition": "Benign lesion of oral mucosa",
        "clinical_note": "Likely indicative of mild irritation or early stage leukoplakia. Monitor regularly."
    },
    "yellow lesion": {
        "icd_code": "K02.9",
        "condition": "Dental caries, unspecified",
        "clinical_note": "May reflect bacterial infection or necrotic tissue. Recommend further dental imaging."
    },
    "red lesion": {
        "icd_code": "D10.0",
        "condition": "Benign neoplasm of lip",
        "clinical_note": "Possibly erythroplakia or carcinoma in situ. Immediate biopsy is recommended."
    }
}

# ---------------------- Diagnostic Error Warning ---------------------- #
def check_diagnosis_risk_warnings(structured_result, threshold=0.85):
    if structured_result.get("score", 0) >= threshold and structured_result.get("risk_level", "") == "High":
        structured_result["warning"] = "⚠️ High-risk lesion detected. Double-check required."
    return structured_result

# ---------------------- Doctor Feedback Integration (Reinforcement Logging) ---------------------- #
FEEDBACK_LOG_PATH = "doctor_feedback_log.jsonl"

def log_doctor_feedback(case_id, prediction, feedback_text, correct_label=None):
    entry = {
        "case_id": case_id,
        "prediction": prediction,
        "doctor_feedback": feedback_text,
        "correct_label": correct_label
    }
    with open(FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ---------------------- Reinforcement Dataset Builder ---------------------- #
def build_rlhf_dataset(log_path=FEEDBACK_LOG_PATH, output_path="rlhf_training_data.json"):
    dataset = []
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            prediction = entry.get("prediction", {})
            correct_label = entry.get("correct_label")
            if prediction and correct_label:
                prompt = f"The model predicted: {prediction['label']} with summary: {prediction['summary']}\n"
                prompt += f"Doctor feedback: {entry['doctor_feedback']}\n"
                prompt += f"Correct diagnosis is: {correct_label}"
                dataset.append({"prompt": prompt, "completion": correct_label})
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"RLHF dataset saved to {output_path}, total: {len(dataset)} examples")

# ---------------------- GPT-4-Vision + CLIP Text/Image Prompt ---------------------- #
def build_vision_prompt(image_path, lesion_labels):
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    image = Image.open(image_path).convert("RGB")
    text_inputs = [f"photo of {label}" for label in lesion_labels]
    inputs = clip_processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image.softmax(dim=1)
        scores = logits_per_image.squeeze().tolist()
    # build prompt for GPT-4-Vision
    prompt = f"This is an oral cancer image with suspected lesions: {', '.join(lesion_labels)}."
    return prompt, scores

# ---------------------- GPT-4 Diagnosis via Function Calling ---------------------- #
def query_gpt4_structured(prompt, label, score):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical diagnosis assistant specialized in oral cancer imaging."},
            {"role": "user", "content": prompt}
        ],
        functions=[
            {
                "name": "report_diagnosis",
                "description": "Return structured diagnostic output for lesion report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "score": {"type": "number"},
                        "summary": {"type": "string"},
                        "icd_code": {"type": "string"},
                        "risk_level": {"type": "string"},
                        "recommendation": {"type": "string"}
                    },
                    "required": ["label", "summary", "icd_code", "risk_level", "recommendation"]
                }
            }
        ],
        function_call={"name": "report_diagnosis"},
        temperature=0.5,
        max_tokens=300
    )
    func_data = response.choices[0].message.get("function_call", {}).get("arguments", "{}")
    try:
        result = json.loads(func_data)
        result = check_diagnosis_risk_warnings(result)
        return result
    except:
        return {"label": label, "score": score, "summary": "(parse failed)", "icd_code": "N/A"}

# ---------------------- LLM Diagnosis Head ---------------------- #
class LLMDiagnosisHead(nn.Module):
    def __init__(self, model_name="microsoft/phi-2", use_gpt4=False, use_function_calling=False):
        super().__init__()
        self.use_gpt4 = use_gpt4
        self.use_function_calling = use_function_calling
        if not use_gpt4:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name).eval()

    def build_prompt(self, label, score):
        import random
        meta = MEDICAL_KNOWLEDGE_BASE.get(label.lower(), {})
        template = random.choice(DIAGNOSIS_TEMPLATES)
        return template.format(
            label=label,
            score=score,
            icd_code=meta.get("icd_code", "N/A"),
            condition=meta.get("condition", "Unknown condition")
        )

    def forward(self, clip_texts, clip_scores, device="cpu"):
        batch_outputs = []
        for labels, scores in zip(clip_texts, clip_scores):
            prompts = [self.build_prompt(t, s) for t, s in zip(labels, scores)]
            if self.use_gpt4 and self.use_function_calling:
                results = [query_gpt4_structured(p, t, s) for p, t, s in zip(prompts, labels, scores)]
                batch_outputs.append(results)
            elif self.use_gpt4:
                results = [query_gpt4(p) for p in prompts]
                batch_outputs.append(results)
            else:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                outputs = self.llm.generate(**inputs, max_new_tokens=80)
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_outputs.append(decoded)
        return batch_outputs

# ---------------------- UNETR + MoE + RoI + CLIP + Semi-Supervised + LLM ---------------------- #
'''
class UNETR_MoE_CLIP_RCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, img_size=(128, 128),
                 feature_size=16, hidden_size=768, num_layers=12,
                 roi_output_size=(7, 7), num_heads=12):
        super().__init__()

        self.img_size = img_size
        self.patch_size = 16

        # self.hidden_size = self.get_valid_hidden_size((img_size[0] // self.patch_size) * (img_size[1] // self.patch_size), num_heads)
        self.hidden_size = hidden_size
        # self.num_heads = self.find_divisible_heads(self.hidden_size)
        self.num_heads = num_heads
        self.roi_output_size = roi_output_size

        self.num_patches = (img_size[0] // self.patch_size) * (img_size[1] // self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # --- Patch Embedding ---
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )

        # --- Transformer Encoder ---
        self.encoder_layers = nn.ModuleList([
            MoETransformerBlock(hidden_size, heads=num_heads) for _ in range(num_layers)
        ])

        # --- Decoder ---
        self.decoder4 = UnetrBasicBlock(spatial_dims=2, in_channels=hidden_size, out_channels=feature_size * 8, kernel_size=3, stride=1, norm_name="instance")
        self.decoder3 = UnetrBasicBlock(spatial_dims=2, in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=3, stride=1, norm_name="instance")
        self.decoder2 = UnetrBasicBlock(spatial_dims=2, in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=3, stride=1, norm_name="instance")
        self.decoder1 = UnetrBasicBlock(spatial_dims=2, in_channels=feature_size * 2, out_channels=feature_size, kernel_size=3, stride=1, norm_name="instance")
        self.upsample_final = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.ReLU()
        )

        self.seg_embedding = nn.Conv2d(feature_size, 512, kernel_size=1)
        self.seg_output = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

        self.roi_align = RoIAlign(output_size=self.roi_output_size, spatial_scale=1.0, sampling_ratio=-1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_size, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 1)
        )

    def find_divisible_heads(self, hidden_size, max_heads=12):
        for h in reversed(range(1, max_heads + 1)):
            if hidden_size % h == 0:
                return h
        return 1  # 最差 fallback
    
    def _make_pos_embed(self, H_, W_):
        """Create new position embedding dynamically."""
        print(f"[🔧] Creating new pos_embed for {H_}x{W_} patches")
        N = H_ * W_
        pos_embed = nn.Parameter(torch.zeros(1, N, self.hidden_size))
        nn.init.trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def get_valid_hidden_size(self, desired_size, num_heads):
        if desired_size % num_heads == 0:
            return desired_size
        return ((desired_size // num_heads) + 1) * num_heads

    def resize_pos_embed(self, posemb, new_grid_size, old_grid_size):
        """Reshape and interpolate position embeddings"""
        if posemb.shape[1] == posemb.shape[2]:
            posemb = posemb.transpose(1, 2)

        B, N, D = posemb.shape
        posemb = posemb.reshape(1, old_grid_size[0], old_grid_size[1], D).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_grid_size, mode='bilinear', align_corners=False)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, D)
        return posemb
    
    def forward(self, x, rois=None, text_labels=None, mask_labels=None, generate_report=False):

        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input height and width must be divisible by patch_size ({self.patch_size})"

        H_, W_ = H // self.patch_size, W // self.patch_size
        N = H_ * W_

        # Patch Embedding
        x = self.patch_embed(x)                     # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)            # [B, N, C]

        # Dynamic Positional Embedding Resize
        if self.pos_embed.shape[1] != N:
            old_N = self.pos_embed.shape[1]
            old_grid = int(old_N ** 0.5)
            resized = self.resize_pos_embed(self.pos_embed, new_grid_size=(H_, W_), old_grid_size=(old_grid, old_grid))
            self.pos_embed = nn.Parameter(resized)

        x = x + self.pos_embed                      # [B, N, C]

        # Transformer Encoder
        for blk in self.encoder_layers:
            x = blk(x)

        # Reshape for decoder
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_, w=W_)
        tokens = x

        # Decoder
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        seg_feat = self.upsample_final(x)
        seg_feat = F.interpolate(seg_feat, size=(H, W), mode='bilinear', align_corners=False)

        # Segmentation
        seg_feat_512 = self.seg_embedding(seg_feat)
        seg_logits = self.seg_output(seg_feat_512)

        # ROI Head
        masks = None
        if rois is not None:
            roi_feat = self.roi_align(tokens, rois)
            mask_logits = self.mask_head(roi_feat)
            masks = F.interpolate(mask_logits, size=(H, W), mode='bilinear', align_corners=False)

        return seg_feat, seg_logits, None, masks, None, None, None
'''

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks
    
dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)