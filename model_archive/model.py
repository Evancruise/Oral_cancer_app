import torch
import torch.nn as nn
import openai
import os
import json
import torch.nn.functional as F
# from dinov2.models.vision_transformer import vit_large
from torchvision.models.detection.image_list import ImageList
from transformers import SwinModel, SwinConfig, CLIPTextModel, CLIPVisionModel, CLIPProcessor, CLIPTokenizer, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
from timm import create_model
import copy
from torchvision.ops import roi_align
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.rpn import AnchorGenerator
from model_archive.utils_func import apply_nms_to_proposals_with_index
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import LoraModel
from monai.networks.blocks import UnetrBasicBlock
from einops import rearrange
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import RoIAlign
import math

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
        print("x.shape:", x.shape)

        with torch.no_grad():
            swin_outputs = self.backbone(x, output_hidden_states=True)
            patch_tokens = swin_outputs.last_hidden_state  # [B, N, C]

        prompt_tokens = self.prompt(B)  # [B, P, C]
        tokens = torch.cat([prompt_tokens, patch_tokens], dim=1)  # [B, P+N, C]
        tokens = tokens.permute(1, 0, 2)  # [P+N, B, C]

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        decoder_output = self.transformer_decoder(queries, tokens).permute(1, 0, 2)  # [B, Q, C]

        class_logits = self.class_head(decoder_output)
        mask_embed = self.mask_embed_head(decoder_output)

        patch_tokens_count = patch_tokens.shape[1]
        patch_hw = int(patch_tokens_count ** 0.5)
        while patch_hw > 0 and patch_hw * (patch_tokens_count // patch_hw) != patch_tokens_count:
            patch_hw -= 1
        patch_h = patch_hw
        patch_w = patch_tokens_count // patch_hw

        src = patch_tokens.permute(0, 2, 1).reshape(B, self.hidden_dim, patch_h, patch_w)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, src)
        mask_pred = F.interpolate(mask_pred, size=(H, W), mode="bilinear", align_corners=False)

        seg_logits = class_logits.softmax(dim=-1)[..., :-1].permute(0, 2, 1) @ mask_pred.flatten(2)
        seg_logits = seg_logits.view(B, self.num_classes, H, W)

        return {"pred_logits": class_logits, "pred_masks": mask_pred, "sem_seg": seg_logits}

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
