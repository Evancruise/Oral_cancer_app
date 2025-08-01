# line_infer.py
import os
import uuid
import cv2
import torch
from flask import request
from linebot.models import MessageEvent, ImageMessage, ImageSendMessage, TextSendMessage
from linebot import LineBotApi
from DINO_v2_self.model import YOLOv9_M4  # 可換成你的模型
from app import draw_predictions, generate_prompt_response, transform_image_to_tensor
import torch.nn.functional as F
import torchvision.ops as ops

# === 假設這些是全域變數或由主程式注入 ===
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
web_url = "your_ngrok_domain_or_domain.com"  # 改成實際 ngrok 或網域
num_classes = 2
class_names = ["normal", "cancer"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def handle_image(event, line_bot_api: LineBotApi):
    message_id = event.message.id
    user_id = event.source.user_id

    filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(UPLOAD_DIR, filename)
    result_path = os.path.join(RESULT_DIR, f"result_{filename}")

    image_content = line_bot_api.get_message_content(message_id)
    with open(img_path, 'wb') as f:
        for chunk in image_content.iter_content():
            f.write(chunk)

    # === 推論開始 ===
    orig_img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_tensor = transform_image_to_tensor(img_rgb).unsqueeze(0).to(device)

    model = YOLOv9_M4(num_classes=num_classes).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)

        boxes_list, scores_list = [], []
        for level in outputs:
            boxes = level["boxes"][0]
            obj = level["obj"][0].squeeze()
            cls = level["cls"][0].sigmoid()
            score, cls_idx = cls.max(dim=1)
            conf = obj * score
            mask = conf > 0.3
            boxes = boxes[mask]
            cls_idx = cls_idx[mask]
            conf = conf[mask]

            if len(boxes) > 0:
                scores = torch.stack([conf, cls_idx.float()], dim=1)
                boxes_list.append(boxes)
                scores_list.append(scores)

        if boxes_list:
            all_boxes = torch.cat(boxes_list, dim=0)
            all_scores = torch.cat(scores_list, dim=0)
            keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold=0.5)
            final_boxes = all_boxes[keep]
            final_classes = all_scores[keep][:, 1]
            preds = torch.cat([final_boxes, final_classes.unsqueeze(1)], dim=1)
        else:
            preds = torch.empty((0, 5))

    # === 繪製 & 儲存圖片 ===
    img_with_boxes = draw_predictions(orig_img.copy(), preds.cpu(), class_names)
    cv2.imwrite(result_path, img_with_boxes)

    # === 診斷文字 ===
    if preds.size(0) > 0:
        preds_with_conf = torch.cat([
            preds[:, :4],
            all_scores[keep][:, 0].unsqueeze(1),
            preds[:, 4].unsqueeze(1)
        ], dim=1)
        chatbot_response = generate_prompt_response(preds_with_conf.cpu().numpy())
    else:
        chatbot_response = "未偵測到病灶，請確認圖像品質或重試。"

    result_url = f"https://{web_url}/{result_path}"

    line_bot_api.reply_message(
        event.reply_token,
        [
            ImageSendMessage(
                original_content_url=result_url,
                preview_image_url=result_url
            ),
            TextSendMessage(text=chatbot_response)
        ]
    )
