from flask import Flask, render_template, request, redirect, session, url_for, jsonify, send_from_directory, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage, 
    TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent,
    URIAction, ButtonsTemplate
)
from linebot.exceptions import InvalidSignatureError
from app import save_record, draw_predictions, generate_prompt_response, transform_image_to_tensor
from DINO_v2_self.model import YOLOv9_M4, DINOv2TokenSegmentation
import cv2
import torch
import os
import requests
from pyngrok import ngrok
import webbrowser
import uuid
from DINO_v2_self.config import Config
import torchvision.ops as ops
# from line_infer import handle_image
import psutil

def kill_existing_ngrok():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'ngrok' in proc.info['name'].lower():
                print(f"🔪 殺掉舊的 ngrok：PID {proc.info['pid']}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

# === Initialization ===
app = Flask(__name__)

all_config = Config()

liff_id = all_config.liff_id
user_id = all_config.user_id
line_channel_access_token = all_config.line_channel_access_token
web_url = all_config.web_url
channel_id = all_config.channel_id
handler = all_config.handler
line_bot_api = all_config.line_bot_api
device = all_config.device
num_classes = all_config.num_classes
class_names = all_config.class_names

IMAGE_DIR = "static/images"
MODEL_DIR = "static/models"
ANNOT_DIR = "annotations"
RESULT_DIR = "static/results"
UPLOAD_DIR = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 從 .env 或 config.json 載入設定
def load_config():
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        config = {
            "channel_access_token": line_channel_access_token,
            "liff_id": liff_id,
            "channel_id": channel_id,
            "port": 5000,
        }
    return config

def update_liff_endpoint(liff_id, access_token, new_url):
    url = f"https://api.line.me/liff/v1/apps/{liff_id}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    data = {
        "view": {
            "type": "full",
            "url": new_url
        }
    }

    response = requests.put(url, headers=headers, json=data)

    if response.status_code == 200:
        print(f"✅ LIFF endpoint 已成功更新為: {new_url}")
    else:
        print(f"❌ 更新失敗: {response.status_code}")
        print(response.text)

def update_line_webhook_url(new_url):
    print(f"Updating LINE webhook URL to: {new_url}")
    url = "https://api.line.me/v1/bot/channel/webhook/endpoint"
    headers = {
        "Authorization": f"Bearer {line_channel_access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "endpoint": new_url
    }
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 200:
        print("✅ LINE webhook URL updated successfully!")
    else:
        print(f"❌ Failed to update LINE webhook URL: {response.status_code}")
        print(response.text)

# 自動啟動 ngrok 並回傳 URL
def start_ngrok(port=5000):
    public_url = ngrok.connect(port, bind_tls=True)
    print(f"🌐 ngrok 啟動成功: {public_url}")
    return public_url

@app.route("/callback", methods=['POST'])
def callback():
    print("[DEBUG] /callback called")
    signature = request.headers.get('X-Line-Signature', None)
    if signature is None:
        print("No X-Line-Signature header")
        abort(400)
    body = request.get_data(as_text=True)
    print(f"[DEBUG] Body: {body}")
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature error")
        abort(400)
    except Exception as e:
        print(f"Handler error: {e}")
        abort(500)
    return "OK", 200

# === Model inference API
def send_liff_button(user_id, liff_url):
    line_bot_api.push_message(
        user_id,
        TemplateSendMessage(
            alt_text="點選進入口腔癌診斷工具",
            template=ButtonsTemplate(
                title="AI 口腔癌診斷",
                text="請點下方按鈕開啟圖像診斷頁面",
                actions=[
                    URIAction(
                        label="開啟診斷頁",
                        uri=liff_url
                    )
                ]
            )
        )
    )

@app.route("/inference", methods=["POST"])
def inference():

    image_name = request.form.get("image")
    model_name = request.form.get("model")

    print("image_name:", image_name, " | model_name:", model_name)

    # session["image_name"] = image_name
    # session["model_name"] = model_name

    img_path = os.path.join(UPLOAD_DIR, image_name)
    # image, pred_mask = infer_single_image_for_web(img_path)

    orig_img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_tensor = transform_image_to_tensor(img_rgb).unsqueeze(0).to(device)  # 假設你有這個轉換

    if model_name.split("_")[0] == "yolov9":
        model = YOLOv9_M4(num_classes=num_classes).to(device)
            # 推論單張圖片
        with torch.no_grad():
            model.eval()
            outputs = model(img_tensor)  # List[dict] 格式
            
            boxes_list, scores_list = [], []

            for level in outputs:
                boxes = level["boxes"][0]               # (N, 4)
                obj = level["obj"][0].squeeze()         # (N,)
                cls_logits = level["cls"][0]            # (N, num_classes)

                # 轉成機率後，取每個 row 最大值當作預測類別與 score
                cls_scores = cls_logits.sigmoid()       # (N, num_classes)
                max_scores, cls_indices = cls_scores.max(dim=1)  # (N,), (N,)

                conf = obj * max_scores                 # (N,)，YOLO 風格 conf 計算
                mask = conf > 0.3                       # 篩掉太低的信心預測

                boxes = boxes[mask]
                cls_indices = cls_indices[mask]
                conf = conf[mask]

                if len(boxes) > 0:
                    scores = torch.stack([conf, cls_indices.float()], dim=1)  # (N, 2)
                    boxes_list.append(boxes)
                    scores_list.append(scores)

            if boxes_list:
                all_boxes = torch.cat(boxes_list, dim=0)       # [N, 4]
                all_scores = torch.cat(scores_list, dim=0)     # [N, 2] → conf, class_id

                keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold=0.5)

                final_boxes = all_boxes[keep]                  # [M, 4]
                final_confs = all_scores[keep][:, 0]           # [M]
                final_classes = all_scores[keep][:, 1]         # [M]

                # ➤ concat [x1,y1,x2,y2,conf,class_id]
                preds = torch.cat([
                    final_boxes,
                    final_confs.unsqueeze(1),
                    final_classes.unsqueeze(1)
                ], dim=1)   # [M, 6]
            else:
                preds = torch.empty((0, 6))

            # 繪製預測框
            img_with_boxes = draw_predictions(orig_img.copy(), preds.cpu(), class_names, 0.3)

            # 儲存圖片
            result_filename = f"result_{image_name}"
            result_path = os.path.join(RESULT_DIR, result_filename)
            cv2.imwrite(result_path, img_with_boxes)

            # ...推論結果 preds = [x1,y1,x2,y2,class_id]
            if preds.size(0) > 0:
                # 加上 conf
                preds_with_conf = torch.cat([
                    preds[:, :4],
                    all_scores[keep][:, 0].unsqueeze(1),  # confidence
                    preds[:, 4].unsqueeze(1)  # class_id
                ], dim=1)
                chatbot_response = generate_prompt_response(preds_with_conf.cpu().numpy(), class_names)
            else:
                chatbot_response = "未偵測到病灶，請確認圖像品質或重試。"

        return jsonify({
            "filename": image_name,
            "result_img_url": RESULT_DIR + "/" + result_filename,
            "image_index": image_name,
            "model_index": model_name,
            "diagnosis_text": chatbot_response
        })
        
    elif model_name.split("_")[0] == "dinov2":
        model = DINOv2TokenSegmentation(num_classes=num_classes).to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(img_tensor)  # List[dict] 格式

            # 處理預測輸出
            pred_mask = outputs["sem_seg"][0]  # [C, H, W]
            pred_label = torch.argmax(pred_mask, dim=0).cpu().numpy().astype("uint8")  # [H, W]

            # 上色（可選）
            color_map = get_pascal_colormap(num_classes=21)  # 或你自己的 palette
            pred_color = color_map[pred_label]  # [H, W, 3]

            # 儲存圖片
            result_filename = f"seg_{image_name}"
            result_path = os.path.join(RESULT_DIR, result_filename)
            imageio.imwrite(result_path, pred_color)

            return jsonify({
                "filename": image_name,
                "result_img_url": RESULT_DIR + "/" + result_filename,  # 這路徑要對應你的靜態資料夾設置
                "image_index": image_name,
                "model_index": model_name
            })

# === 圖片訊息處理 ===
# Line Bot API
@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("Exception in handler.handle:", e)
        abort(500)

    return 'OK'

def handle_image(event, line_bot_api):
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

    print("Load model Yolov9_M4...")
    model = YOLOv9_M4(num_classes=num_classes).to(device)
    model.eval()

    with torch.no_grad():

        print("Start inferencing...")
        outputs = model(img_tensor)
        print("Outputs:", outputs)

        boxes_list, scores_list = [], []
        for level in outputs:
            boxes = level["boxes"][0]               # (N, 4)
            obj = level["obj"][0].squeeze()         # (N,)
            cls_logits = level["cls"][0]            # (N, num_classes)
            # 轉成機率後，取每個 row 最大值當作預測類別與 score
            cls_scores = cls_logits.sigmoid()       # (N, num_classes)
            max_scores, cls_indices = cls_scores.max(dim=1)  # (N,), (N,)
            conf = obj * max_scores                 # (N,)，YOLO 風格 conf 計算
            mask = conf > 0.3                       # 篩掉太低的信心預測
            boxes = boxes[mask]
            cls_indices = cls_indices[mask]
            conf = conf[mask]
            if len(boxes) > 0:
                scores = torch.stack([conf, cls_indices.float()], dim=1)  # (N, 2)
                boxes_list.append(boxes)
                scores_list.append(scores)

        if boxes_list:
            all_boxes = torch.cat(boxes_list, dim=0)       # [N, 4]
            all_scores = torch.cat(scores_list, dim=0)     # [N, 2] → conf, class_id
            keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold=0.5)
            final_boxes = all_boxes[keep]                  # [M, 4]
            final_confs = all_scores[keep][:, 0]           # [M]
            final_classes = all_scores[keep][:, 1]         # [M]
            # ➤ concat [x1,y1,x2,y2,conf,class_id]
            preds = torch.cat([
                final_boxes,
                final_confs.unsqueeze(1),
                final_classes.unsqueeze(1)
            ], dim=1)   # [M, 6]
        else:
            preds = torch.empty((0, 6))

    print("Finish inferencing...")

    print("Generating image...")
    # === 繪製 & 儲存圖片 ===
    img_with_boxes = draw_predictions(orig_img.copy(), preds.cpu(), class_names, 0.3)
    cv2.imwrite(result_path, img_with_boxes)

    print("Generating diagnosis...")
    # === 診斷文字 ===
    '''
    if preds.size(0) > 0:
        preds_with_conf = torch.cat([
            preds[:, :4],
            all_scores[keep][:, 0].unsqueeze(1),
            preds[:, 4].unsqueeze(1)
        ], dim=1)
        chatbot_response = generate_prompt_response(preds_with_conf.cpu().numpy(), class_names)
    else:
        chatbot_response = "未偵測到病灶，請確認圖像品質或重試。"
    '''

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

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    print("Into handle_image_message function...")
    handle_image(event, line_bot_api)

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text

    # 使用者輸入 "幫我診斷 abc.jpg"
    if "診斷" in user_text:
        image_name = user_text.replace("幫我診斷", "").strip()

        img_path = os.path.join(IMAGE_DIR, image_name)

        image, pred_mask = infer_single_image_for_web(img_path)
        
        if pred_mask is None:
            reply_text = "無法辨識圖片，請確認圖片格式或內容"
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            return
        
        result_filename = f"result_{image_name}"
        result_path = os.path.join(RESULT_DIR, result_filename)
        image.save(result_path)

        reply_text = f"已完成辨識 ✅\n結果圖連結:\nhttps://{web_url}/{result_path}"
    else:
        reply_text = "請輸入：幫我診斷 xxx.jpg"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(PostbackEvent)
def handle_postback(event):
    data = event.postback.data
    user_id = event.source.user_id

    if data == "action=query_record":
        # 📋 查詢該使用者最近篩檢紀錄（假設你有紀錄到 DB）
        record = query_user_latest_record(user_id)

        reply_text = (
            f"📋 你的最新篩檢紀錄：\n"
            f"- 檔名：{record['filename']}\n"
            f"- 模型：{record['model']}\n"
            f"- 結果：{record['result']}\n"
            f"- 時間：{record['timestamp']}"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    elif data == "action=go_home":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🏠 你已回到主選單，可以再次上傳圖片進行篩檢。")
        )

# LIFF API
@app.route('/liff_index')
def liff_index():
    # return render_template('liff_index.html')
    model_dict = []
    for model in os.listdir(MODEL_DIR):
        model_dict.append(model.split("_")[0])

    return render_template(
        "liff_index.html",
        model_dict=model_dict,
        liff_id=liff_id)

@app.route("/liff_infer", methods=["POST"])
def liff_infer():
    chatbot_response = ""

    user_id = request.form["userId"]
    file = request.files["image"]
    modelid = request.files["modelId"]

    filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(IMAGE_DIR, filename)
    file.save(img_path)
    
    if modelid == "yolov9":
        model = YOLOv9_M4(num_classes=num_classes).to(device)
        
        # 推論單張圖片
        with torch.no_grad():
            model.eval()
            outputs = model(img_tensor)  # List[dict] 格式
            
            boxes_list, scores_list = [], []

            for level in outputs:
                boxes = level["boxes"][0]               # (N, 4)
                obj = level["obj"][0].squeeze()         # (N,)
                cls_logits = level["cls"][0]            # (N, num_classes)

                # 轉成機率後，取每個 row 最大值當作預測類別與 score
                cls_scores = cls_logits.sigmoid()       # (N, num_classes)
                max_scores, cls_indices = cls_scores.max(dim=1)  # (N,), (N,)

                conf = obj * max_scores                 # (N,)，YOLO 風格 conf 計算
                mask = conf > 0.3                       # 篩掉太低的信心預測

                boxes = boxes[mask]
                cls_indices = cls_indices[mask]
                conf = conf[mask]

                if len(boxes) > 0:
                    scores = torch.stack([conf, cls_indices.float()], dim=1)  # (N, 2)
                    boxes_list.append(boxes)
                    scores_list.append(scores)

            if boxes_list:
                all_boxes = torch.cat(boxes_list, dim=0)       # [N, 4]
                all_scores = torch.cat(scores_list, dim=0)     # [N, 2] → conf, class_id

                keep = ops.nms(all_boxes, all_scores[:, 0], iou_threshold=0.5)

                final_boxes = all_boxes[keep]                  # [M, 4]
                final_confs = all_scores[keep][:, 0]           # [M]
                final_classes = all_scores[keep][:, 1]         # [M]

                # ➤ concat [x1,y1,x2,y2,conf,class_id]
                preds = torch.cat([
                    final_boxes,
                    final_confs.unsqueeze(1),
                    final_classes.unsqueeze(1)
                ], dim=1)   # [M, 6]
            else:
                preds = torch.empty((0, 6))

            # 繪製預測框
            img_with_boxes = draw_predictions(orig_img.copy(), preds.cpu(), class_names, 0.3)

            # 儲存圖片
            result_filename = f"result_{image_name}"
            result_path = os.path.join(RESULT_DIR, result_filename)
            cv2.imwrite(result_path, img_with_boxes)

            # ...推論結果 preds = [x1,y1,x2,y2,class_id]
            if preds.size(0) > 0:
                # 加上 conf
                preds_with_conf = torch.cat([
                    preds[:, :4],
                    all_scores[keep][:, 0].unsqueeze(1),  # confidence
                    preds[:, 4].unsqueeze(1)  # class_id
                ], dim=1)
                chatbot_response = generate_prompt_response(preds_with_conf.cpu().numpy(), class_names)
            else:
                chatbot_response = "未偵測到病灶，請確認圖像品質或重試。"

    elif modelid == "dinov2":    
        model = DINOv2TokenSegmentation(num_classes=num_classes).to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(img_tensor)  # List[dict] 格式

            # 處理預測輸出
            pred_mask = outputs["sem_seg"][0]  # [C, H, W]
            pred_label = torch.argmax(pred_mask, dim=0).cpu().numpy().astype("uint8")  # [H, W]

            # 上色（可選）
            color_map = get_pascal_colormap(num_classes=3)  # 或你自己的 palette
            pred_color = color_map[pred_label]  # [H, W, 3]

            # 儲存圖片
            result_filename = f"seg_{image_name}"
            result_path = os.path.join(RESULT_DIR, result_filename)
            imageio.imwrite(result_path, pred_color)

    # 儲存記錄
    save_record(user_id, filename, modelname=modelid, result=chatbot_response)

    return jsonify({
        "result_img_url": f"/{result_path}",
        "text": chatbot_response
    })

# === 測試用網頁 ===
@app.route('/')
def index():
    # return render_template('liff_index.html')
    model_dict = []
    for model in os.listdir(MODEL_DIR):
        model_dict.append(model.split("_")[0])

    return render_template(
        "liff_index.html",
        model_dict=model_dict,
        liff_id=liff_id)

#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=5000, debug=True)

# === Flask Run + ngrok ===
if __name__ == "__main__":
    kill_existing_ngrok()
    config = load_config()

    # 🚀 開啟 ngrok
    public_url = str(ngrok.connect(5000))
    print("🌐 ngrok URL:", public_url)

    # 🛠 更新 LIFF endpoint（選用）
    update_liff_endpoint(config["liff_id"], config["channel_access_token"], f"{public_url}/liff_index")

    # 組成新的 webhook 完整網址（包含路徑）
    new_webhook_url = public_url + "/callback"

    # 更新 LINE webhook URL
    update_line_webhook_url(new_webhook_url)

    '''
    new_url = f"{public_url}/liff_index"
    update_liff_endpoint(
        liff_id=config["liff_id"],
        access_token=config["channel_access_token"],
        new_url=new_url
    )
    '''

    # 可選：自動打開瀏覽器測試頁面
    # webbrowser.open(f"{public_url}/liff_index")
    
    # 如果你要自動顯示這個 ngrok URL，更新到一個檔案
    with open("ngrok_url.txt", "w") as f:
        f.write(str(public_url))

    line_bot_api.push_message(
        user_id,
        TextSendMessage(text=f"模型已準備就緒！請幫我上傳一張口腔照片")
    )

    app.run(host="0.0.0.0", port=config["port"], debug=True)