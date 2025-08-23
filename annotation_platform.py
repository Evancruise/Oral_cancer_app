from flask import Flask, render_template, request, redirect, session, url_for, jsonify, send_from_directory, abort
# from openai_helper import generate_diagnosis
# from openai import OpenAI
# from DINO_v2.inference import infer_single_image_for_web
from model_archive.model import YOLOv9_M4, DINOv2TokenSegmentation
from model_archive.config import Config
from model_archive.main_entry import model_trainvaltest_process
from torchvision import transforms as T
from PIL import Image
# from model_archive.dataset.generate_anno import reshuffle_datasets
# from linebot import LineBotApi, WebhookHandler
# from linebot.exceptions import InvalidSignatureError
# from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage, TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent
import torchvision.ops as ops
import torch.nn.functional as F
import numpy as np
# import uuid
from flask_cors import CORS
import sqlite3
import bcrypt
import os
import json
import csv
import cv2
from flask import make_response
from werkzeug.utils import secure_filename
from google.cloud import storage
import requests
import io
from io import BytesIO
from PIL import Image, ImageDraw
import zipfile
from flask import send_file
from dotenv import load_dotenv
import random
import mlflow
from threading import Thread
from mlflow.tracking import MlflowClient
import shutil
from collections import defaultdict
from init_database_sql import init_db

load_dotenv()

password = os.getenv("PASSWORD")
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
all_config = Config()
device = all_config.device
num_classes = all_config.num_classes
client = all_config.client
class_names = all_config.class_names
img_size = all_config.img_size
save_dir = all_config.save_dir
class_color_map = all_config.class_color_map
optimizer_type = all_config.optimizer_type
scheduler_mode = all_config.scheduler_mode
epochs = all_config.total_epochs
mode = all_config.mode
ml = all_config.model
model_tuning_enable = all_config.model_tuning_enable
tensorboard_enable = all_config.tensorboard_mode_enable
start_epoch = all_config.start_epoch
lr = all_config.lr

IMAGE_DIR = "static/images"
MODEL_DIR = "static/models"
LOG_DIR = "logs"
ANNOT_DIR = "annotations"
RESULT_DIR = "static/results"
UPLOAD_DIR = "static/uploads"
progress_path_train = os.path.join(LOG_DIR, "train_progress.json")
progress_path_inference = os.path.join(LOG_DIR, "inference_progress.json")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

model_list = defaultdict(list)
model_list["yolov9"] = [model_name for model_name in os.listdir(MODEL_DIR) if model_name.startswith("yolov9")]
model_list["mask2former"] = [model_name for model_name in os.listdir(MODEL_DIR) if model_name.startswith("mask2former")]
model_list["unetr_moe"] = [model_name for model_name in os.listdir(MODEL_DIR) if model_name.startswith("unetr")]
model_list["dinov2"] = [model_name for model_name in os.listdir(MODEL_DIR) if model_name.startswith("dinov2")]

training_status = {
    "epoch": 1,
    "total_epochs": epochs,
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

with open(progress_path_train, "w") as f:
    json.dump(training_status, f, indent=2)

inference_status = {
    "current_index": 0,  
    "total_num_imgs": 0,
    "cancel": False,
    "success": False
}

with open(progress_path_inference, "w") as f:
    json.dump(inference_status, f, indent=2)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
CORS(app)

DB_PATH = "users.db"

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/history/clear_records", methods=["POST"])
def clear_records():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM records;")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='records';")  # 重設自增編號
        conn.commit()
        conn.close()
        print("所有紀錄已清除", "success")
    except Exception as e:
        print(f"發生錯誤: {str(e)}", "error")
    return redirect(url_for("history_init"))

def save_record(user_id, filename, result):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO records (user_id, filename, result, timestamp) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (user_id, filename, result)
    )
    conn.commit()
    conn.close()
    
@app.route("/export/<user_id>")
def export_user_csv(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, model, result, timestamp
        FROM records
        WHERE user_id = ?
        ORDER BY timestamp DESC
    """, (user_id,))
    records = cursor.fetchall()
    conn.close()

    output = make_response()
    output.headers["Content-Disposition"] = f"attachment; filename={user_id}_records.csv"
    output.headers["Content-Type"] = "text/csv"

    writer = csv.writer(output.stream)
    writer.writerow(["Filename", "Model", "Result", "Timestamp"])
    for row in records:
        writer.writerow(row)

    return output

@app.route("/history")
def history_init():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, filename, model, result, timestamp
        FROM records
        ORDER BY timestamp DESC
    """)
    records = cursor.fetchall()
    conn.close()

    return render_template("history.html", records=records)

@app.route("/history/<user_id>")
def history_individual(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, model, result, timestamp
        FROM records WHERE user_id=? 
        ORDER BY timestamp DESC
        """, (user_id,))
    
    records = cursor.fetchall()
    conn.close()
    return render_template("history_individual.html", user_id=user_id, records=records)

@app.route("/history/<filename>")
def history_filename(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route("/history/search")
def search_user():
    user_id = request.args.get("user_id")
    if user_id:
        return redirect(url_for('history', user_id=user_id))
    return redirect(url_for('all_history'))

def query_user_latest_record(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, result, timestamp FROM records WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "filename": row[0],
            "result": row[1],
            "timestamp": row[2]
        }
    else:
        return None

init_db()

# AI chatbot API
def generate_prompt_response(detections, class_names, user_input=""):
    det_texts = []
    for x1, y1, x2, y2, conf, cls_id in detections:
        name = class_names[int(cls_id)]
        det_texts.append(f"{name}({conf:.2f}) 座標[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    prompt = (
        f"根據模型偵測結果：{'; '.join(det_texts) or '無偵測'}。\n"
        f"病患問診描述：{user_input or '尚未輸入'}。\n"
        "請給出初步篩檢建議。"
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一位口腔癌篩檢助理醫師，請以專業但溫和口吻提供建議。"},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message["content"]

# Model inference API
@app.route('/')
def entry():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            return redirect('/login')
        except sqlite3.IntegrityError:
            return render_template('register.html', error="使用者已存在")
    return render_template('register.html')

@app.route("/init_setting", methods=['GET', 'POST'])
def init_set():
    if "optimizer" not in session:
        session["optimizer"] = optimizer_type
    if "lr" not in session:
        session["lr"] = float(lr)
    if "scheduler_mode" not in session:
        session["scheduler_mode"] = scheduler_mode
    if "total_epochs" not in session:
        session["total_epochs"] = int(epochs)
    if "step" not in session:
        session["step"] = 0
    if "total_steps" not in session:
        session["total_steps"] = 0
    if "ml" not in session:
        session["ml"] = ml
    if "model_tuning_enable" not in session:
        session["model_tuning_enable"] = model_tuning_enable
    if "tensorboard_enable" not in session:
        session["tensorboard_enable"] = tensorboard_enable
    if "start_epoch" not in session:
        session["start_epoch"] = int(start_epoch)
    if "mode" not in session:
        session["mode"] = mode
    if "start_training" not in session:
        session["start_training"] = False
    if "cancel_training" not in session or session["cancel_training"] == True:
        session["cancel_training"] = False
    if "train_ratio" not in session:
        session["train_ratio"] = 0.7
    if "val_ratio" not in session:
        session["val_ratio"] = 0.2
    if "test_ratio" not in session:
        session["test_ratio"] = 0.1

    client = MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["start_time desc"])

    for run in runs:
        print("Run ID:", run.info.run_id)
        print("Metrics:", run.data.metrics)

    return render_template("model_training.html", 
                           optimizer_type=session["optimizer"],
                           lr=session["lr"],
                           scheduler_mode=session["scheduler_mode"],
                           total_epochs=session["total_epochs"],
                           ml=session["ml"],
                           model_tuning_enable=session["model_tuning_enable"],
                           tensorboard_enable=session["tensorboard_enable"],
                           start_epoch=session["start_epoch"],
                           mode=session["mode"],
                           train_ratio=session["train_ratio"],
                           val_ratio=session["val_ratio"],
                           test_ratio=session["test_ratio"]
                           )

@app.route("/reset_setting", methods=["POST"])
def model_reset():
    if os.path.exists(progress_path_train):
        os.remove(progress_path_train)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False})

@app.route("/model_setting", methods=["POST"])
def model_setting():
    session["optimizer"] = request.form.get("optimizer_type")
    session["lr"] = float(request.form.get("lr"))
    session["scheduler_mode"] = request.form.get("scheduler_mode")
    session["total_epochs"] = int(request.form.get("total_epochs"))
    session["ml"] = request.form.get("ml")
    session["model_tuning_enable"] = request.form.get("model_tuning_enable") == "on"
    session["start_epoch"] = int(request.form.get("start_epoch"))

    apply_train_ratio = float(request.form.get("train_ratio"))
    apply_val_ratio = float(request.form.get("val_ratio"))
    apply_test_ratio = float(request.form.get("test_ratio"))

    #if session["train_ratio"] != apply_train_ratio or session["val_ratio"] != apply_val_ratio or session["test_ratio"] != apply_test_ratio:
    #    reshuffle_datasets(apply_train_ratio, apply_val_ratio)
       
    session["train_ratio"] = float(request.form.get("train_ratio"))
    session["val_ratio"] = float(request.form.get("val_ratio"))
    session["test_ratio"] = float(request.form.get("test_ratio"))

    return jsonify({'optimizer_type': session["optimizer"], 
                    "lr": session["lr"],
                    "scheduler_mode": session["scheduler_mode"],
                    "total_epochs": session["total_epochs"],
                    "ml": session["ml"],
                    "model_tuning_enable": session["model_tuning_enable"],
                    "tensorboard_enable": session["tensorboard_enable"],
                    "start_epoch": session["start_epoch"],
                    "mode": session["mode"],
                    "train_ratio": session["train_ratio"],
                    "val_ratio": session["val_ratio"],
                    "test_ratio": session["test_ratio"]})

@app.route("/model_train", methods=["POST"])
def model_training():
    '''
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["start_time desc"])

    for run in runs:
        print("Run ID:", run.info.run_id)
        print("Metrics:", run.data.metrics)
    '''
    data = request.get_json()
    ml = data.get("ml")
    optimizer_type = data.get("optimizer_type")
    lr = data.get("lr")
    scheduler_mode = data.get("scheduler_mode")
    total_epochs = data.get("total_epochs")
    model_tuning_enable = data.get("model_tuning_enable")
    tensorboard_enable = data.get("tensorboard_enable")
    start_epoch = data.get("start_epoch")
    mode = data.get("mode")
    run_id = data.get("run_id")

    session["start_training"] = True
    session["cancel_training"] = False

    if os.path.exists(progress_path_train):
        with open(progress_path_train) as f:
            training_status = json.load(f)
        return jsonify({"success": True, "epoch": training_status["epoch"], "total_epochs": total_epochs})
    else:
        # if run_id != "":
        #     model = mlflow.pytorch.load_model(f"runs:/{run_id}/{ml}")
        return jsonify({"success": True, "epoch": 1, "total_epochs": total_epochs})

@app.route("/continue_training", methods=["POST"])
def continue_training():
    if os.path.exists(progress_path_train):
        with open(progress_path_train) as f:
            training_status = json.load(f)
        return jsonify({"success": True, "epoch": training_status["epoch"], "total_epochs": training_status["total_epochs"]})
    else:
        # if run_id != "":
        #     model = mlflow.pytorch.load_model(f"runs:/{run_id}/{ml}")
        return jsonify({"success": True, "epoch": 1, "total_epochs": training_status["total_epochs"]})

@app.route("/cancel_training", methods=["POST"])
def cancel_training():
    training_status = None
    session["cancel_training"] = True
    if os.path.exists(progress_path_train):
        with open(progress_path_train) as f:
            training_status = json.load(f)
            training_status["cancel"] = True

        with open(progress_path_train, "w") as f:
            json.dump(training_status, f, indent=2)

    return jsonify({"success": True, "message": "Training cancelled."})

@app.route("/start_training")
def main_training_process():
    '''
    def train_thread():
        # 模擬實際的訓練流程
        model_trainvaltest_process(
            optimizer=session.get("optimizer_type", "adam"),
            lr=session.get("lr", 1e-4),
            scheduler=session.get("scheduler_mode", "cosineanneal"),
            total_epochs=int(session.get("total_epochs", 10)),
            mode=session.get("mode", "train"),
            ml=session.get("ml", "dinov2"),
            model_tuning_enable=session.get("model_tuning_enable", False),
            log_enable=session.get("tensorboard_enable", False),
            start_epoch=int(session.get("start_epoch", 0))
        )

    # 啟動背景訓練
    thread = Thread(target=train_thread)
    thread.start()
    '''

    model_trainvaltest_process(
            optimizer_type=session.get("optimizer_type", "adam"),
            lr=session.get("lr", 1e-4),
            scheduler_mode=session.get("scheduler_mode", "cosineanneal"),
            epochs=int(session.get("total_epochs", 10)),
            mode=session.get("mode", "train"),
            ml=session.get("ml", "dinov2"),
            model_tuning_enable=session.get("model_tuning_enable", False),
            log_enable=session.get("tensorboard_enable", False),
            start_epoch=int(session.get("start_epoch", 0)),
            input_inference_path=None,
            original_dir=UPLOAD_DIR,
            save_dir=save_dir,
            progress_path=progress_path_train
    )
    
    return jsonify({"success": True})

@app.route("/get_training_progress")
def get_progress():
    print("/get_training_progress response...")
    if session["start_training"] is False:
        session["start_training"] = True
        training_status = {
            "epoch": 1,
            "total_epochs": session["total_epochs"],
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
    else:
        # 如果檔案不存在，回傳預設狀態
        if not os.path.exists(progress_path_train):
            training_status = {
                "epoch": 1,
                "total_epochs": session["total_epochs"],
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
        else:
            with open(progress_path_train) as f:
                training_status = json.load(f)

        if session["cancel_training"] == True:
            training_status["cancel"] = True

        if training_status["epoch"] >= training_status["total_epochs"]:
            training_status["finished"] = True

    # 寫回進度檔
    # with open(progress_path, "w") as f:
    #     json.dump(training_status, f, indent=2)

    return jsonify(training_status)

@app.route("/model_test", methods=["GET", "POST"])
def model_test():

    if "inference_complete" not in session:
        session["inference_complete"] = False
    else:
        session["inference_complete"] = False
    
    if "cancel_inference" not in session:
        session["cancel_inference"] = False

    #if "image_name" not in session:
    #    session["image_name"] = os.listdir(UPLOAD_DIR)[0]
    
    if "model_type" not in session:
        session["model_type"] = "yolov9"

    return render_template(
        "model_test.html",
        model_type=session["model_type"],
        model_list=dict(model_list))

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        image_bytes = BytesIO(response.content)
        pil_img = Image.open(image_bytes).convert("RGB")  # PIL Image
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # OpenCV 格式
        return cv_img
    except Exception as e:
        print(f"無法從 URL 載入圖片: {e}")
        return None
    
def download_from_gcs(bucket_name, blob_name, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

def parse_gcs_uri(gcs_uri):
    """
    解析 GCS URI，返回 (bucket_name, blob_path)
    例如：'gs://my-bucket/images/img.jpg' → ('my-bucket', 'images/img.jpg')
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI: must start with 'gs://'")

    no_prefix = gcs_uri[5:]  # 移除 'gs://'
    parts = no_prefix.split('/', 1)

    if len(parts) != 2:
        raise ValueError("Invalid GCS URI: must contain bucket and object path")

    bucket_name, blob_path = parts
    return bucket_name, blob_path

@app.route("/download", methods=["POST"])
def load_image(image_name):
    if image_name.startswith("http"):
        return load_image_from_url(image_name)
    elif image_name.startswith("gs://"):
        # parse GCS URI
        bucket, blob_path = parse_gcs_uri(image_name)
        local_path = os.path.join(UPLOAD_DIR, os.path.basename(blob_path))
        download_from_gcs(bucket, blob_path, local_path)
        return cv2.imread(local_path)
    else:
        return cv2.imread(os.path.join(UPLOAD_DIR, image_name))

def model_inference(image_path, model_name):
    # 模擬：打開原圖，畫紅框
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 100, 100], outline="red", width=5)
    return img

@app.route('/download_results_zip', methods=['GET'])
def download_results_zip():
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for fname in os.listdir(RESULT_DIR):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                fpath = os.path.join(RESULT_DIR, fname)
                zipf.write(fpath, arcname=fname)

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', download_name='results.zip', as_attachment=True)

@app.route("/start_inference")
def start_inference():

    model_name = request.form.get("model_name")
    model_type = request.form.get("model_type")

    if len(os.listdir(RESULT_DIR)) == 0 or not model_name or not model_type:
        return jsonify({"success": False})
    
    session["model_name"] = model_name
    session["model_type"] = model_type
    
    model_trainvaltest_process(
        optimizer_type=session.get("optimizer_type", "adam"),
        lr=session.get("lr", 1e-4),
        scheduler_mode=session.get("scheduler_mode", "cosineanneal"),
        epochs=int(session.get("total_epochs", 10)),
        mode="inference",
        ml=session.get("ml", "dinov2"),
        model_name=model_name,
        model_tuning_enable=session.get("model_tuning_enable", False),
        log_enable=session.get("tensorboard_enable", False),
        start_epoch=int(session.get("start_epoch", 0)),
        input_inference_path=None,
        original_dir=UPLOAD_DIR,
        save_dir=save_dir,
        progress_path=progress_path_inference
    )
    session["inference_complete"] = True
    return jsonify({"success": True})

@app.route("/cancel_inference", methods=["POST"])
def cancel_inference():
    session["cancel_inference"] = True

    if os.path.exists(progress_path_inference):
        with open(progress_path_inference) as f:
            inference_status = json.load(f)
            inference_status["cancel"] = True

        with open(progress_path_inference, "w") as f:
            json.dump(inference_status, f, indent=2)

    return jsonify({"success": True, "message": "Inferencing cancelled."})

@app.route("/inference", methods=["POST"])
def inference():
    model_name = request.form.get("model")
    filenames = request.form.getlist("filenames")  # 這裡取得前端 8 張圖名

    if model_name == "":
        return jsonify({
            "success": False,
            "message": "no model found",
            "filenames": None,
            "result_img_urls": None
        })

    input_path = None

    if len(os.listdir(RESULT_DIR)) == 0:
        return jsonify({
                "success": False,
                "message": "no file found",      
                "filenames": [],
                "result_img_urls": []
            })

    for fname in filenames:
        input_path = os.path.join(UPLOAD_DIR, fname)
        if not os.path.exists(input_path):
            return jsonify({
                "success": False,
                "message": "no file found",      
                "filenames": [],
                "result_img_urls": []
            })

    return jsonify({
            "success": True,   
            "message": "start inferencing",   
            "filenames": [],
            "result_img_urls": []
        })

@app.route("/inferencing")
def inferencing():

    result_urls = []
    filenames_list = []

    with open(progress_path_inference) as f:
        inference_status = json.load(f)

    # 這裡呼叫模型推論
    # result_img = model_inference(input_path, model_name)
    print("session[\"inference_complete\"]:", session["inference_complete"])

    if len(os.listdir(RESULT_DIR)) == 0:
        return jsonify({
                "success": False,
                "message": "no file found", 
                "current_index": inference_status["current_index"],  
                "total_num_imgs": inference_status["total_num_imgs"],
                "filenames": [],
                "result_img_urls": []
            })

    if session["inference_complete"] == True:
        session["inference_complete"] = False
        delete_files_in_folder(UPLOAD_DIR)
        for fname in os.listdir(save_dir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                #fpath = os.path.join(UPLOAD_DIR, fname)
                filenames_list.append(f"/static/uploads/{fname}")
                result_urls.append(f"/static/test_preds/{fname}")

        return jsonify({
            "success": True,
            "message": "inference complete",
            "current_index": inference_status["current_index"],  
            "total_num_imgs": inference_status["total_num_imgs"],
            "filenames": filenames_list,
            "result_img_urls": result_urls
        })
    else:
        return jsonify({
            "success": True,   
            "message": "still inferencing",
            "current_index": inference_status["current_index"],  
            "total_num_imgs": inference_status["total_num_imgs"],
            "filenames": [],
            "result_img_urls": []
        })

# Basic user management
@app.route("/change_password", methods=["GET", "POST"])
def change_password():
    if 'user' not in session:
        return redirect('/login')
    message = ""
    if request.method == 'POST':
        old_pw = request.form['old']
        new_pw = request.form['new']
        username = session['user']
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT password FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if row and bcrypt.checkpw(old_pw.encode(), row[0]):
                new_hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt())
                cur.execute("UPDATE users SET password=? WHERE username=?", (new_hashed, username))
                conn.commit()
                message = "密碼已更新"
            else:
                message = "舊密碼錯誤"
    return render_template('change_password.html', message=message)

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT password FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if row and bcrypt.checkpw(password.encode(), row[0]):
                session['user'] = username
                return redirect('/basic_info')
        return render_template('login.html', error="帳號或密碼錯誤")
    return render_template('login.html')

@app.route('/basic_info')
def basic_info():
    return render_template('basic_info.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route('/annotate')
def annotate():
    if 'user' not in session:
        return redirect('/login')
    return render_template('annotate.html')

@app.route("/models", methods=["GET"])
def list_models():
    try:
        models = [
            "yolov9",
            "mask2former",
            "unetr_moe",
            "dinov2"
        ]
        return jsonify(models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(os.path.join(IMAGE_DIR, file.filename))
    return 'Uploaded!'

@app.route("/upload_one", methods=["POST"])
def upload_image():
    files = request.files.getlist('images')
    filenames = [f for f in os.listdir(IMAGE_DIR) if f.lower().split('.')[1] in ALLOWED_EXTENSIONS]
    annotate_flag = []

    for file in files:
        if file == '':
            continue
        if not file.mimetype.startswith("image/"):
            continue
        filename = secure_filename(file.filename)
        save_path = os.path.join(IMAGE_DIR, filename)
        file.save(save_path)
        filenames.append(filename)

        json_path = os.path.join(ANNOT_DIR, filename.split('.')[0] + ".json")
        if os.path.exists(json_path):
            annotate_flag.append(True)
        else:
            annotate_flag.append(False)

    return jsonify({"success": True, "filenames": filenames, "annotated": annotate_flag})

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

@app.route('/upload_multi', methods=['POST'])
def upload_multi():

    # 預期的部位欄位名稱
    part_codes = ["1", "2", "3", "4", "5", "6", "7", "8"]
    saved_files = []
    result_img_urls = []

    delete_files_in_folder(UPLOAD_DIR)

    for code in part_codes:
        file = request.files.get(code)
        if file and file.filename:
            filename = f"{code}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)
            saved_files.append(filename)
            result_img_urls.append("/static/results/" + filename)

    return jsonify({"filenames": saved_files, "result_img_urls": result_img_urls})

@app.route('/images_upload')
def list_images_upload():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().split('.')[1] in ALLOWED_EXTENSIONS]
    return jsonify(files)

@app.route('/images')
def list_images():
    filenames = [f for f in os.listdir(IMAGE_DIR) if f.lower().split('.')[1] in ALLOWED_EXTENSIONS]
    annotate_flag = []

    for file in filenames:
        if file == '':
            continue

        json_path = os.path.join(ANNOT_DIR, file.split('.')[0] + ".json")
        if os.path.exists(json_path):
            annotate_flag.append(True)
        else:
            annotate_flag.append(False)

    return jsonify({"success": True, "filenames": filenames, "annotated": annotate_flag})

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/image_results')
def list_image_results():
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().split('.')[1] in ALLOWED_EXTENSIONS]
    return jsonify(files)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    fname = data['fname']
    ann = data['annotations']

    for item in ann["polygons"]:
        item['points'] = [[int(round(x)), int(round(y))] for x, y in item['points']]

    with open(os.path.join(ANNOT_DIR, fname.split('.')[0] + ".json"), 'w') as f:
        json.dump(ann, f, indent=2)

    return jsonify({"status": "ok"})

@app.route('/load_annotation/<filename>')
def load_annotation(filename):
    json_path = os.path.join(ANNOT_DIR, filename.split('.')[0] + ".json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)