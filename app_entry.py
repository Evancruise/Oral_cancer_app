import os
import io
import base64
import uuid
import psutil
import datetime
import qrcode
import glob
import sqlite3
import json
import threading
from zoneinfo import ZoneInfo
from collections import defaultdict
import redis, jwt, requests
from linebot.exceptions import InvalidSignatureError

from model_archive.model import YOLOv9_M4, DINOv2TokenSegmentation
from model_archive.main_entry import model_trainvaltest_process
from model_archive.config import Config
from model_archive.utils_func import delete_files_in_folder

from werkzeug.utils import secure_filename
from pyngrok import ngrok
from flask import Flask, render_template, request, redirect, session, url_for, jsonify, send_from_directory, abort
from linebot import LineBotApi, WebhookHandler
from flask_cors import CORS
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage, 
    TemplateSendMessage, ButtonsTemplate, PostbackAction, PostbackEvent,
    URIAction, ButtonsTemplate
)

def kill_existing_ngrok():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'ngrok' in proc.info['name'].lower():
                print(f"殺掉舊的 ngrok：PID {proc.info['pid']}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

# === Initialization ===
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

all_config = Config()
save_dir = all_config.save_dir

r = redis.Redis(host="172.20.48.1", port=6379, db=0, decode_responses=True)

JWT_SECRET = os.getenv("JWT_SECRET")
QR_SESSION_TTL = 300 # 5 minutes

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_BOT_ID = os.getenv("LINE_BOT_ID")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
USER_ID = os.getenv("USER_ID")
TIMEZONE = os.getenv("TIMEZONE")

LOG_DIR = "logs"

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
DB_PATH = os.getenv("DB_PATH")
progress_path = os.path.join(LOG_DIR, "train_progress.json")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
parts = ["pic1", "pic2", "pic3", "pic4", "pic5", "pic6", "pic7", "pic8"]
IMAGE_DIR = "static/images"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # 建立 users 表
    cursor.execute('''
        DROP TABLE IF EXISTS users;
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL
        )
    ''')

    # 建立 records 表
    cursor.execute('''
        DROP TABLE IF EXISTS records;
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            patient_id TEXT NOT NULL,
            result TEXT,
            notes TEXT,
            status TEXT,
            progress INT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            img1 TEXT,
            img2 TEXT,
            img3 TEXT,
            img4 TEXT,
            img5 TEXT,
            img6 TEXT,
            img7 TEXT,
            img8 TEXT,
            img1_result TEXT,
            img2_result TEXT,
            img3_result TEXT,
            img4_result TEXT,
            img5_result TEXT,
            img6_result TEXT,
            img7_result TEXT,
            img8_result TEXT
        )
    ''')

    # 建立 records 表 (垃圾桶)
    cursor.execute('''
        DROP TABLE IF EXISTS records_gb;
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records_gb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            patient_id TEXT NOT NULL,
            result TEXT,
            notes TEXT,
            status TEXT,
            progress INT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            img1 TEXT,
            img2 TEXT,
            img3 TEXT,
            img4 TEXT,
            img5 TEXT,
            img6 TEXT,
            img7 TEXT,
            img8 TEXT,
            img1_result TEXT,
            img2_result TEXT,
            img3_result TEXT,
            img4_result TEXT,
            img5_result TEXT,
            img6_result TEXT,
            img7_result TEXT,
            img8_result TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()

'''
@app.route("/record/view/<record_id>")
def view_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM records WHERE user_id=?", (record_id,))
    row = cursor.fetchone()
    if row is None:
        abort(404)
        
    return render_template("record_view.html", record=dict(row))
'''

def check_db_table():
    # 假設你的資料庫檔案是 records.db
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查詢表格的 schema 資訊
    cursor.execute("PRAGMA table_info(records)")
    columns = cursor.fetchall()

    # 顯示表格欄位資訊
    print("欄位資訊：")
    print(f"{'cid':<3} {'name':<10} {'type':<10} {'notnull':<8} {'dflt_value':<20} {'pk':<3}")
    for col in columns:
        cid, name, col_type, notnull, dflt_value, pk = col
        print(f"{cid:<3} {name:<10} {col_type:<10} {notnull:<8} {str(dflt_value):<20} {pk:<3}")

    conn.close()

def file_reload(patient_id):

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT img1, img2, img3, img4, img5, img6, img7, img8 FROM records
        WHERE patient_id = ?;
    """, (patient_id, ))
    rows = cursor.fetchone()

    conn.close()

    if not rows:
        return None
    
    saved_paths = {}

    for i in range(1, 9):
        key = f"pic{i}"
        saved_paths[key] = rows[f"img{i}"]
    
    return saved_paths

def extract_uploaded_images(files, patient_id):
    print("files:", files)

    # 過濾有效的 FileStorage
    uploaded_files = {
        key: file
        for key, file in files.items()
        if getattr(file, "filename", "").strip()
    }

    print("uploaded_files:", uploaded_files)

    if uploaded_files:
        # 回傳檔名字典（統一輸出格式）
        saved_paths = {}

        os.makedirs(f"{UPLOAD_DIR}/{patient_id}", exist_ok=True)
        delete_files_in_folder(f"{UPLOAD_DIR}/{patient_id}")

        for i in range(1, 9):
            key = f"pic{i}"
            file = uploaded_files.get(key)
            
            if file and file.filename:
                filename = secure_filename(file.filename)  # 確保安全檔名
                save_path = f"{UPLOAD_DIR}/{patient_id}/{filename}"
                file.save(save_path)
                saved_paths[key] = save_path
            else:
                return None
        
        if len(os.listdir(f"{UPLOAD_DIR}/{patient_id}")) != 8:
            return None
        
        print("saved_paths1:", saved_paths)
        return saved_paths
    else:
        # 回傳資料庫中已存的檔案名稱（統一輸出格式）
        saved_paths = file_reload(patient_id)
        print("saved_paths2:", saved_paths)
        return saved_paths

@app.route("/record/edit/<record_id>", methods=["POST"])
def edit_record(record_id):

    if record_id != session["user_id"]:
        return jsonify({"status": "failed", "message": "request token changed", "redirect": url_for("history")})

    form = request.form
    files = request.files

    action = form.get("action")
    print("action:", action)
    print("files:", files)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if action == "save":
        
        image_paths = extract_uploaded_images(files, form['patient_id'])

        print("image_paths1:", image_paths)

        if image_paths and check_image_upload(files):
            print("insert include image_paths")

            print("image_paths[\"pic1\"]:", image_paths["pic1"])
            print("image_paths[\"pic2\"]:", image_paths["pic2"])
            print("image_paths[\"pic3\"]:", image_paths["pic3"])
            print("image_paths[\"pic4\"]:", image_paths["pic4"])
            print("image_paths[\"pic5\"]:", image_paths["pic5"])
            print("image_paths[\"pic6\"]:", image_paths["pic6"])
            print("image_paths[\"pic7\"]:", image_paths["pic7"])
            print("image_paths[\"pic8\"]:", image_paths["pic8"])

            cursor.execute("""
                UPDATE records
                SET name=?, gender=?, age=?, notes=?, timestamp=CURRENT_TIMESTAMP, 
                img1=?, img2=?, img3=?, img4=?, img5=?, img6=?, img7=?, img8=?
                WHERE patient_id=?
            """, (form['name'], form['gender'], int(form['age']), form['notes'], 
                image_paths["pic1"], image_paths["pic2"], image_paths["pic3"], image_paths["pic4"], 
                image_paths["pic5"], image_paths["pic6"], image_paths["pic7"], image_paths["pic8"], form['patient_id'],))
        else:
            print("insert not include image_paths")
            cursor.execute("""
                UPDATE records
                SET name=?, gender=?, age=?, notes=?, timestamp=CURRENT_TIMESTAMP
                WHERE patient_id=?
            """, (form['name'], form['gender'], int(form['age']), form['notes'], form['patient_id'],))

        conn.commit()

        check_db_table()

        conn.close()
        
        return jsonify({"status": "ok", "redirect": url_for("history")})
    
    elif action == "infer":

        image_paths = extract_uploaded_images(files, form['patient_id'])

        if not image_paths:
            return jsonify({"status": "failed", "message": "No image found","redirect": url_for("history")})

        cursor.execute("""
            UPDATE records
            SET name=?, gender=?, age=?, notes=?, timestamp=CURRENT_TIMESTAMP, img1=?, img2=?, img3=?, img4=?, img5=?, img6=?, img7=?, img8=?, status=?, progress=?
            WHERE patient_id=?
        """, (form['name'], form['gender'], int(form['age']), form['notes'], 
              image_paths["pic1"], image_paths["pic2"], image_paths["pic3"], image_paths["pic4"], 
              image_paths["pic5"], image_paths["pic6"], image_paths["pic7"], image_paths["pic8"], form['patient_id'], "not_started", 0,))
        
        conn.commit()

        check_db_table()

        conn.close()

        return jsonify({"status": "ok", "redirect": url_for("history")})

@app.route("/record/resume_delete/<record_id>/<patient_id>", methods=["POST"])
def resume_delete_record(record_id, patient_id):
    if record_id != session["user_id"]:
        return jsonify({"status": "failed", "message": "request token changed", "redirect": url_for("history")})

    form = request.form

    action = form.get("action")
    print("patient_id:", patient_id)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # cursor.execute("DELETE FROM records WHERE patient_id = ?", (patient_id,))
    # conn.commit()
    if action == "resume":
        cursor.execute("""
            INSERT INTO records (
                name, gender, age, patient_id, result, notes, status, progress, message, timestamp,
                img1, img2, img3, img4, img5, img6, img7, img8,
                img1_result, img2_result, img3_result, img4_result, img5_result, img6_result, img7_result, img8_result
            )
            SELECT
                name, gender, age, patient_id, result, notes, status, progress, message, timestamp,
                img1, img2, img3, img4, img5, img6, img7, img8,
                img1_result, img2_result, img3_result, img4_result, img5_result, img6_result, img7_result, img8_result
            FROM records_gb
            WHERE patient_id = ?;
        """, (patient_id, ))

        conn.commit()

        cursor.execute("SELECT * FROM records WHERE patient_id = ?", (patient_id,))
        conn.commit()
        row = cursor.fetchone()

        cursor.execute("DELETE FROM records_gb WHERE patient_id = ?", (patient_id,))
        conn.commit()

        cursor.execute("SELECT * FROM records_gb WHERE patient_id = ?", (patient_id,))
        conn.commit()
        row_gb = cursor.fetchone()

        conn.close()

        if row and not row_gb:
            return jsonify({"status": "ok", "redirect": url_for("discard_history")})
        else:
            return jsonify({"status": "failed", "message": "resume failed", "redirect": url_for("discard_history")})

    elif action == "delete_confirm":

        cursor.execute("DELETE FROM records WHERE patient_id = ?", (patient_id,))
        conn.commit()

        cursor.execute("DELETE FROM records_gb WHERE patient_id = ?", (patient_id,))
        conn.commit()

        cursor.execute("SELECT * FROM records_gb WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        conn.close()

        delete_files_in_folder(f"{UPLOAD_DIR}/{patient_id}")

        if row:
            return jsonify({"status": "failed", "message": "delete failed", "redirect": url_for("discard_history")})
        else:
            return jsonify({"status": "ok", "redirect": url_for("discard_history")})

@app.route("/record/delete_confirm/<record_id>/<patient_id>", methods=["POST"])
def delete_record_confirm(record_id, patient_id):
    if record_id != session["user_id"]:
        return jsonify({"status": "failed", "message": "request token changed", "redirect": url_for("history")})
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("DELETE FROM records_gb WHERE patient_id = ?", (patient_id,))
    conn.commit()

    cursor.execute("SELECT * FROM records_gb WHERE patient_id = ?", (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    if rows:
        return jsonify({"status": "failed", "message": "delete failed", "redirect": url_for("history")})
    else:
        return jsonify({"status": "ok", "redirect": url_for("history")})

@app.route("/record/delete/<record_id>/<patient_id>", methods=["POST"])
def delete_record(record_id, patient_id):

    if record_id != session["user_id"]:
        return jsonify({"status": "failed", "message": "request token changed", "redirect": url_for("history")})

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # cursor.execute("DELETE FROM records WHERE patient_id = ?", (patient_id,))
    # conn.commit()

    cursor.execute("""
        INSERT INTO records_gb (
            name, gender, age, patient_id, result, notes, status, progress, message, timestamp,
            img1, img2, img3, img4, img5, img6, img7, img8,
            img1_result, img2_result, img3_result, img4_result, img5_result, img6_result, img7_result, img8_result
        )
        SELECT
            name, gender, age, patient_id, result, notes, status, progress, message, timestamp,
            img1, img2, img3, img4, img5, img6, img7, img8,
            img1_result, img2_result, img3_result, img4_result, img5_result, img6_result, img7_result, img8_result
        FROM records
        WHERE patient_id = ?;
    """, (patient_id, ))
    conn.commit()

    cursor.execute("DELETE FROM records WHERE patient_id = ?", (patient_id,))
    conn.commit()

    cursor.execute("SELECT * FROM records WHERE patient_id = ?", (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    if rows:
        return jsonify({"status": "failed", "message": "delete failed", "redirect": url_for("history")})
    else:
        return jsonify({"status": "ok", "redirect": url_for("history")})

def check_image_upload(files):
    for i in range(1, 9):
        key = f'pic{i}'
        file = files.get(key)
        if not (file and file.filename):
            return False
    return True

@app.route("/record/retrieve_results/<patient_id>", methods=["GET"])
def retrieve_result(patient_id):
    record_dict = {}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT img1_result, img2_result, img3_result, img4_result,
           img5_result, img6_result, img7_result, img8_result FROM records 
           WHERE patient_id = ?
    """, (patient_id,))

    rows = cursor.fetchone()
    conn.close()

    print('rows:', dict(rows))

    for i in range(1, 9):
        result = rows[f"img{i}_result"]
        if not result:
            return jsonify({"message": "not found", "record_dict": None})
        record_dict[f"img{i}"] = result

    return jsonify({"message": "exist", "record_dict": record_dict})

@app.route("/record/new/<record_id>", methods=["POST"])
def new_record(record_id):

    if record_id != session["user_id"]:
        return jsonify({"status": "failed", "message": "request token changed", "redirect": url_for("history")})

    form = request.form
    files = request.files

    action = form.get("action")
    print("action:", action)
    print("files:", files)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM records WHERE patient_id = ?", (form['patient_id'],))
    existing = cursor.fetchone()

    if action == "create":

        if existing:
            print("已存在這個 patient_id，請確認是否重複新增")
            return jsonify({"status": "failed", "message": "patient id existed", "redirect": url_for("history")})
        
        image_paths = extract_uploaded_images(files, form['patient_id'])

        print("image_paths2:", image_paths)

        if image_paths and check_image_upload(files):

            print("image_paths[\"pic1\"]:", image_paths["pic1"])
            print("image_paths[\"pic2\"]:", image_paths["pic2"])
            print("image_paths[\"pic3\"]:", image_paths["pic3"])
            print("image_paths[\"pic4\"]:", image_paths["pic4"])
            print("image_paths[\"pic5\"]:", image_paths["pic5"])
            print("image_paths[\"pic6\"]:", image_paths["pic6"])
            print("image_paths[\"pic7\"]:", image_paths["pic7"])
            print("image_paths[\"pic8\"]:", image_paths["pic8"])

            print("insert include image_paths")
            cursor.execute("""
                INSERT INTO records (
                    name, gender, age, patient_id, notes, 
                    timestamp, img1, img2, img3, img4, img5, img6, img7, img8, status, progress
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                form['name'], form['gender'], int(form['age']), form['patient_id'], form['notes'],
                image_paths["pic1"], image_paths["pic2"], image_paths["pic3"], image_paths["pic4"],
                image_paths["pic5"], image_paths["pic6"], image_paths["pic7"], image_paths["pic8"], "not_started", 0,
            ))
        else:
            print("insert not include image_paths")
            cursor.execute("""
                INSERT INTO records (
                    name, gender, age, patient_id, notes, timestamp, status, progress
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """, (form['name'], form['gender'], int(form['age']), form['patient_id'], form['notes'], "not_started", 0,))

        conn.commit()

        check_db_table()

        conn.close()

        return jsonify({"status": "ok", "redirect": url_for("history")})

    elif action == "infer":

        image_paths = extract_uploaded_images(files, form['patient_id'])

        if not image_paths:
            return jsonify({"status": "failed", "message": "No image found","redirect": url_for("history")})

        cursor.execute("""
                INSERT INTO records (
                    name, gender, age, patient_id, notes, 
                    timestamp, img1, img2, img3, img4, img5, img6, img7, img8, status, progress
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                form['name'], form['gender'], int(form['age']), form['patient_id'], form['notes'],
                image_paths["pic1"], image_paths["pic2"], image_paths["pic3"], image_paths["pic4"],
                image_paths["pic5"], image_paths["pic6"], image_paths["pic7"], image_paths["pic8"], "not_started", 0,
            ))

        conn.commit()

        check_db_table()

        conn.close()

        return jsonify({"status": "ok", "redirect": url_for("history")})

@app.route("/record/result/<record_id>", methods=["POST"])
def upload_images(record_id):
    if record_id != session["user_id"]:
        return jsonify({"status": "failed", "message": "request token changed"})

    # action = request.form.get("action")
    # return redirect(url_for("history", status="ok", message="start uploading"))

    return jsonify({"status": "ok", "message": "start uploading"})

@app.route("/record/upload_result/<patient_id>", methods=["GET"])
def retrieve_upload_status(patient_id):
    # return redirect(url_for("history", patient_id=patient_id))
    return jsonify({"status": "done", "redirect": url_for("history")})

@app.route("/record/cancel_inference/<patient_id>", methods=["POST"])
def cancel_inference(patient_id):

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE records
        SET status=?, progress=?
        WHERE patient_id=?
    """, ("canceled", 0, patient_id,))

    conn.commit()
    conn.close()

    return jsonify({"status": "done", "message": "Inference cancelled."})

@app.route("/record/trigger_infer_process/<patient_id>")
def inference_process(patient_id):
    # ✅ 先抓資料，在主 request context 內
    optimizer_type = session.get("optimizer_type", "adam")
    lr = session.get("lr", 1e-4)
    scheduler_mode = session.get("scheduler_mode", "cosineanneal")
    epochs = int(session.get("total_epochs", 10))
    ml = session.get("ml", "dinov2")
    model_tuning_enable = session.get("model_tuning_enable", False)
    log_enable = session.get("tensorboard_enable", False)
    start_epoch = int(session.get("start_epoch", 0))
    input_inference_path = sorted(glob.glob(f"{UPLOAD_DIR}/{patient_id}/*.png"))
    save_dir = f"{RESULT_DIR}/{patient_id}"

    def run_inference():
        model_trainvaltest_process(
            optimizer_type=optimizer_type,
            lr=lr,
            scheduler_mode=scheduler_mode,
            epochs=epochs,
            mode="inference",
            ml=ml,
            model_tuning_enable=model_tuning_enable,
            log_enable=log_enable,
            start_epoch=start_epoch,
            input_inference_path=input_inference_path,
            save_dir=save_dir,
            progress_path=None,
            patient_id=patient_id,
            db_path=DB_PATH
        )

    # 背景執行
    thread = threading.Thread(target=run_inference)
    thread.start()

    return '', 202

@app.route("/record/start_inference/<patient_id>", methods=["POST"])
def start_infer(patient_id):
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    '''
    cursor.execute("""
        SELECT img1, img2, img3, img4, img5, img6, img7, img8 FROM records
	    WHERE patient_id = ?;
    """, (patient_id,))

    row = cursor.fetchone()
    '''

    cursor.execute("""
        UPDATE records
        SET status=?, progress=?
        WHERE patient_id=?
    """, ("startup", 0, patient_id,))

    '''
    upload_images = []
    for i in range(1, 9):
        upload_images.append(row['img' + str(i+1)])
    '''

    os.makedirs(f"{UPLOAD_DIR}/{patient_id}", exist_ok=True)
    os.makedirs(f"{RESULT_DIR}/{patient_id}", exist_ok=True)

    conn.close()

    return jsonify({"status": "done"})

@app.route("/record/check_inference/<patient_id>", methods=["GET"])
def check_inference(patient_id):
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT status, progress FROM records
	    WHERE patient_id = ?;
    """, (patient_id,))

    row = cursor.fetchone()
    conn.commit()
    conn.close()

    if not row:
        return jsonify({
            "status": "unknown",
            "progress": 0,
            "message": f"無法讀取狀態"
        })
    
    status = row["status"]     # "running"
    progress = row["progress"]
    
    if status == "not_started":
        return jsonify({
            "status": status,
            "progress": progress,
            "message": "尚未開始推論"
        })
    elif status == "startup":
        return jsonify({
            "status": status,
            "progress": progress,
            "message": "正開始推論"
        })
    elif status == "in_progress":
        return jsonify({
            "status": status,
            "progress": progress,
            "message": "推論進行中"
        })
    elif status == "done":
        return jsonify({
            "status": status,
            "progress": progress,
            "message": "推論已完成"
        })
    elif status == "error":
        return jsonify({
            "status": status,
            "progress": progress,
            "message": "推論發生錯誤"
        })
    else:
        return jsonify({
            "status": "unknown",
            "progress": 0,
            "message": f"未知狀態：{status}"
        })

@app.route('/discard_history')
def discard_history():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    patient_id_dict = []

    cursor.execute("SELECT * FROM records_gb ORDER BY timestamp DESC")
    rows = cursor.fetchall()

    grouped = defaultdict(list)
    for row in rows:
        dt_naive = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
        dt_utc = dt_naive.replace(tzinfo=ZoneInfo("UTC"))
        dt = dt_utc.astimezone(ZoneInfo(TIMEZONE))

        datetime_str = dt.strftime("%Y-%m-%d")  # 精準到秒
        weekday_str = dt.strftime("%A")

        weekday_map = {
            'Monday': '一',
            'Tuesday': '二',
            'Wednesday': '三',
            'Thursday': '四',
            'Friday': '五',
            'Saturday': '六',
            'Sunday': '日',
        }

        chinese_weekday = weekday_map.get(weekday_str, '')
        date_display = f"{datetime_str}（{chinese_weekday}）"

        current_table = {
            'name': row['name'],
            'gender': row['gender'],
            'age': row['age'],
            'patient_id': row['patient_id'],
            'time': row['timestamp'],  # 原始 timestamp 可用於排序
            'notes': row['notes'],
            'icon': 'camera',
            'bg_class': '',
            'pic1': row['img1'] or "/static/guide/1.png",
            'pic2': row['img2'] or "/static/guide/2.png",
            'pic3': row['img3'] or "/static/guide/3.png",
            'pic4': row['img4'] or "/static/guide/4.png",
            'pic5': row['img5'] or "/static/guide/5.png",
            'pic6': row['img6'] or "/static/guide/6.png",
            'pic7': row['img7'] or "/static/guide/7.png",
            'pic8': row['img8'] or "/static/guide/8.png",
            'pic1_r': row['img1_result'] or "/static/guide/1.png",
            'pic2_r': row['img2_result'] or "/static/guide/2.png",
            'pic3_r': row['img3_result'] or "/static/guide/3.png",
            'pic4_r': row['img4_result'] or "/static/guide/4.png",
            'pic5_r': row['img5_result'] or "/static/guide/5.png",
            'pic6_r': row['img6_result'] or "/static/guide/6.png",
            'pic7_r': row['img7_result'] or "/static/guide/7.png",
            'pic8_r': row['img8_result'] or "/static/guide/8.png"
        }

        print("current_table:", current_table)

        grouped[date_display].append(current_table)
        patient_id_dict.append(row['patient_id'])
    
    if "user_id" not in session:
        return redirect(url_for('index'))

    print("dict(grouped):", dict(grouped))

    return render_template("liff_discard_records.html", grouped_records=dict(grouped), user_id=session["user_id"], all_patient_ids=patient_id_dict)

@app.route('/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    patient_id_dict = []

    cursor.execute("SELECT * FROM records ORDER BY timestamp DESC")
    rows = cursor.fetchall()

    grouped = defaultdict(list)
    for row in rows:
        dt_naive = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
        dt_utc = dt_naive.replace(tzinfo=ZoneInfo("UTC"))
        dt = dt_utc.astimezone(ZoneInfo(TIMEZONE))

        datetime_str = dt.strftime("%Y-%m-%d")  # 精準到秒
        weekday_str = dt.strftime("%A")

        weekday_map = {
            'Monday': '一',
            'Tuesday': '二',
            'Wednesday': '三',
            'Thursday': '四',
            'Friday': '五',
            'Saturday': '六',
            'Sunday': '日',
        }

        chinese_weekday = weekday_map.get(weekday_str, '')
        date_display = f"{datetime_str}（{chinese_weekday}）"

        print("row[\"img1\"]:", row['img1'])
        print("row[\"img2\"]:", row['img2'])
        print("row[\"img3\"]:", row['img3'])
        print("row[\"img4\"]:", row['img4'])
        print("row[\"img5\"]:", row['img5'])
        print("row[\"img6\"]:", row['img6'])
        print("row[\"img7\"]:", row['img7'])
        print("row[\"img8\"]:", row['img8'])

        current_table = {
            'name': row['name'],
            'gender': row['gender'],
            'age': row['age'],
            'patient_id': row['patient_id'],
            'time': row['timestamp'],  # 原始 timestamp 可用於排序
            'notes': row['notes'],
            'icon': 'camera',
            'bg_class': '',
            'pic1': row['img1'] or "/static/guide/1.png",
            'pic2': row['img2'] or "/static/guide/2.png",
            'pic3': row['img3'] or "/static/guide/3.png",
            'pic4': row['img4'] or "/static/guide/4.png",
            'pic5': row['img5'] or "/static/guide/5.png",
            'pic6': row['img6'] or "/static/guide/6.png",
            'pic7': row['img7'] or "/static/guide/7.png",
            'pic8': row['img8'] or "/static/guide/8.png",
            'pic1_r': row['img1_result'] or "/static/guide/1.png",
            'pic2_r': row['img2_result'] or "/static/guide/2.png",
            'pic3_r': row['img3_result'] or "/static/guide/3.png",
            'pic4_r': row['img4_result'] or "/static/guide/4.png",
            'pic5_r': row['img5_result'] or "/static/guide/5.png",
            'pic6_r': row['img6_result'] or "/static/guide/6.png",
            'pic7_r': row['img7_result'] or "/static/guide/7.png",
            'pic8_r': row['img8_result'] or "/static/guide/8.png"
        }

        print("current_table:", current_table)

        grouped[date_display].append(current_table)
        patient_id_dict.append(row['patient_id'])
    
    if "user_id" not in session:
        return redirect(url_for('index'))

    print("dict(grouped):", dict(grouped))

    return render_template("liff_records.html", grouped_records=dict(grouped), user_id=session["user_id"], all_patient_ids=patient_id_dict)

qr_sessions = {}

# === JWT generator ===
def generate_jwt(user_id):
    payload = {
        'user_id': USER_ID,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

# === callback notification ===
def notify_callback(session_id, source):
    callback_url = r.get(f"qr:{session_id}:callback")
    if callback_url:
        try:
            requests.post(callback_url, json={
                'session_id': session_id,
                'source': source,
                'status': 'scanned'
            })
        except Exception as e:
            print("Callback failed:", e)

# === API: Generate session ===
@app.route("/api/gen-session")
def gen_session():
    try:
        session_id = str(uuid.uuid4())
        login_url = url_for('qr_login', session_id=session_id, _external=True)

        r.hmset(f"qr:{session_id}", {
            'status': 'pending',
            'user_id': USER_ID,
            'source': 'web'
        })
        r.expire(f"qr:{session_id}", QR_SESSION_TTL)

        img = qrcode.make(login_url)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            'session_id': session_id,
            'login_url': login_url,
            'qr_base64': img_b64
        })
    
    except Exception as e:
        print("Redis exception:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/qr_login/<session_id>', methods=['GET', 'POST'])
def qr_login(session_id):
    # 模擬手機掃描 + 點擊確認登入流程
    if request.method == 'GET':
        return render_template('scan_confirm.html', session_id=session_id)
    
    elif request.method == 'POST':
        if session_id in qr_sessions:
            qr_sessions[session_id]['status'] = 'scanned'
            qr_sessions[session_id]['user_id'] = request.form.get("user_id", "anonymous")
            qr_sessions[session_id]['source'] = request.form.get("source", "unknown")
            return jsonify({'success': True})
        return jsonify({'success': False})

@app.route("/uploads/<user_id>/<filename>")
def get_uploaded_file(user_id, filename):
    folder = os.path.join(app.root_path, 'static/uploads', user_id)
    return send_from_directory(folder, filename)

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return f'Welcome, {session["user"]}!'
    return redirect(url_for('index'))

# ===== LINE Bot webhook =====
@app.route("/line/webhook", methods=['POST'])
def line_webhook():
    signature = request.headers['X-line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return 'Invalid Signature', 400

    return 'OK'

# === Web UI 顯示 QR code + 輪詢登入狀態 ===
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 處理帳密登入
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password == PASSWORD:
            if 'user_id' not in session or session['user_id'] != USER_ID:
                session['user_id'] = USER_ID

            token = generate_jwt(username)
            return render_template("liff_login.html", user=username, token=token)
        
        return render_template("liff_index.html", error="登入失敗")

    # 原本 QR code 流程
    '''
    session_id = str(uuid.uuid4())
    if session_id not in session:
        session["user_id"] = session_id
    
    login_url = url_for('qr_login', session_id=session_id, _external=True)
    r.hmset(f"qr:{session_id}", {
        'status': 'pending',
        'user_id': '',
        'source': 'web'
    })
    r.expire(f"qr:{session_id}", 300)

    img = qrcode.make(login_url)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return render_template("liff_index.html", session_id=session_id, qr=img_b64)
    '''
    session["user_id"] = USER_ID

    return render_template("liff_index.html")

@app.route("/logout")
def logout():
    return redirect(url_for("index"))

@app.route("/login", methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == '1234':
            token = generate_jwt(username)  # 產生 JWT
            return render_template("liff_login.html", user=username, token=token)
        return render_template("liff_login.html", error="登入失敗")
    return render_template("liff_login.html")

@app.route("/api/set-callback/<session_id>", methods=['POST'])
def set_callback(session_id):
    url = request.json.get("callback_url")
    r.set(f"qr:{session_id}:callback", url, ex=QR_SESSION_TTL)
    return jsonify({'success': True})

@app.route("/infer_entry")
def infer_page():
    return render_template("liff_infer_entry.html")

# === 處理訊息事件 ===
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    if text.startswith('session_id='):
        session_id = text.replace('session_id=', '').strip()
        if session_id in qr_sessions:
            qr_sessions[session_id]['status'] = 'scanned'
            qr_sessions[session_id]['user_id'] = USER_ID

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='掃碼成功')
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextMessage(text='無效的 QR code')
            )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='請掃描電腦端的 QR code 並點擊後自動登入')
        )

# === Flask Run + ngrok ===
if __name__ == "__main__":
    '''
    kill_existing_ngrok()
    config = load_config()

    # 開啟 ngrok
    public_url = str(ngrok.connect(5000))
    print("ngrok URL:", public_url)

    # 🛠 更新 LIFF endpoint（選用）
    update_liff_endpoint(config["liff_id"], config["channel_access_token"], f"{public_url}/liff_index")

    # 組成新的 webhook 完整網址（包含路徑）
    new_webhook_url = public_url + "/callback"

    # 更新 LINE webhook URL
    update_line_webhook_url(new_webhook_url)

    # 可選：自動打開瀏覽器測試頁面
    # webbrowser.open(f"{public_url}/liff_index")
    
    # 如果你要自動顯示這個 ngrok URL，更新到一個檔案
    with open("ngrok_url.txt", "w") as f:
        f.write(str(public_url))

    line_bot_api.push_message(
        user_id,
        TextSendMessage(text=f"模型已準備就緒！請幫我上傳一張口腔照片")
    )
    '''
    app.run(host="0.0.0.0", port=5000, debug=True)