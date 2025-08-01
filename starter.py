import json, os, webbrowser
from pyngrok import ngrok
from linebot import LineBotApi
from linebot.models import TextSendMessage
from flask import Flask
from threading import Thread
from dotenv import load_dotenv

# === Initialization ===
load_dotenv()
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
channel_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
liff_id = os.getenv("LIFF_ID")
user_id = os.getenv("USER_ID")
channel_id = os.getenv("CHANNEL_ID")

# ✅ 載入設定
def load_config():
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        config = {
            "channel_access_token": channel_token,
            "liff_id": liff_id,
            "user_id": user_id,
            "channel_id": channel_id,
            "port": 5000,
        }
    return config

# ✅ 更新 LIFF Endpoint（可略過只做通知）
def update_liff_endpoint(liff_id, access_token, new_url):
    import requests
    url = f"https://api.line.me/liff/v1/apps/{liff_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    body = {
        "view": {
            "type": "full",
            "url": new_url
        }
    }
    r = requests.put(url, headers=headers, json=body)
    if r.status_code == 200:
        print(f"✅ LIFF endpoint 已更新為: {new_url}")
    else:
        print("❌ 更新 LIFF endpoint 失敗:", r.text)

# ✅ 啟動 Flask App
def run_flask():
    os.system("python app.py")

if __name__ == "__main__":
    config = load_config()

    # 🧼 清除舊 ngrok
    ngrok.kill()

    # 🚀 開啟 ngrok
    public_url = str(ngrok.connect(5000))
    print("🌐 ngrok URL:", public_url)

    # 🛠 更新 LIFF endpoint（選用）
    update_liff_endpoint(config["liff_id"], config["channel_access_token"], f"{public_url}/liff_index")

    # ✉️ 發送 LINE 訊息
    bot = LineBotApi(config["channel_access_token"])
    bot.push_message(
        config["user_id"],
        TextSendMessage(text=f"🚀 AI LIFF App 啟動成功，點這裡進入診斷 👉 {public_url}/liff_index")
    )

    # 🌍 開啟瀏覽器
    webbrowser.open(f"{public_url}/liff_index")

    # ▶️ 啟動 Flask
    t = Thread(target=run_flask)
    t.start()
