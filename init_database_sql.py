import sqlite3
import os

DB_PATH = os.getenv("DB_PATH")

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
            qr_session_id TEXT,
            create_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            name TEXT,
            username TEXT UNIQUE,
            password TEXT,
            priority INTEGER,
            status TEXT,
            note TEXT,
            unit TEXT
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
            start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
            start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
