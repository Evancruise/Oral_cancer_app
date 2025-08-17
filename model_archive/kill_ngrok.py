import os
import signal
import subprocess
import platform

def find_ngrok_pids():
    system = platform.system()
    ngrok_pids = []

    try:
        if system == "Windows":
            # 使用 tasklist 搜索 ngrok
            result = subprocess.check_output("tasklist", shell=True, encoding="utf-8")
            for line in result.splitlines():
                if "ngrok.exe" in line.lower():
                    parts = line.split()
                    pid = int(parts[1])
                    ngrok_pids.append(pid)
        else:
            # macOS / Linux 使用 ps + grep
            result = subprocess.check_output("ps aux | grep ngrok", shell=True, encoding="utf-8")
            for line in result.splitlines():
                if "ngrok" in line and "grep" not in line:
                    parts = line.split()
                    pid = int(parts[1])
                    ngrok_pids.append(pid)
    except Exception as e:
        print(f"發生錯誤：{e}")

    return ngrok_pids

def kill_processes(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)  # 或 signal.SIGKILL 強制關閉
            print(f"已成功結束 ngrok 進程：PID {pid}")
        except Exception as e:
            print(f"無法終止 PID {pid}：{e}")

if __name__ == "__main__":
    pids = find_ngrok_pids()
    if pids:
        print(f"找到 {len(pids)} 個 ngrok 進程，正在結束...")
        kill_processes(pids)
    else:
        print("沒有找到 ngrok 進程。")