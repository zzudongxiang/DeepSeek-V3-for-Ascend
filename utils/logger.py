import os
from datetime import datetime

rank = int(os.getenv("RANK", "0"))
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

try:
    if not os.path.exists("log"):
        os.makedirs("log", exist_ok=True)
    log_file = open(f"log/{log_filename}.log", "w+", encoding="utf8")
except:
    log_file = None
    print(f"Unable to write to log file: {log_filename}")

def log(message):
    log_str = f"[{datetime.now()}] {message}\n"
    if log_file is not None:
        log_file.writelines(log_str)
        log_file.flush()
    print(log_str[:-1])

def log_rank0(message):
    if rank != 0:
        return
    log(message)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:05.02f}s"
    elif minutes > 0:
        return f"0:{minutes:02d}:{seconds:05.02f}s"
    else:
        return f"0:00:{seconds:05.02f}s"
