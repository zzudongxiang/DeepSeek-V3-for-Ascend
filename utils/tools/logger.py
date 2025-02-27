import os
from datetime import datetime

rank = int(os.getenv("RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

try:
    if rank == 0:
        if not os.path.exists("log"):
            os.makedirs("log", exist_ok=True)
        log_file = open(f"log/{log_filename}.log", "w+", encoding="utf8")
    else:
        log_file = None
except:
    log_file = None
    print(f"Unable to write to log file: {log_filename}")

def log(message):
    global log_file
    log_str = f"[{datetime.now()}] {message}\n"
    if log_file is not None:
        try:
            log_file.writelines(log_str)
            log_file.flush()
        except:
            print("Failed to write the log file, will stop writing")
            log_file = None
    print(log_str[:-1])

def log_last_rank(message):
    if rank != world_size - 1:
        return
    log(message)

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
