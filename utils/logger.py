import os
from datetime import datetime

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

def log(message):
    print(f"[{datetime.now()}] {message}")

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
