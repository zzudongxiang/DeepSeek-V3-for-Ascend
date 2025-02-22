import os
import threading
from datetime import datetime
from dataclasses import dataclass
from utils.logger import log_rank0, format_time

rank = int(os.getenv("RANK", "0"))

@dataclass
class ThreadTokens:
    thread: threading.Thread
    stop_event: threading.Event

def print_progress(stop_event, read_progress_func, description="", interval=10):
    t0 = datetime.now()
    now = datetime(1970, 1, 1)
    while not stop_event.is_set():
        if (datetime.now() - now).total_seconds() > interval:
            now = datetime.now()
            elapsed = (datetime.now() - t0).total_seconds()
            progress = read_progress_func()
            progress = progress if progress < 1 else 1
            if progress > 0:
                eta = (elapsed / progress) * (1 - progress)
                eta = eta if eta < 360060 else 360060
                log_rank0(f"{description}: {(progress * 100):05.02f}% | "
                            f"Elapsed: {format_time(elapsed)} | "
                            f"ETA: {format_time(eta)}")
            else:
                log_rank0(f"{description}: 0.000% | "
                            f"Elapsed: {format_time(elapsed)}")
        stop_event.wait(1)
    total_time = (datetime.now() - t0).total_seconds()
    log_rank0(f"{description}: 100.0% | Total time: {format_time(total_time)}")

def start_progress(read_progress_func, reset_progress_value=None, description="", interval=10):
    if rank != 0:
        return None
    if reset_progress_value is not None:
        reset_progress_value()
    stop_event = threading.Event()
    thread = threading.Thread(target=print_progress, args=(stop_event, read_progress_func, description, interval,))
    thread.daemon = True
    thread.start()
    return ThreadTokens(thread, stop_event)

def stop_progress(thread_tokens):
    if rank != 0:
        return
    thread_tokens.stop_event.set()
    thread_tokens.thread.join()
