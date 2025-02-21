import threading
from datetime import datetime
from dataclasses import dataclass
from utils.logger import log_rank0, format_time

@dataclass
class ThreadTokens:
    thread: threading.Thread
    stop_event: threading.Event

def print_progress(stop_event, read_progress_func, description="", interval=10):
    t0 = now = datetime.now()
    while not stop_event.is_set():
        if (datetime.now() - now).total_seconds() > interval:
            now = datetime.now()
            elapsed = (datetime.now() - t0).total_seconds()
            progress = read_progress_func()
            progress = progress if progress < 1 else 1
            if progress > 0:
                log_rank0(f"{description}: {(progress * 100):.2f}% | "
                            f"Elapsed: {format_time(elapsed)} | "
                            f"ETA: {format_time((elapsed / progress) * (1 - progress))}")
            else:
                log_rank0(f"{description}: 0% | "
                            f"Total time: {format_time(elapsed)}")
        stop_event.wait(1)
    total_time = (datetime.now() - t0).total_seconds()
    log_rank0(f"{description}: 100% | Total time: {format_time(total_time)}")

def start_progress(read_progress_func, description="", interval=10):
    stop_event = threading.Event()
    thread = threading.Thread(target=print_progress, args=(stop_event, read_progress_func, description, interval,))
    thread.daemon = True
    thread.start()
    return ThreadTokens(thread, stop_event)

def stop_progress(thread_tokens):
    thread_tokens.stop_event.set()
    thread_tokens.thread.join()
