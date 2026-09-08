import logging
import queue
import threading
from logging.handlers import TimedRotatingFileHandler


_log_queue = queue.Queue()
_log_thread = None
_log_lock = threading.Lock()


def configure_logging():
    """Configure global logging and start the log worker once."""
    global _log_thread

    with _log_lock:
        # Logging is already configured.
        if _log_thread is not None and _log_thread.is_alive():
            return

        root_logger = logging.getLogger()

        # Prevent duplicate/default console handlers.
        root_logger.handlers.clear()

        log_handler = TimedRotatingFileHandler(
            "logs/runtime.log",
            when="midnight",
            interval=1,
            backupCount=7,
        )
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        log_handler.suffix = "%Y%m%d"

        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.DEBUG)

        for noisy_logger in ("web3", "urllib3", "requests"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        # Start exactly one global logging worker.
        _log_thread = threading.Thread(
            target=_log_worker,
            name="LogWorker",
            daemon=True,
        )
        _log_thread.start()


def log_message(
    level,
    file_message,
    console_message=None,
    should_print=False,
):
    """Queue a log message for the global logging worker."""
    configure_logging()

    _log_queue.put(
        (
            level,
            file_message,
            console_message,
            should_print,
        )
    )


def _log_worker():
    """Consume queued logs and write them sequentially."""
    while True:
        item = _log_queue.get()

        try:
            if item is None:
                return

            level, file_message, console_message, should_print = item

            # Only this thread writes to the file.
            logging.log(level, file_message)

            if should_print and console_message:
                print(console_message, flush=True)

        finally:
            _log_queue.task_done()
