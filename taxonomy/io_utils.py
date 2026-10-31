import io
import logging
import pickle
import threading

class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    self.logger.log(self.level, line)
        return len(s)

    def flush(self):
        with self._lock:
            if self._buf.strip():
                self.logger.log(self.level, self._buf.strip())
            self._buf = ""

def save_to_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
