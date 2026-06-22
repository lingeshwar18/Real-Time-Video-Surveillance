# save_clip.py
from collections import deque

class FrameBuffer:
    def __init__(self, maxlen=200):
        self.buf = deque(maxlen=maxlen)

    def add(self, frame):
        self.buf.append(frame.copy())

    def get_all(self):
        return list(self.buf)

    def clear(self):
        self.buf.clear()
