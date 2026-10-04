"""
Video Ingestion Worker (Thread 1)
Reads video frames non-blockingly into a bounded circular thread-safe buffer.
"""

import cv2
import threading
from queue import Queue
import time

class VideoIngestWorker(threading.Thread):
    def __init__(self, video_source: str, frame_queue: Queue, target_fps: int = 60):
        super().__init__()
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.target_fps = target_fps
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        delay = 1.0 / self.target_fps

        while not self.stopped and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue

            if not self.frame_queue.full():
                self.frame_queue.put((time.monotonic_ns(), frame))
            time.sleep(delay)

        cap.release()

    def stop(self):
        self.stopped = True
