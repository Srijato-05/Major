"""
ByteTrack Multi-Object Tracker Implementation
Associates detections across frames using two-stage IoU matching and Kalman Filter state estimation
without requiring deep ReID appearance extraction overhead.
"""

import numpy as np

class KalmanFilter2D:
    """Simple 2D Constant Velocity Kalman Filter for Bounding Box Trajectories."""
    def __init__(self, bbox: list):
        # State: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        self.mean = np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def predict(self):
        # State transition: x = x + vx
        self.mean[0] += self.mean[4]
        self.mean[1] += self.mean[5]

    def update(self, bbox: list):
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        
        # Velocity update
        self.mean[4] = cx - self.mean[0]
        self.mean[5] = cy - self.mean[1]
        
        self.mean[0] = cx
        self.mean[1] = cy
        self.mean[2] = w / h
        self.mean[3] = h

    def get_bbox(self) -> list:
        cx, cy, r, h = self.mean[:4]
        w = r * h
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

class Tracklet:
    def __init__(self, track_id: int, bbox: list, score: float, class_id: int):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.hits = 1
        self.time_since_update = 0
        self.kalman = KalmanFilter2D(bbox)

    def predict(self):
        self.kalman.predict()
        self.bbox = self.kalman.get_bbox()
        self.time_since_update += 1

    def update(self, bbox: list, score: float):
        self.kalman.update(bbox)
        self.bbox = self.kalman.get_bbox()
        self.score = score
        self.hits += 1
        self.time_since_update = 0

def compute_iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class ByteTracker:
    def __init__(self, track_thresh: float = 0.50, match_thresh: float = 0.75, max_age: int = 45):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracked_stracks = []

    def update(self, detections: list) -> list:
        """
        Two-stage ByteTrack association:
        Stage 1: Match high-confidence detections (score >= track_thresh) with active tracks.
        Stage 2: Match remaining unmatched tracks with low-confidence detections (score < track_thresh).
        """
        # Predict Kalman states for existing tracks
        for t in self.tracked_stracks:
            t.predict()

        high_dets = [d for d in detections if d[4] >= self.track_thresh]
        low_dets = [d for d in detections if d[4] < self.track_thresh]

        unmatched_tracks = list(self.tracked_stracks)
        matched_tracks = []

        # Stage 1: Match high-score detections
        for det in high_dets:
            det_bbox = det[:4]
            best_iou = 0.0
            best_track = None

            for track in unmatched_tracks:
                iou = compute_iou(det_bbox, track.bbox)
                if iou > best_iou and iou >= self.match_thresh:
                    best_iou = iou
                    best_track = track

            if best_track is not None:
                best_track.update(det_bbox, det[4])
                matched_tracks.append(best_track)
                unmatched_tracks.remove(best_track)
            else:
                # Create new tracklet
                new_t = Tracklet(self.next_id, det_bbox, det[4], int(det[5]))
                self.next_id += 1
                matched_tracks.append(new_t)

        # Stage 2: Match low-score detections against remaining unmatched tracks
        still_unmatched = []
        for track in unmatched_tracks:
            best_iou = 0.0
            best_det = None
            for det in low_dets:
                iou = compute_iou(det[:4], track.bbox)
                if iou > best_iou and iou >= 0.50:
                    best_iou = iou
                    best_det = det

            if best_det is not None:
                track.update(best_det[:4], best_det[4])
                matched_tracks.append(track)
            else:
                if track.time_since_update <= self.max_age:
                    still_unmatched.append(track)

        self.tracked_stracks = matched_tracks + still_unmatched
        return self.tracked_stracks
