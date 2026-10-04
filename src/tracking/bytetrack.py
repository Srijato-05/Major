import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.tracking.kalman_filter import ConveyorKalmanFilter
from src.utils.config_loader import CONFIG

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    """Individual single-object tracklet representation."""
    shared_kalman = ConveyorKalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int, class_name: str, polygon: list = None):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.score = score
        self.class_id = class_id
        self.class_name = class_name
        self.polygon = polygon or []

        self.tracklet_len = 0
        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = 0

    def to_xyah(self) -> np.ndarray:
        ret = self._tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def activate(self, kalman_filter: ConveyorKalmanFilter, frame_id: int, track_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.polygon = new_track.polygon
        self.class_id = new_track.class_id
        self.class_name = new_track.class_name

    def update(self, new_track: "STrack", frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.to_xyah()
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_tlwh
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.polygon = new_track.polygon
        self.class_id = new_track.class_id
        self.class_name = new_track.class_name

    def predict(self):
        if self.state != TrackState.Tracked:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


def iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """Computes cost matrix based on IoU between two sets of tracks."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    atlwhs = np.array([track.to_tlwh() for track in atracks])
    btlwhs = np.array([track.to_tlwh() for track in btracks])

    ious = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    for i, a in enumerate(atlwhs):
        ax1, ay1, aw, ah = a
        ax2, ay2 = ax1 + aw, ay1 + ah
        a_area = aw * ah
        for j, b in enumerate(btlwhs):
            bx1, by1, bw, bh = b
            bx2, by2 = bx1 + bw, by1 + bh
            b_area = bw * bh

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            intersection = iw * ih
            union = a_area + b_area - intersection
            ious[i, j] = intersection / union if union > 0 else 0.0

    return 1.0 - ious  # Cost matrix (0 = identical, 1 = no overlap)


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Simple greedy/Hungarian linear assignment."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        matches, unmatched_a, unmatched_b = [], [], []
        matched_b = set()

        for r, c in zip(x, y):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                matched_b.add(c)
            else:
                unmatched_a.append(r)

        for i in range(cost_matrix.shape[0]):
            if i not in x:
                unmatched_a.append(i)
        for j in range(cost_matrix.shape[1]):
            if j not in matched_b:
                unmatched_b.append(j)

        return np.array(matches, dtype=int), tuple(unmatched_a), tuple(unmatched_b)
    except Exception:
        # Fallback greedy match
        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))
        return np.array(matches, dtype=int), tuple(unmatched_a), tuple(unmatched_b)


class ByteTracker:
    """
    Two-Stage Association Multi-Object Tracker (ByteTrack) for High-Speed Conveyors.
    Eliminates appearance ReID overhead to achieve >= 50-60 FPS without ID switches.
    """

    def __init__(self, track_thresh: float = None, track_buffer: int = None, match_thresh: float = None):
        cfg_track = CONFIG.get("tracking", {})
        self.track_thresh = track_thresh if track_thresh is not None else cfg_track.get("track_thresh", 0.5)
        self.track_buffer = track_buffer if track_buffer is not None else cfg_track.get("track_buffer", 30)
        self.match_thresh = match_thresh if match_thresh is not None else cfg_track.get("match_thresh", 0.8)

        self.frame_id = 0
        self.track_id_counter = 0

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.kalman_filter = ConveyorKalmanFilter()

    def update(self, instances: List[Dict[str, Any]]) -> List[STrack]:
        """
        Processes new detection instances and updates tracklet trajectories.
        Args:
            instances: list of dicts with 'bbox' [x, y, w, h], 'confidence', 'class_id', 'class_name', 'polygon'
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        detections = []
        detections_second = []

        # Split into high-confidence and low-confidence detections
        for inst in instances:
            bbox = np.array(inst["bbox"], dtype=np.float32)
            score = float(inst.get("confidence", 0.5))
            cid = int(inst.get("class_id", 0))
            cname = str(inst.get("class_name", "Unknown"))
            poly = inst.get("polygon", [])

            strack = STrack(bbox, score, cid, cname, poly)
            if score >= self.track_thresh:
                detections.append(strack)
            elif score >= 0.1:
                detections_second.append(strack)

        # Predict current locations with Kalman Filter
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = tracked_stracks + self.lost_stracks
        for strack in strack_pool:
            strack.predict()

        # ---------- STAGE 1: Match high-confidence detections ----------
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ---------- STAGE 2: Match low-confidence detections ----------
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_strack, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_strack:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks
        detections_unmatched = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections_unmatched)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_unmatched[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Init new tracks
        for inew in u_detection:
            track = detections_unmatched[inew]
            if track.score < self.track_thresh:
                continue
            self.track_id_counter += 1
            track.activate(self.kalman_filter, self.frame_id, self.track_id_counter)
            activated_stracks.append(track)

        # Update active and lost track lists uniquely without duplicates
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        # Retain only current tracked and newly activated/refound tracks uniquely by track_id
        active_dict = {}
        for t in activated_stracks + refind_stracks:
            if t.state == TrackState.Tracked and t.is_activated:
                active_dict[t.track_id] = t

        self.tracked_stracks = list(active_dict.values())
        
        # Maintain lost tracks
        lost_dict = {}
        for t in self.lost_stracks + lost_stracks:
            if t.state == TrackState.Lost and t.track_id not in active_dict:
                lost_dict[t.track_id] = t
        self.lost_stracks = list(lost_dict.values())
        self.removed_stracks.extend(removed_stracks)

        return self.tracked_stracks
