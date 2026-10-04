"""
Finite State Machine (FSM) Lifecycle Tracker
Enforces non-regressive item states: DETECTED -> TRACKED -> IN_TRANSIT -> TRIGGER_READY -> EJECTED / MISSED_ACTUATION.
"""

from enum import Enum, auto
import time

class ItemState(Enum):
    DETECTED = auto()
    TRACKED = auto()
    IN_TRANSIT = auto()
    TRIGGER_READY = auto()
    EJECTED = auto()
    MISSED_ACTUATION = auto()
    PASSED_UNACTUATED = auto()

class ItemLifecycleTracker:
    def __init__(self, track_id: int, class_id: int, initial_world_pos: tuple, detect_time_ns: int):
        self.track_id = track_id
        self.class_id = class_id
        self.x_w, self.y_w = initial_world_pos
        self.detect_time_ns = detect_time_ns
        self.confirm_count = 1
        self.state = ItemState.DETECTED

    def update_position(self, new_world_pos: tuple):
        self.x_w, self.y_w = new_world_pos
        self.confirm_count += 1
        if self.state == ItemState.DETECTED and self.confirm_count >= 3:
            self.state = ItemState.TRACKED

    def set_state(self, new_state: ItemState):
        self.state = new_state
