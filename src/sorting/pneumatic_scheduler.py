import heapq
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
from src.utils.config_loader import CONFIG


class ItemState(Enum):
    DETECTED = auto()
    TRACKED = auto()
    IN_TRANSIT = auto()
    EJECTED = auto()
    MISSED_ACTUATION = auto()
    PASSED_UNACTUATED = auto()


@dataclass
class ItemLifecycleRecord:
    track_id: int
    class_id: int
    x_w_mm: float
    y_w_mm: float
    detect_time_ns: int
    confirm_count: int = 1
    state: ItemState = ItemState.DETECTED


@dataclass
class ActuationPulse:
    t_fire_ns: int
    valve_idx: int
    pulse_duration_ms: float
    track_id: int
    class_id: int
    target_bin: str

    def __lt__(self, other: "ActuationPulse") -> bool:
        return self.t_fire_ns < other.t_fire_ns


class PneumaticActuatorScheduler:
    """
    64-Nozzle Pneumatic Air-Jet Scheduler & Finite State Machine Lifecycle Tracker.
    Computes nanosecond solenoid actuation timestamps from conveyor kinematics:
        t_fire_ns = now_ns + ((ejection_line_x_mm - x_w_mm) / (speed_mps * 1000)) * 1e9 - lag_ns
    """

    def __init__(
        self,
        ejection_line_x_mm: Optional[float] = None,
        actuation_lag_ms: Optional[float] = None,
        solenoid_pitch_mm: Optional[float] = None,
        total_nozzles: Optional[int] = None,
        speed_mps: Optional[float] = None,
        eject_classes: Optional[Set[int]] = None,
    ):
        pneu_cfg = CONFIG.get("pneumatics", {})
        conv_cfg = CONFIG.get("conveyor", {})

        self.ejection_line_x_mm = float(
            ejection_line_x_mm if ejection_line_x_mm is not None else pneu_cfg.get("ejection_line_x_mm", 1200.0)
        )
        self.actuation_lag_ms = float(
            actuation_lag_ms if actuation_lag_ms is not None else pneu_cfg.get("actuation_lag_ms", 2.5)
        )
        self.solenoid_lag_ns = int(self.actuation_lag_ms * 1e6)
        self.solenoid_pitch_mm = float(
            solenoid_pitch_mm if solenoid_pitch_mm is not None else pneu_cfg.get("solenoid_pitch_mm", 15.625)
        )
        self.total_nozzles = int(
            total_nozzles if total_nozzles is not None else pneu_cfg.get("total_nozzles", 64)
        )
        self.speed_mps = float(speed_mps if speed_mps is not None else conv_cfg.get("speed_mps", 2.5))
        self.eject_classes = eject_classes if eject_classes is not None else {1, 2, 3, 4, 5, 6}

        self._queue: List[ActuationPulse] = []
        self.items: Dict[int, ItemLifecycleRecord] = {}
        self.total_scheduled = 0
        self.total_ejected = 0
        self.total_missed = 0
        self.last_fired_nozzles: List[int] = []

    def calculate_nozzle_index(self, y_w_mm: float) -> int:
        """Maps transverse belt position y_w_mm [0..belt_width_mm] to discrete nozzle [0..total_nozzles-1]."""
        idx = int(y_w_mm / self.solenoid_pitch_mm)
        return max(0, min(self.total_nozzles - 1, idx))

    def update_and_schedule(
        self,
        track_id: int,
        class_id: int,
        x_w_mm: float,
        y_w_mm: float,
        target_bin: str,
        speed_mps: Optional[float] = None,
        now_ns: Optional[int] = None,
        pulse_duration_ms: float = 20.0,
    ) -> Optional[ActuationPulse]:
        """Updates tracklet FSM state and schedules a pneumatic pulse once confirmed."""
        if now_ns is None:
            now_ns = time.monotonic_ns()
        belt_speed = float(speed_mps if speed_mps is not None else self.speed_mps)

        if track_id not in self.items:
            self.items[track_id] = ItemLifecycleRecord(
                track_id=track_id,
                class_id=class_id,
                x_w_mm=x_w_mm,
                y_w_mm=y_w_mm,
                detect_time_ns=now_ns,
                confirm_count=1,
                state=ItemState.DETECTED,
            )
            return None

        rec = self.items[track_id]
        rec.x_w_mm = x_w_mm
        rec.y_w_mm = y_w_mm
        rec.confirm_count += 1
        if rec.state == ItemState.DETECTED and rec.confirm_count >= 2:
            rec.state = ItemState.TRACKED

        if rec.state == ItemState.TRACKED and class_id in self.eject_classes:
            dist_mm = self.ejection_line_x_mm - x_w_mm
            if dist_mm <= 0:
                rec.state = ItemState.MISSED_ACTUATION
                self.total_missed += 1
                return None

            travel_sec = dist_mm / max(belt_speed * 1000.0, 1e-3)
            t_target_ns = now_ns + int(travel_sec * 1e9)
            t_fire_ns = max(now_ns, t_target_ns - self.solenoid_lag_ns)
            valve_idx = self.calculate_nozzle_index(y_w_mm)

            pulse = ActuationPulse(
                t_fire_ns=t_fire_ns,
                valve_idx=valve_idx,
                pulse_duration_ms=pulse_duration_ms,
                track_id=track_id,
                class_id=class_id,
                target_bin=target_bin,
            )
            heapq.heappush(self._queue, pulse)
            rec.state = ItemState.IN_TRANSIT
            self.total_scheduled += 1
            return pulse
        elif rec.state == ItemState.TRACKED and class_id not in self.eject_classes:
            rec.state = ItemState.PASSED_UNACTUATED

        return None

    def pop_due_pulses(self, now_ns: Optional[int] = None) -> List[ActuationPulse]:
        """Pops all actuation pulses due at or before now_ns and marks items EJECTED."""
        if now_ns is None:
            now_ns = time.monotonic_ns()
        due: List[ActuationPulse] = []
        while self._queue and self._queue[0].t_fire_ns <= now_ns:
            pulse = heapq.heappop(self._queue)
            due.append(pulse)
            if pulse.track_id in self.items:
                self.items[pulse.track_id].state = ItemState.EJECTED
            self.total_ejected += 1
        self.last_fired_nozzles = [p.valve_idx for p in due]
        return due

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_scheduled": self.total_scheduled,
            "total_ejected": self.total_ejected,
            "total_missed": self.total_missed,
            "pending_pulses": len(self._queue),
            "active_nozzles": list(self.last_fired_nozzles),
        }

