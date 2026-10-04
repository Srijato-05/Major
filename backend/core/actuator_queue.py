"""
Actuator Event Queue
Microsecond-resolution priority queue for scheduling pneumatic air-jet ejection pulses.
Calculates target arrival timestamp t_target and fire timestamp t_fire based on time.monotonic_ns().
"""

import heapq
import time

class ActuationEvent:
    def __init__(self, t_fire_ns: int, valve_idx: int, pulse_duration_ms: float, track_id: int):
        self.t_fire_ns = t_fire_ns
        self.valve_idx = valve_idx
        self.pulse_duration_ms = pulse_duration_ms
        self.track_id = track_id

    def __lt__(self, other):
        return self.t_fire_ns < other.t_fire_ns

class PneumaticActuatorQueue:
    def __init__(self, ejection_distance_m: float, v_belt_m_s: float, solenoid_delay_ms: float, num_valves: int, valve_spacing_m: float):
        self.ejection_distance_m = ejection_distance_m
        self.v_belt_m_s = v_belt_m_s
        self.solenoid_delay_ns = int(solenoid_delay_ms * 1e6)
        self.num_valves = num_valves
        self.valve_spacing_m = valve_spacing_m
        self.queue = []

    def calculate_nozzle_index(self, x_w: float) -> int:
        """
        Maps metric transverse position x_w to discrete valve index [0, num_valves - 1].
        """
        val = int(x_w / self.valve_spacing_m)
        return max(0, min(self.num_valves - 1, val))

    def schedule_ejection(self, track_id: int, y_w: float, x_w: float, pulse_duration_ms: float, now_ns: int = None):
        """
        Computes arrival timestamp t_target and firing timestamp t_fire in nanoseconds.
        """
        if now_ns is None:
            now_ns = time.monotonic_ns()

        dist_remaining = self.ejection_distance_m - y_w
        travel_time_sec = dist_remaining / self.v_belt_m_s
        travel_time_ns = int(travel_time_sec * 1e9)

        t_target_ns = now_ns + travel_time_ns
        t_fire_ns = t_target_ns - self.solenoid_delay_ns

        valve_idx = self.calculate_nozzle_index(x_w)
        event = ActuationEvent(t_fire_ns, valve_idx, pulse_duration_ms, track_id)
        heapq.heappush(self.queue, event)
        return t_fire_ns, valve_idx

    def pop_due_events(self, now_ns: int = None) -> list:
        """
        Returns list of actuation events that are due for execution (t_fire <= now_ns).
        """
        if now_ns is None:
            now_ns = time.monotonic_ns()

        due = []
        while self.queue and self.queue[0].t_fire_ns <= now_ns:
            due.append(heapq.heappop(self.queue))
        return due
