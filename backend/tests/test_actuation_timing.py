"""
Unit Test: Pneumatic Actuation Priority Queue Microsecond Scheduling
"""

import pytest
import time
from core.actuator_queue import PneumaticActuatorQueue

def test_actuation_queue_scheduling():
    queue = PneumaticActuatorQueue(
        ejection_distance_m=1.5,
        v_belt_m_s=2.5,
        solenoid_delay_ms=12.0,
        num_valves=16,
        valve_spacing_m=0.05
    )

    now_ns = time.monotonic_ns()
    # Object at y_w = 0.5m -> 1.0m remaining -> 0.4s travel time = 400,000,000 ns
    # solenoid delay = 12ms = 12,000,000 ns
    # expected t_fire = now_ns + 388,000,000 ns
    t_fire_ns, valve_idx = queue.schedule_ejection(
        track_id=1,
        y_w=0.5,
        x_w=0.25,
        pulse_duration_ms=25.0,
        now_ns=now_ns
    )

    expected_fire = now_ns + int(0.4 * 1e9) - int(12 * 1e6)
    assert abs(t_fire_ns - expected_fire) < 1000 # Microsecond precision check
    assert valve_idx == 5 # 0.25 / 0.05 = 5
