"""
Operator HUD (Thread 3 Visual Telemetry Overlay)
Renders high-speed industrial visual display with tracked polygons, ejection line indicators, and real-time WEEE metrics.
"""

import cv2
import numpy as np

CLASS_COLORS = {
    0: (120, 120, 120),  # pcb_substrate
    1: (0, 255, 0),      # ic_chip
    2: (255, 165, 0),    # capacitor_transformer
    3: (200, 200, 0),    # heat_sink
    4: (0, 215, 255),    # gold_finger_connector
    5: (0, 0, 255)       # hazardous_battery
}

def render_operator_hud(frame: np.ndarray, tracks: list, active_valves: list, fps: float, latency_ms: float) -> np.ndarray:
    """
    Overlays telemetry onto live conveyor frame.
    """
    hud_frame = frame.copy()
    h, w = hud_frame.shape[:2]

    # Draw Conveyor Ejection Line
    ejection_y = int(h * 0.75)
    cv2.line(hud_frame, (0, ejection_y), (w, ejection_y), (0, 0, 255), 2)
    cv2.putText(hud_frame, "AIR-JET EJECTION LINE", (10, ejection_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw Nozzle indicators
    num_valves = 16
    valve_w = w / num_valves
    for i in range(num_valves):
        x1 = int(i * valve_w)
        x2 = int((i + 1) * valve_w)
        color = (0, 0, 255) if i in active_valves else (50, 50, 50)
        cv2.rectangle(hud_frame, (x1 + 2, ejection_y + 2), (x2 - 2, ejection_y + 15), color, -1)

    # Telemetry sidebar
    cv2.rectangle(hud_frame, (w - 200, 0), (w, 120), (20, 20, 20), -1)
    cv2.putText(hud_frame, f"FPS: {fps:.1f}", (w - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(hud_frame, f"Latency: {latency_ms:.1f}ms", (w - 190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(hud_frame, "Status: RUNNING", (w - 190, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return hud_frame
