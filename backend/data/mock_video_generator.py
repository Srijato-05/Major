"""
Mock Video Generator
Generates synthetic multi-modal MP4 conveyor test streams (RGB + NIR) for zero-cost digital twin simulation.
"""

import cv2
import numpy as np
import os

def generate_mock_conveyor_stream(output_path: str, num_frames: int = 300, fps: int = 60, res: tuple = (640, 640)):
    """
    Generates a synthetic MP4 video simulating e-waste items moving down a conveyor belt.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, res)

    for i in range(num_frames):
        frame = np.full((res[1], res[0], 3), 40, dtype=np.uint8) # Belt background
        # Draw moving conveyor grid lines
        offset = (i * 5) % 40
        for y in range(offset, res[1], 40):
            cv2.line(frame, (0, y), (res[0], y), (60, 60, 60), 1)

        # Draw simulated moving e-waste component
        item_y = (i * 8) % (res[1] + 100) - 50
        item_x = 250
        cv2.rectangle(frame, (item_x, item_y), (item_x + 80, item_y + 60), (0, 200, 100), -1)
        cv2.putText(frame, "IC_CHIP", (item_x, item_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated synthetic conveyor video: {output_path}")

if __name__ == "__main__":
    generate_mock_conveyor_stream("data/mock_conveyor.mp4")
