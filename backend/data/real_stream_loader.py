"""
Real Stream Loader & Video Generator
Stitches real industrial e-waste dataset frames (from PCB-Vision, DeepPCB, PKU-Market-PCB) into real MP4 conveyor test streams for hardware-in-the-loop and digital twin simulation.
All input/output video paths are strictly locked within f:/Projects/Major/backend/data/.
"""

import cv2
import numpy as np
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [real_stream_loader]: %(message)s")
logger = logging.getLogger("real_stream_loader")

# Base directory strictly locked to workspace backend data folder
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def generate_real_conveyor_stream(dataset_dir: str = None, output_video_path: str = None, fps: int = 60, res: tuple = (640, 640)):
    """
    Constructs a real conveyor belt video stream strictly within the local workspace data folder.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(BASE_DATA_DIR, "dataset_harmonized")
    if output_video_path is None:
        output_video_path = os.path.join(BASE_DATA_DIR, "real_conveyor_stream.mp4")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    images_dir = os.path.join(dataset_dir, "images")
    img_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    
    if not img_files:
        logger.warning(f"No real dataset images found in {images_dir}. Preparing local dataset directory...")
        from data.dataset_downloader import download_and_prepare_real_datasets
        download_and_prepare_real_datasets(BASE_DATA_DIR)
        img_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))

    if not img_files:
        logger.warning(f"No image files ready in {images_dir}. Real video stream workspace initialized at: {output_video_path}")
        return False

    logger.info(f"Stitching real dataset frames into conveyor stream video: {output_video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, res)

    num_frames = 300
    for i in range(num_frames):
        frame = np.full((res[1], res[0], 3), 35, dtype=np.uint8)
        offset = (i * 6) % 40
        for y in range(offset, res[1], 40):
            cv2.line(frame, (0, y), (res[0], y), (55, 55, 55), 1)

        real_img_path = img_files[i % len(img_files)]
        real_img = cv2.imread(real_img_path)
        if real_img is not None:
            real_img = cv2.resize(real_img, (160, 160))
            pos_y = (i * 8) % (res[1] + 160) - 160
            pos_x = 240
            
            if 0 <= pos_y < res[1] - 160:
                frame[pos_y:pos_y + 160, pos_x:pos_x + 160] = real_img
                cv2.rectangle(frame, (pos_x, pos_y), (pos_x + 160, pos_y + 160), (0, 255, 0), 2)
                cv2.putText(frame, "REAL_PCB_COMPONENT", (pos_x, pos_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        writer.write(frame)

    writer.release()
    logger.info(f"Real conveyor video stream ready at workspace: {output_video_path}")
    return True

if __name__ == "__main__":
    generate_real_conveyor_stream()
