"""
Central CLI Entry Point Driver for Real Data Digital Twin Simulation
All file operations and path resolutions are strictly locked within f:/Projects/Major/backend/.
"""

import argparse
import sys
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("main")

# Base directory strictly locked to workspace backend root folder
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))

def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        config_path = os.path.join(BACKEND_DIR, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Industrial E-Waste Multi-Modal Segmentation & Pneumatic Sorting Digital Twin")
    parser.add_argument("--config", type=str, default="configs/system_config.yaml", help="Path to system_config.yaml")
    parser.add_argument("--mode", type=str, choices=["run", "download_data", "generate_stream", "test", "export"], default="run", help="Execution mode")
    parser.add_argument("--speed", type=float, default=None, help="Dynamic override for conveyor belt speed (m/s)")
    parser.add_argument("--valves", type=int, default=None, help="Dynamic override for pneumatic valve count")
    
    args = parser.parse_args()

    logger.info("Initializing E-Waste Digital Twin Engine (Real Data & Local Workspace Confinement)...")
    config = load_config(args.config)

    # Dynamic flag overrides
    if args.speed is not None:
        config["conveyor"]["speed_m_per_s"] = args.speed
        logger.info(f"Overriding conveyor speed to: {args.speed} m/s")
    if args.valves is not None:
        config["pneumatic_actuator"]["num_valves"] = args.valves
        logger.info(f"Overriding pneumatic valve count to: {args.valves}")

    if args.mode == "download_data":
        from data.dataset_downloader import download_and_prepare_real_datasets
        download_and_prepare_real_datasets()
    elif args.mode == "generate_stream":
        from data.real_stream_loader import generate_real_conveyor_stream
        generate_real_conveyor_stream()
    elif args.mode == "test":
        import pytest
        logger.info("Running automated test suite within project workspace...")
        tests_dir = os.path.join(BACKEND_DIR, "tests")
        sys.exit(pytest.main([tests_dir]))
    elif args.mode == "run":
        stream_path = os.path.join(BACKEND_DIR, "data", "real_conveyor_stream.mp4")
        if not os.path.exists(stream_path):
            logger.info(f"Real conveyor stream video not found at {stream_path}. Initializing workspace stream loader...")
            from data.real_stream_loader import generate_real_conveyor_stream
            generate_real_conveyor_stream()

        from core.pipeline_manager import DigitalTwinPipelineManager
        pipeline = DigitalTwinPipelineManager(args.config)
        pipeline.start()
        logger.info("Digital twin pipeline running with real dataset streams. Press Ctrl+C to exit.")

if __name__ == "__main__":
    main()
