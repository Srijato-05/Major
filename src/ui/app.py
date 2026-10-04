import os
import cv2
import base64
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, Optional

from src.simulation.digital_twin import ConveyorDigitalTwinSimulator
from src.utils.config_loader import CONFIG, get_project_root

app = FastAPI(title="Adaptive Multi-Modal AI E-Waste Sorting Simulation")

root_path = get_project_root()
static_dir = os.path.join(root_path, "src", "ui", "static")
templates_dir = os.path.join(root_path, "src", "ui", "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Initialize singleton simulator engine
simulator = ConveyorDigitalTwinSimulator()

@app.get("/", response_class=HTMLResponse)
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-Time WebSocket stream delivering live processed conveyor frames
    and industrial telemetry KPIs at ~25-30 FPS to the dashboard.
    """
    await websocket.accept()
    print("WebSocket client connected.")

    try:
        while True:
            # Check for client commands without blocking
            try:
                msg_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                cmd = json.loads(msg_text)
                if cmd.get("action") == "set_speed":
                    new_speed = float(cmd.get("speed_mps", 2.5))
                    simulator.kinematics.set_speed(new_speed)
                    print(f"Updated belt speed to: {new_speed} m/s")
                elif cmd.get("action") == "set_confidence":
                    new_conf = float(cmd.get("conf_threshold", 0.35))
                    simulator.inference_engine.conf_threshold = new_conf
                    print(f"Updated confidence threshold to: {new_conf}")
            except asyncio.TimeoutError:
                pass

            # Step simulation in background thread so event loop and commands stay 100% responsive
            step_result = await asyncio.to_thread(simulator.step_simulation)
            annotated_frame = step_result["annotated_frame"]

            # Encode frame to JPEG base64 for real-time web streaming
            def encode_jpeg(img):
                _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                return base64.b64encode(buf).decode("utf-8")

            frame_b64 = await asyncio.to_thread(encode_jpeg, annotated_frame)

            # Payload delivery
            payload = {
                "frame_base64": frame_b64,
                "latency_ms": step_result["latency_ms"],
                "fps": step_result["telemetry"].get("fps", 0),
                "active_tracklets": step_result["active_tracklets"],
                "newly_sorted": step_result["newly_sorted"],
                "sorted_history": step_result.get("sorted_history", []),
                "telemetry": step_result["telemetry"],
                "accuracy_metrics": step_result.get("accuracy_metrics", {})
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("WebSocket client disconnected cleanly.")
    except Exception as e:
        print(f"WebSocket session ended: {e}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles user upload of custom images or videos, executes full AI segmentation,
    ByteTrack tracking, and returns annotated media + sorted item manifest.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Invalid image or video format."}

        # Run through digital twin processing
        res = simulator.process_frame(frame)
        annotated_frame = res["annotated_frame"]

        # Encode annotated output
        _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "filename": file.filename,
            "annotated_base64": frame_b64,
            "num_detections": len(res.get("active_tracklets", [])),
            "sorted_items": res.get("newly_sorted", []),
            "telemetry": res.get("telemetry", {})
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ui.app:app", host="127.0.0.1", port=8000, reload=False)
