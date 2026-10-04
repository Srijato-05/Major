import sys
import os
import uvicorn
from src.utils.config_loader import get_project_root

def main():
    root = get_project_root()
    sys.path.insert(0, str(root))
    print("=" * 70)
    print(" Starting Adaptive Multi-Modal AI E-Waste Sorting Digital Twin ")
    print("=" * 70)
    print("Web Dashboard URL: http://127.0.0.1:8000")
    print("Press Ctrl+C to terminate.")
    print("=" * 70)
    uvicorn.run("src.ui.app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()
