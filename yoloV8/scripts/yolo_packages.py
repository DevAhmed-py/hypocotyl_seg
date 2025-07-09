import subprocess
import sys
import os
import importlib

def is_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_packages(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

# YOLOv8
if not is_installed("ultralytics"):
    install_packages("ultralytics")
else:
    print("YOLOv8 already installed.")

# Other packages
standard_packages = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "PIL": "pillow"
}

for module_name, pip_name in standard_packages.items():
    if not is_installed(module_name):
        install_packages(pip_name)
    else:
        print(f"{module_name} already installed.")


# Instance segmentation model
from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8n-seg.pt')  # transfer the weights from a pretrained model
