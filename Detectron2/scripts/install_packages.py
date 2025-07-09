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

# Detectron2
if not is_installed("detectron2"):
	install_packages("git+https://github.com/facebookresearch/detectron2.git")
else:
 	print("Detectron2 already installed.")

# PyTorch
if not is_installed("torch") or not is_installed("torchvision") or not is_installed("torchaudio"):
    install_packages("torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118")
else:
    print("torch/vision/audio already installed.")

# Other packages
standard_packages = {
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "scipy": "scipy",
    "PIL": "pillow"
}

for module_name, pip_name in standard_packages.items():
    if not is_installed(module_name):
        install_packages(pip_name)
    else:
        print(f"{module_name} already installed.")
