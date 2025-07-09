from yolo_packages import model
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import os

# define number of classes based on YAML
import yaml
with open("/work/akinfalabi/machine_learning/hypocotyl_segment/YOLOv8/yolo_dataset/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

print(f'The Number of classes in the data is: {num_classes}')

#Define a project --> Destination directory for all results
project = "/work/akinfalabi/machine_learning/hypocotyl_segment/YOLOv8/yolo_dataset/test_results"

#Define subdirectory for this specific training
name = "200_epochs-" #note that if you run the training again, it creates a directory: 200_epochs-2

# Train the model
results = model.train(data='/work/akinfalabi/machine_learning/hypocotyl_segment/YOLOv8/yolo_dataset/data.yaml',
                      project=project,
                      name=name,
                      epochs=200,
                      patience=0, # setting patience=0 to disable early stopping.
                      batch=4,
                      imgsz=800)

# Load the best or last model. Make sure the path and name of the model. nb: 200_epochs-4 was used.
my_new_model = YOLO('/work/akinfalabi/machine_learning/hypocotyl_segment/YOLOv8/yolo_dataset/test_results/200_epochs-4/weights/last.pt')


# evaluating the model on the test set
test_dir = '/work/akinfalabi/machine_learning/hypocotyl_segment/YOLOv8/yolo_dataset/test/'  # Source directory with images
output_dir = '/work/akinfalabi/machine_learning/hypocotyl_segment/YOLOv8/yolo_dataset/test_predictions/'  # Where to save results

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Perform prediction on all images in test directory and save results
results = my_new_model.predict(
    source=test_dir,
    conf=0.5,        # Confidence threshold
    save=True,       # Save processed images
    save_dir=output_dir,  # Directory to save predictions
    exist_ok=True    # Overwrite existing files in output directory
)
