import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Register the dataset
register_coco_instances("my_dataset_train", {}, "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/data/train/train.json", "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/data/train/")
register_coco_instances("my_dataset_val", {}, "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/data/val/val.json", "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/data/val")

# Train set
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

# Validation set
val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

# Train model with data
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.OUTPUT_DIR = "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/model2/"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.TEST.EVAL_PERIOD = 500  # Evaluate periodically
cfg.DATALOADER.NUM_WORKERS = 2
# Had to download the model in log-in node on HPC since I wouldn't be able to in the srun session
cfg.MODEL.WEIGHTS = "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/model/model_final_f10217.pkl"
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001  # changed from 0.00025 to 0.001
cfg.SOLVER.MAX_ITER = 5000    # Increased from 1000 to 5000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"   # to remove the squiggly masks of the results
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1     # hypocotyl

# Added these configurations for data augmentation for better generalization
# cfg.SOLVER.WARMUP_ITERS = 500   # Warmup phase
# cfg.INPUT.MIN_SIZE_TRAIN = (512, 640)  # Multi-scale training
# cfg.INPUT.CROP.ENABLED = True   # Random cropping
# cfg.INPUT.RANDOM_FLIP = "horizontal"
# cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 56  # Higher-res masks

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# trainer = DefaultTrainer(cfg) #Create an instance of of DefaultTrainer with the given congiguration

# Used this to add evaluation for the validation data during training
from detectron2.evaluation import COCOEvaluator

class ValidationTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

trainer = ValidationTrainer(cfg)  # Instead of DefaultTrainer

trainer.resume_or_load(resume=False) #Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available

trainer.train()

# Inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(val_dataset_dicts, 1):    # Select number of images for display
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=val_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])


# Test set to evaluate the model on unseen data
# Directory path to the input images folder
input_images_directory = "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/data/test/"

# Output directory where the segmented images will be saved
output_directory = "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/data/test_results"  # Replace this with the path to your desired output directory

os.makedirs(output_directory, exist_ok=True)

# Loop over the images in the input folder
for image_filename in os.listdir(input_images_directory):
    if image_filename == ".DS_Store":
        continue  # skip this file
    image_path = os.path.join(input_images_directory, image_filename)
    new_im = cv2.imread(image_path)

    # Perform prediction on the new image
    outputs = predictor(new_im)  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(new_im[:, :, ::-1], metadata=train_metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Create the output filename with _result extension
    result_filename = os.path.splitext(image_filename)[0] + "_result.png"
    output_path = os.path.join(output_directory, result_filename)

    # Save the segmented image
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

print("Segmentation of all images completed.")

# Save the configuration to a config.yaml file
import yaml
config_yaml_path = "/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/model2/config.yaml"
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)

print(f"Configuration saved to {config_yaml_path}")
