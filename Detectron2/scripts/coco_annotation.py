import os
import json
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# function to create COCO annotations for a single category
def create_coco_annotations(root_dir, output_dir, category_name="object"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Single category configuration
    category_id = 1
    mask_ext = 'png'
    # image_exts = ['png', 'jpg', 'jpeg']

    for split in ["train", "val"]:
        coco_data = {
            "info": {},
            "licenses": [],
            "categories": [{"id": category_id, "name": category_name, "supercategory": ""}],
            "images": [],
            "annotations": []
        }
        image_id = 1
        annotation_id = 1
        mask_dir = os.path.join(root_dir, split, "mask")
        rgb_dir = os.path.join(root_dir, split, "rgb")

        # Check if mask directory exists
        if not os.path.exists(mask_dir):
            print(f"Mask directory not found: {mask_dir}")
            continue

        mask_files = glob.glob(os.path.join(mask_dir, f'*.{mask_ext}'))
        if not mask_files:
            print(f"No mask files found in {mask_dir}")
            continue

        # Iterate through all mask files
        for mask_path in tqdm(mask_files, desc=f"Processing {split}"):
            # Derive RGB filename by removing _mask suffix
            mask_filename = os.path.basename(mask_path)
            base_name = mask_filename.replace('_mask', '')
            rgb_path = None
            potential_path = os.path.join(rgb_dir, f"{base_name}")

            if os.path.exists(potential_path):
                rgb_path = potential_path
            if not rgb_path:
                print(f"RGB image not found for mask {os.path.basename(mask_path)}")
                continue

            # Add image entry if not exists
            existing_image = next((img for img in coco_data["images"] if img["file_name"] == os.path.basename(rgb_path)), None)
            if not existing_image:
                try:
                    with Image.open(rgb_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Error loading {rgb_path}: {e}")
                    continue

                coco_data["images"].append({
                    "id": image_id,
                    "file_name": os.path.basename(rgb_path),
                    "width": width,
                    "height": height,
                    "license": 0,
                    "date_captured": ""
                })
                current_image_id = image_id
                image_id += 1
            else:
                current_image_id = existing_image["id"]

            # Process mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue
            # Binarize and find instances
            _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

            for label in range(1, num_labels):
                # Get instance properties
                x, y, w, h, area = stats[label][:5]
                instance_mask = (labels == label).astype(np.uint8) * 255

                # Find contours
                contours = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                segmentation = []
                
                for contour in contours:
                    # Flatten and add to segmentation
                    segmentation.append(contour.flatten().tolist())
                if not segmentation:
                    continue

                contour_area = sum(cv2.contourArea(c) for c in contours)

                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": float(contour_area),  # or use area from stats: float(area)
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0
                })
                annotation_id += 1

        # Save COCO JSON
        output_path = os.path.join(output_dir, f"{split}_annotations.json")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"Created {len(coco_data['annotations'])} annotations for {split} set")

create_coco_annotations(
    root_dir="../Data/hyp_data/",
    output_dir="../Data/hyp_data/annotations/",
    category_name="hypocotyl"
)

# Visualization function to display COCO annotations
from matplotlib import pyplot as plt

def visualize_annotations(image_path, annotation_path, num_samples=5):
    # Load COCO annotations
    with open(annotation_path) as f:
        coco_data = json.load(f)
    
    # sample images
    images = np.random.choice(coco_data['images'], num_samples, replace=False)
    for img_info in images:
        image = cv2.imread(os.path.join(image_path, img_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = [ann for ann in coco_data['annotations'] 
                      if ann['image_id'] == img_info['id']]
        
        # Draw polygons
        for ann in annotations:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2).astype(int)
                cv2.polylines(image, [poly], isClosed=True, color=(255,0,0), thickness=2)
                # to fill the polygon
                # cv2.fillPoly(image, [poly], color=(255, 0, 0))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(f"Image ID: {img_info['id']}")
        plt.axis('off')
        plt.show()


# visualize_annotations(
#     image_path = "../Data/rgb/train/",
#     annotation_path = "../Data/rgb/train/train.json",
#     num_samples=5
# )
