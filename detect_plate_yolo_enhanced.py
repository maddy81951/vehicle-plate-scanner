import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path

class LicensePlateDetector:
    def __init__(self, model_path=None, confidence_threshold=0.15):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=model_path, force_reload=True)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                      pretrained=True)
        
        self.model.to(self.device)
        self.model.conf = confidence_threshold
    
    def detect_plates(self, image_path, save_crops=True, output_dir='detected_plates'):
        if save_crops:
            os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.model(img_rgb)
        
        detections = results.pandas().xyxy[0]
        plate_info = []
        
        print(f"Found {len(detections)} potential license plates")
        
        for idx, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                            int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            
            cropped_plate = img[y1:y2, x1:x2]
            
            plate_data = {
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'cropped_image': cropped_plate,
                'image_path': None
            }
            
            if save_crops and cropped_plate.size > 0:
                crop_filename = f"plate_{idx}_{confidence:.2f}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_plate)
                plate_data['image_path'] = crop_path
                print(f"Saved cropped plate: {crop_path}")
            
            plate_info.append(plate_data)
        
        self.display_results(img_rgb, plate_info)
        
        return plate_info
    
    def display_results(self, image, plate_info):
        fig, axes = plt.subplots(1, len(plate_info) + 1, figsize=(15, 5))
        if len(plate_info) == 0:
            axes = [axes]
        elif len(plate_info) == 1:
            axes = [axes[0], axes[1]]
        
        img_with_boxes = image.copy()
        for plate in plate_info:
            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{plate['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[0].imshow(img_with_boxes)
        axes[0].set_title("Detected Plates")
        axes[0].axis('off')
        
        for idx, plate in enumerate(plate_info):
            if idx + 1 < len(axes):
                cropped_rgb = cv2.cvtColor(plate['cropped_image'], cv2.COLOR_BGR2RGB)
                axes[idx + 1].imshow(cropped_rgb)
                axes[idx + 1].set_title(f"Plate {idx + 1}")
                axes[idx + 1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    detector = LicensePlateDetector()
    
    image_path = "examples"
    if os.path.exists(image_path):
        plates = detector.detect_plates(image_path)
        print(f"Detected {len(plates)} license plates")
    else:
        print(f"Image not found: {image_path}")