import os
import cv2
import json
from datetime import datetime
from detect_plate_yolo_enhanced import LicensePlateDetector
from ocr_plate_enhanced import LicensePlateOCR

class LicensePlatePipeline:
    def __init__(self, model_path=None, confidence_threshold=0.05):
        self.detector = LicensePlateDetector(model_path, confidence_threshold)
        self.ocr = LicensePlateOCR()
        
    def process_image(self, image_path, output_dir="results", save_intermediates=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(output_dir, f"result_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        print("Step 1: Detecting license plates...")
        detected_plates = self.detector.detect_plates(
            image_path, 
            save_crops=save_intermediates,
            output_dir=os.path.join(result_dir, "detected_plates")
        )
        
        print("Step 2: Performing OCR...")
        results = {
            'input_image': image_path,
            'timestamp': timestamp,
            'detected_plates': len(detected_plates),
            'plates': []
        }
        
        for idx, plate_info in enumerate(detected_plates):
            print(f"Processing plate {idx + 1}/{len(detected_plates)}")
            
            if plate_info['image_path']:
                ocr_result = self.ocr.extract_text_from_image(plate_info['image_path'])
            else:
                ocr_result = self.ocr.extract_text_from_image(plate_info['cropped_image'])
            
            plate_result = {
                'plate_id': idx,
                'detection_confidence': float(plate_info['confidence']),
                'bbox': plate_info['bbox'],
                'recognized_text': ocr_result['best_text'],
                'ocr_confidence': float(ocr_result['confidence']),
                'cropped_image_path': plate_info['image_path'],
                'all_ocr_results': ocr_result
            }
            
            results['plates'].append(plate_result)
            print(f"  Detected text: '{ocr_result['best_text']}' (confidence: {ocr_result['confidence']:.2f})")
        
        if save_intermediates:
            results_file = os.path.join(result_dir, "results.json")
            with open(results_file, 'w') as f:
                json_results = results.copy()
                for plate in json_results['plates']:
                    if 'all_ocr_results' in plate:
                        del plate['all_ocr_results']
                json.dump(json_results, f, indent=2)
            print(f"Results saved to: {results_file}")
        
        return results
    
    def process_batch(self, input_directory, output_dir="batch_results"):
        all_results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(output_dir, f"batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        for filename in os.listdir(input_directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(input_directory, filename)
                print(f"\nProcessing: {filename}")
                
                try:
                    result = self.process_image(
                        image_path,
                        output_dir=batch_output_dir,
                        save_intermediates=True
                    )
                    result['filename'] = filename
                    all_results.append(result)
                    
                    print(f"Found {result['detected_plates']} plates in {filename}")
                    for plate in result['plates']:
                        print(f"  - '{plate['recognized_text']}' (confidence: {plate['ocr_confidence']:.2f})")
                        
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        summary_file = os.path.join(batch_output_dir, "batch_summary.json")
        with open(summary_file, 'w') as f:
            summary = []
            for result in all_results:
                summary_result = {
                    'filename': result['filename'],
                    'detected_plates': result['detected_plates'],
                    'plates': [
                        {
                            'recognized_text': plate['recognized_text'],
                            'detection_confidence': plate['detection_confidence'],
                            'ocr_confidence': plate['ocr_confidence']
                        }
                        for plate in result['plates']
                    ]
                }
                summary.append(summary_result)
            
            json.dump(summary, f, indent=2)
        
        print(f"\nBatch processing complete. Results saved to: {batch_output_dir}")
        return all_results

"""if __name__ == "__main__":
    pipeline = LicensePlatePipeline()
    
    image_path = "examples"
    if os.path.exists(image_path):
        print("Processing single image...")
        result = pipeline.process_image(image_path)
        print(f"\nSummary:")
        print(f"Detected {result['detected_plates']} license plates")
        for plate in result['plates']:
            print(f"- Text: '{plate['recognized_text']}' (OCR confidence: {plate['ocr_confidence']:.2f})")
    else:
        print(f"Image not found: {image_path}")
    
    if os.path.exists("examples") and len(os.listdir("examples")) > 1:
        print("\nProcessing batch...")
        batch_results = pipeline.process_batch("examples")
        print(f"Processed {len(batch_results)} images in batch")"""
if __name__ == "__main__":
    pipeline = LicensePlatePipeline()
    
    input_directory = "examples"  
    
    if os.path.exists(input_directory):
        print("Processing multiple car images...")
        batch_results = pipeline.process_batch(input_directory)
        
        print(f"\n=== SUMMARY ===")
        print(f"Processed {len(batch_results)} car images")
        
        for result in batch_results:
            print(f"\nCar Image: {result['filename']}")
            print(f"License plates found: {result['detected_plates']}")
            for i, plate in enumerate(result['plates']):
                print(f"  Plate {i+1}: '{plate['recognized_text']}' (confidence: {plate['ocr_confidence']:.2f})")