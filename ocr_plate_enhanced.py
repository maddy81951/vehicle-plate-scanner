import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from typing import List, Tuple, Dict

class LicensePlateOCR:
    def __init__(self, languages=['en'], gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def clean_text(self, text):
        cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        corrections = {
            'O': '0',  
            'I': '1', 
            'S': '5',  
            'B': '8',  
        }
        
        result = cleaned
        for old, new in corrections.items():
            pass  
        
        return result
    
    def extract_text_from_image(self, image_path, show_preprocessing=False):
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            image = image_path 
        
        processed_image = self.preprocess_image(image)
        
        if show_preprocessing:
            self.show_preprocessing_steps(image, processed_image)
        
        original_results = self.reader.readtext(image)
        processed_results = self.reader.readtext(processed_image)
        
        all_results = {
            'original': self.parse_ocr_results(original_results),
            'processed': self.parse_ocr_results(processed_results),
            'best_text': None,
            'confidence': 0.0
        }
        
        best_result = self.select_best_result(all_results['original'], all_results['processed'])
        all_results['best_text'] = best_result['text']
        all_results['confidence'] = best_result['confidence']
        
        return all_results
    
    def parse_ocr_results(self, ocr_results):
        parsed_results = []
        for bbox, text, confidence in ocr_results:
            cleaned_text = self.clean_text(text)
            if len(cleaned_text) > 0:
                parsed_results.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'original_text': text
                })
        return parsed_results
    
    def select_best_result(self, original_results, processed_results):
        all_candidates = original_results + processed_results
        
        if not all_candidates:
            return {'text': '', 'confidence': 0.0}
        
        scored_results = []
        for result in all_candidates:
            score = result['confidence']
            text = result['text']
            
            if re.match(r'^[A-Z0-9]{5,8}$', text):
                score += 0.2
            if re.match(r'^[A-Z]{2,3}[0-9]{2,4}[A-Z]?$', text): 
                score += 0.3
            
            if len(text) < 4 or len(text) > 10:
                score -= 0.1
            
            scored_results.append({
                'text': text,
                'confidence': result['confidence'],
                'score': score,
                'original_text': result['original_text']
            })
        
        best = max(scored_results, key=lambda x: x['score'])
        return best
    
    def show_preprocessing_steps(self, original, processed):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title("Preprocessed Image")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def batch_process_directory(self, directory_path, output_file=None):
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                try:
                    result = self.extract_text_from_image(image_path)
                    result['filename'] = filename
                    results.append(result)
                    print(f"Processed {filename}: {result['best_text']} (confidence: {result['confidence']:.2f})")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        if output_file:
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"{result['filename']}: {result['best_text']} ({result['confidence']:.2f})\n")
        
        return results

if __name__ == "__main__":
    ocr = LicensePlateOCR()
    
    image_path = "detected_plates/plate_0_0.85.jpg" 
    if os.path.exists(image_path):
        result = ocr.extract_text_from_image(image_path, show_preprocessing=True)
        print(f"Detected text: {result['best_text']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"Image not found: {image_path}")
        
    if os.path.exists("detected_plates"):
        batch_results = ocr.batch_process_directory("detected_plates", "ocr_results.txt")
        print(f"Processed {len(batch_results)} images")