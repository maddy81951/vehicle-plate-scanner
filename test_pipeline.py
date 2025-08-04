import os
import sys
from license_plate_pipeline import LicensePlatePipeline

def test_pipeline():
    print("Testing License Plate Scanner Pipeline")
    print("=" * 50)
    
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        print(f"Examples directory not found: {examples_dir}")
        print("Please create an 'examples' directory and add test images")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_images = []
    
    for filename in os.listdir(examples_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(examples_dir, filename))
    
    if not test_images:
        print(f"No test images found in {examples_dir}")
        print("Supported formats: jpg, jpeg, png, bmp, tiff")
        print("Please add at least one car image to the examples directory")
        return
    
    print(f"Found {len(test_images)} test image(s):")
    for img in test_images:
        print(f"  - {img}")
    print()
    
    try:
        pipeline = LicensePlatePipeline()
        
        all_results = []
        total_plates = 0
        
        for i, test_image in enumerate(test_images, 1):
            print(f"[{i}/{len(test_images)}] Testing: {os.path.basename(test_image)}")
            print("-" * 30)
            
            try:
                result = pipeline.process_image(test_image)
                all_results.append(result)
                
                plates_found = result['detected_plates']
                total_plates += plates_found
                
                print(f"Detection completed successfully")
                print(f"Found {plates_found} license plate(s)")
                
                if plates_found > 0:
                    for j, plate in enumerate(result['plates']):
                        text = plate['recognized_text']
                        ocr_conf = plate['ocr_confidence']
                        det_conf = plate['detection_confidence']
                        print(f"  Plate {j+1}: '{text}' (OCR: {ocr_conf:.2f}, Detection: {det_conf:.2f})")
                else:
                    print("  No license plates detected in this image")
                
                print()
                
            except Exception as e:
                print(f"Error processing {os.path.basename(test_image)}: {str(e)}")
                print()
        
        print("=" * 50)
        print("PIPELINE TEST SUMMARY")
        print("=" * 50)
        print(f"Pipeline test completed successfully!")
        print(f"Images processed: {len(test_images)}")
        print(f"Total license plates found: {total_plates}")
        print(f"Average plates per image: {total_plates/len(test_images):.1f}")
        
        if total_plates > 0:
            print(f"\nAll detected license plates:")
            for i, result in enumerate(all_results, 1):
                if result['detected_plates'] > 0:
                    print(f"  Image {i} ({os.path.basename(test_images[i-1])}):")
                    for plate in result['plates']:
                        print(f"    - {plate['recognized_text']}")
        
        print(f"\nResults saved in: results/ directory")
        
    except Exception as e:
        print(f"Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def list_available_images():
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        print(f"Examples directory not found: {examples_dir}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    
    for filename in os.listdir(examples_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            images.append(filename)
    
    if images:
        print(f"Available test images in {examples_dir}:")
        for img in images:
            print(f"  - {img}")
    else:
        print(f"No images found in {examples_dir}")
        print("Please add car images with these formats: jpg, jpeg, png, bmp, tiff")

if __name__ == "__main__":
    list_available_images()
    print()
    
    test_pipeline()