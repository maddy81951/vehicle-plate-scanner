# 🚘 Vehicle Plate Scanner 🔍  
A Python-based license plate detection and recognition system using YOLOv5 for plate localization and EasyOCR for text extraction.

---

## 🚀 Features

- 📸 **YOLOv5 Plate Detection**: Fast and accurate plate detection using pretrained YOLOv5s.
- 🧠 **OCR with EasyOCR**: Recognizes alphanumeric characters from detected plate regions.
- 📦 **Modular Code**: Separate scripts for detection and recognition.
- 🖼️ **Visualization**: Uses matplotlib and OpenCV to display images with results.

---

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/vehicle-plate-scanner.git
cd vehicle-plate-scanner
```

### 2. Set Up Python Environment
Use Python 3.8–3.11 (YOLOv5 may break on 3.12+ for some dependencies)

```bash
pip install -r yolov5/requirements.txt
pip install easyocr matplotlib
```

---

## 📸 Usage

### 1. Detect License Plate with YOLOv5
```bash
python detect_plate_yolo_enhanced.py
```

This will:
- Load `examples/car.jpeg`
- Detect the license plate
- Save cropped plate or draw bounding box
- Save output in `results/`

### 2. Recognize Plate Text (OCR)
Make sure the cropped plate image is correctly extracted.

```bash
python ocr_plate_enhanced.py
```

This will:
- Load the cropped plate image
- Use EasyOCR to recognize text
- Print result to terminal
- Display cropped plate with matplotlib

---

## 🧠 Example Output

```text
Detected Text: WB06A1234 (Confidence: 0.89)
```

---

## 📎 Dependencies

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Torch](https://pytorch.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- OpenCV, Matplotlib

---

## 📌 Notes

- YOLOv5 is cloned inside this repo — you don't need to install it separately.
- Coordinates for cropping can be manually adjusted or automated using bounding box output.

---

## 🧪 To Do / Coming Soon

- [ ] Automatic cropping from YOLO results
- [ ] Support for real-time webcam feed
- [ ] Tesseract OCR as an alternative
- [ ] GUI with Streamlit or Gradio

---

## 📝 License

This project is for educational and research purposes only. Refer to YOLOv5 and EasyOCR repositories for respective licenses.
