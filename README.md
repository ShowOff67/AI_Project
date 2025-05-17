# Real-Time Sign Language Detection System

## Overview
This project aims to develop a real-time sign language detection system that can recognize hand gestures and convert them into readable text. The system leverages a pre-trained TensorFlow object detection model to identify gestures from a live webcam feed.

## Features
- Collect images for custom gesture labels.
- Use a pre-trained demo model to detect objects like cat, dog, person, etc.
- Real-time detection using TensorFlow and OpenCV.
- Supports 5 sign language classes: hello, thanks, yes, no, iloveyou.

## Requirements
- Python 3.10.0
- TensorFlow 2.10.0
- OpenCV
- Virtual environment (venv) recommended

## Setup Instructions

1. Clone the repository or download the project files.
2. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```
3. Install required packages:
   ```bash
   pip install tensorflow==2.10.0 opencv-python
   ```
4. Collect images for your custom sign labels using the provided script:
   ```bash
   python collect_images.py
   ```
   This will save images for the labels: hello, thanks, yes, no, iloveyou.
5. Prepare the dataset and convert it to TFRecord format (not included in this repo).
6. Run the detection script to use the pre-trained demo model and perform live detection:
   ```bash
   python detect_live.py
   ```

## Usage

- Run `collect_images.py` to gather image data for training.
- Use the detection script to test on live webcam input.
- Modify and train your own model if hardware permits.

## Limitations

- Training the model is not included due to hardware constraints.
- Only 5 gesture classes supported.
- Detection uses a pre-trained COCO demo model, so gesture recognition accuracy may vary.

## References

- [Project Dataset and Documentation](https://drive.google.com/file/d/1ofbqUVqFKYyfIAX0wbZmbe0d8FR9wcAV/view?usp=drive_link)
- [YouTube Tutorial](https://youtu.be/IOI0o3Cxv9Q?si=cEG83wnu3obfbFjL)

---

Feel free to contribute or raise issues for improvements!
