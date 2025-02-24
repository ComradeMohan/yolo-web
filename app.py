from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import cv2
import os
import math
import cvzone

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# COCO classes
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", 
               "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
               "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
               "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
               "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
               "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", 
               "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", 
               "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
               "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run YOLOv8 on the uploaded image
        img = cv2.imread(filepath)
        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label = f"{class_names[cls]} {conf}"
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                cvzone.putTextRect(img, label, (x1, y1), scale=1, thickness=1)

        # Save processed image
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{file.filename}")
        cv2.imwrite(processed_path, img)

        return redirect(url_for('display_image', filename=f"processed_{file.filename}"))

@app.route('/display/<filename>')
def display_image(filename):
    return render_template('index.html', filename=filename)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
