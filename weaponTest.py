import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

# Load YOLO model
model_path = "runs/detect/train15/weights/best.pt"
yolo_model = YOLO(model_path)

# Open webcam
video_capture = cv2.VideoCapture(0)


def update_frame():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        return

    # Run YOLO inference
    results = yolo_model(frame)

    # Process detections
    for result in results:
        if result.boxes is not None:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.6:
                    xmin, ymin, xmax, ymax = map(int, detection)
                    class_name = classes[int(cls[pos])]
                    confidence = conf[pos]

                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)

    # Convert frame to Qt format and update label
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    video_label.setPixmap(pixmap.scaled(video_label.size(), Qt.KeepAspectRatio))


# Setup UI
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Weapon Detection")
window.setGeometry(100, 100, 800, 600)
layout = QVBoxLayout()
video_label = QLabel()
layout.addWidget(video_label)
window.setLayout(layout)

# Timer for updating frames
timer = QTimer()
timer.timeout.connect(update_frame)
timer.start(30)

# Show window
window.show()
sys.exit(app.exec_())