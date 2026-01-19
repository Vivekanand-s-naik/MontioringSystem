print("Program Started")
from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import math
from ultralytics import YOLO
import face_recognition
import threading
import time

print("Imported Libraries...")

app = Flask(__name__)

# Global variables for video stream
output_frame = None
lock = threading.Lock()

# Load YOLO model for real/fake classification
model_yolo = YOLO("../models/best.pt")
classNames = ["fake", "real"]
confidence_threshold = 0.6

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Detection statistics
detection_stats = {
	"total_faces_detected": 0,
	"real_faces": 0,
	"fake_faces": 0,
	"last_detection_time": None
}


def process_frame():
	"""Main processing function for face detection and classification"""
	global output_frame, lock, detection_stats

	while True:
		success, img = cap.read()
		if not success:
			continue

		# Create a copy for display
		img_display = img.copy()

		# Detect faces using face_recognition
		imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
		imgS_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

		face_locations = face_recognition.face_locations(imgS_rgb)

		# Process YOLO results for real/fake classification
		results = model_yolo(img, stream=True, verbose=False)

		for r in results:
			boxes = r.boxes
			for box in boxes:
				x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy()).astype(int)
				w, h = x2 - x1, y2 - y1

				conf = math.ceil((box.conf[0] * 100)) / 100
				cls = int(box.cls[0])
				class_name = classNames[cls]

				if conf > confidence_threshold:
					# Update statistics
					detection_stats["total_faces_detected"] += 1
					detection_stats["last_detection_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

					if class_name == "real":
						detection_stats["real_faces"] += 1
						box_color = (0, 255, 0)  # Green for real
						label = f"REAL ({conf:.2f})"
					else:
						detection_stats["fake_faces"] += 1
						box_color = (0, 0, 255)  # Red for fake
						label = f"FAKE ({conf:.2f})"

					# Draw bounding box
					cv2.rectangle(img_display, (x1, y1), (x2, y2), box_color, 2)

					# Draw label background
					cv2.rectangle(img_display, (x1, y1 - 35), (x2, y1), box_color, cv2.FILLED)

					# Draw label text
					cv2.putText(img_display, label, (x1 + 6, y1 - 10),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Draw face count on frame
		cv2.putText(img_display, f"Faces Detected: {len(face_locations)}",
		            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# Update the output frame
		with lock:
			output_frame = img_display.copy()

		time.sleep(0.01)  # Small delay to prevent CPU overload


def generate_video():
	"""Generate video stream for web display"""
	global output_frame, lock

	while True:
		with lock:
			if output_frame is not None:
				try:
					_, buffer = cv2.imencode('.jpg', output_frame)
					frame = buffer.tobytes()
					yield (b'--frame\r\n'
					       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
				except Exception as e:
					print(f"Error encoding frame: {e}")
					time.sleep(0.01)
			else:
				time.sleep(0.01)


@app.route('/')
def index():
	"""Main page route"""
	return render_template('index.html')


@app.route('/video_feed')
def video_feed():
	"""Video streaming route"""
	return Response(generate_video(),
	                mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_stats')
def get_stats():
	"""Get detection statistics"""
	return jsonify(detection_stats)


@app.route('/reset_stats')
def reset_stats():
	"""Reset detection statistics"""
	global detection_stats
	detection_stats = {
		"total_faces_detected": 0,
		"real_faces": 0,
		"fake_faces": 0,
		"last_detection_time": None
	}
	return jsonify({"message": "Statistics reset successfully"})


if __name__ == '__main__':
	# Start the video processing thread
	t1 = threading.Thread(target=process_frame, daemon=True)
	t1.start()

	# Run Flask app
	app.run(host="0.0.0.0", port=9000, debug=False)

	print("Program Finished")