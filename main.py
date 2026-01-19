print("Program Started")
from flask import Flask, Response, render_template, send_from_directory, jsonify
import pickle
import json
import h5py
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from datetime import datetime
import math
from ultralytics import YOLO
import cvzone
import threading
import time
from collections import deque
import dlib
print("Imported Libraries...")

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://projectvision908-default-rtdb.firebaseio.com/"
})

app = Flask(__name__, template_folder="C:/PROJECT_NEW/templates")

# Global variables for weapon detection stream
output_frame4 = None
lock4 = threading.Lock()

# Load YOLO model for weapon detection
model_path = "runs/detect/train15/weights/best.pt"
weapon_yolo_model = YOLO(model_path)

# Firebase reference (assuming you're storing data per student)
student_id = ''
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

custom_objects = {
    'Orthogonal': tf.keras.initializers.Orthogonal
}

with h5py.File("D:\\ProjectVision\\LSTM-Actions-Recognition-main\\lstm-model.h5", 'r') as f:
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config)

    for layer in model_config['config']['layers']:
        if 'time_major' in layer['config']:
            del layer['config']['time_major']

    model_json = json.dumps(model_config)
    # model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    model = tf.keras.models.model_from_json(model_json, custom_objects={"Sequential": Sequential})

    weights_group = f['model_weights']
    for layer in model.layers:
        layer_name = layer.name
        if layer_name in weights_group:
            weight_names = weights_group[layer_name].attrs['weight_names']
            layer_weights = [weights_group[layer_name][weight_name] for weight_name in weight_names]
            layer.set_weights(layer_weights)

lm_list = []
label = "neutral"
neutral_label = "neutral"
i = 0
warm_up_frames = 60
global studentInfo, concentration_lists

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load YOLO model
model_yolo = YOLO("../models/best.pt")
classNames = ["fake", "real"]
confidence_threshold = 0.6

# Load the encoding file for face recognition
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Define global variables for multithreading
output_frame1 = None
lock1 = threading.Lock()
output_frame2 = None
lock2 = threading.Lock()
output_frame3 = None
lock3 = threading.Lock()

# Constants for tracking
FOCUS_THRESHOLD = 0.7
FOCUS_DURATION_WINDOW = 30
BLINK_RATE_WINDOW = 30
BLINK_THRESHOLD = 0.2
SMOOTHING_ALPHA = 0.5
HISTORY_SIZE = 30
UPDATE_INTERVAL = 60  # Update interval in seconds

# Initialize tracking variables
focus_duration = 0
gaze_stability_history = deque(maxlen=HISTORY_SIZE)
focus_quality_history = deque(maxlen=HISTORY_SIZE)
blink_rate_history = deque(maxlen=BLINK_RATE_WINDOW)
last_update_time = time.time()
concentration_lists = []
current_concentration = 0
concentration_values = [0] * 7  # Initialize with 7 zeros

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

concentration_values = []


def stream_weapon_detection():
    global output_frame4, lock4
    print("Starting weapon detection stream...")

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO inference
        results = weapon_yolo_model(frame)

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

        # Update the output frame
        with lock4:
            output_frame4 = frame.copy()


def generate_video4():
    global output_frame4, lock4
    while True:
        with lock4:
            if output_frame4 is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', output_frame4)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

def store_values_to_firebase(student_id, concentration_values):
    if not concentration_values:  # Check if the list is empty
        print("Error: concentration_values is empty. Not storing to Firebase.")
        return

    print("Concentration Values :", concentration_values)
    ref = db.reference(f'Students/{student_id}/concentration_data')
    studentInfo = {
        'Stability-focused': concentration_values[0],
        'Focus-quality-focused': concentration_values[1],
        'Balanced metric': concentration_values[2],
        'Stability-weighted': concentration_values[3],
        'Focus-weighted': concentration_values[4],
        'Combined-weighted': concentration_values[5],
        'Simple average': concentration_values[6],
        'last_update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    ref.set(studentInfo)
    print(f"Updated concentration values for student {student_id}")

def preprocess_image(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduced blur for details
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB for face_recognition
    return img

def calculate_gaze_vector(left_eye, right_eye, frame_center):
    left_eye_center = np.mean(np.array(left_eye), axis=0)
    right_eye_center = np.mean(np.array(right_eye), axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2
    gaze_vector = frame_center - eye_center
    norm = np.linalg.norm(gaze_vector)
    gaze_vector_norm = gaze_vector / norm if norm != 0 else np.array([0, 0])
    return gaze_vector_norm, eye_center

def calculate_focus_quality(gaze_vector, ideal_vector):
    similarity = np.dot(gaze_vector, ideal_vector)
    return max(0, min(1, (similarity + 1) / 2))

def detect_blink(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    ear = (A + B) / (2.0 * C)
    return ear < BLINK_THRESHOLD

def generate_concentration_values(stability, focus, blink_rate):
    # Generate 7 different concentration metrics
    values = [
        stability * 0.8 + focus * 0.2,  # Stability-focused
        focus * 0.8 + stability * 0.2,  # Focus-quality-focused
        (stability + focus + (1 - blink_rate)) / 3,  # Balanced metric
        stability * 0.6 + focus * 0.3 + (1 - blink_rate) * 0.1,  # Stability-weighted
        focus * 0.6 + stability * 0.3 + (1 - blink_rate) * 0.1,  # Focus-weighted
        (stability + focus) * 0.45 + (1 - blink_rate) * 0.1,  # Combined-weighted
        np.mean([stability, focus, 1 - blink_rate])  # Simple average
    ]
    return [max(0, min(1, v)) for v in values]  # Ensure values are between 0 and 1

def update_metrics(gaze_vector, previous_gaze_vector, eye_landmarks):
    global last_update_time, focus_duration, current_concentration

    # Calculate stability
    if previous_gaze_vector is not None:
        stability = 1 - min(1, np.linalg.norm(gaze_vector - previous_gaze_vector) * 5)
        gaze_stability_history.append(stability)

    # Calculate focus quality
    ideal_vector = np.array([0, 0])
    focus_quality = calculate_focus_quality(gaze_vector, ideal_vector)
    focus_quality_history.append(focus_quality)

    # Update blink rate
    if detect_blink(eye_landmarks):
        blink_rate_history.append(1)
    else:
        blink_rate_history.append(0)

    # Calculate current metrics for display
    avg_stability = np.mean(list(gaze_stability_history)) if gaze_stability_history else 0
    avg_focus = np.mean(list(focus_quality_history)) if focus_quality_history else 0
    current_blink_rate = np.mean(list(blink_rate_history)) if blink_rate_history else 0

    # Update focus duration
    if avg_focus > FOCUS_THRESHOLD:
        focus_duration += 1 / 30  # Assuming 30 FPS

    # Update current concentration (using balanced metric for real-time display)
    current_concentration = (avg_stability + avg_focus + (1 - current_blink_rate)) / 3

    current_time = time.time()
    if current_time - last_update_time >= UPDATE_INTERVAL:
        # Generate concentration values
        concentration_values = generate_concentration_values(avg_stability, avg_focus, current_blink_rate)
        print("concentration_values", concentration_values)
        concentration_lists.append(concentration_values)

        # Print concentration values with clear labels
        print("\nConcentration Values (Last minute):")
        print("Student ID :".format(student_id))
        print("1. Stability-focused:     {:.3f}".format(concentration_values[0]))
        print("2. Focus-quality-focused: {:.3f}".format(concentration_values[1]))
        print("3. Balanced metric:       {:.3f}".format(concentration_values[2]))
        print("4. Stability-weighted:    {:.3f}".format(concentration_values[3]))
        print("5. Focus-weighted:        {:.3f}".format(concentration_values[4]))
        print("6. Combined-weighted:     {:.3f}".format(concentration_values[5]))
        print("7. Simple average:        {:.3f}".format(concentration_values[6]))
        print("-" * 40)

        last_update_time = current_time
        return concentration_values, focus_duration, current_blink_rate, current_concentration

    return None, focus_duration, current_blink_rate, current_concentration

def process_frame(frame, previous_gaze_vector):
    frame_resized = cv2.resize(frame, (640, 480))
    if frame_resized.size > 0:
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        frame_center = np.array([frame_resized.shape[1] / 2, frame_resized.shape[0] / 2])
        eye_landmarks = None
        current_gaze_vector = previous_gaze_vector
    else:
        pass
        # print("Cropped image is empty. Skipping this frame.")
        # return frame_resized, previous_gaze_vector, None

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        gaze_vector_norm, _ = calculate_gaze_vector(left_eye, right_eye, frame_center)
        current_gaze_vector = gaze_vector_norm
        eye_landmarks = left_eye + right_eye

    return frame_resized, current_gaze_vector, eye_landmarks


def landmarks_to_image(lm_list, image_shape=(20, 63)):
    """
    Converts a list of pose landmarks to a feature vector.
    :param lm_list: A list of landmarks (x, y, z, visibility)
    :param image_shape: Shape of the output feature vector (height, width)
    :return: Feature vector of shape (1, 20, 63)
    """
    # Truncate or pad the landmarks to match the expected shape
    if len(lm_list) > 20:
        lm_list = lm_list[:20]

    # Pad if less than 20 timesteps
    while len(lm_list) < 20:
        lm_list.append([0] * 4)  # Pad with zeros

    # Reduce to 63 features (x, y, z, visibility for each landmark)
    feature_vector = []
    for timestep in lm_list:
        # Flatten and take only the first 63 values
        flat_timestep = timestep[:63] if len(timestep) >= 63 else timestep + [0] * (63 - len(timestep))
        feature_vector.append(flat_timestep)

    # Convert to numpy array and reshape
    feature_vector = np.array(feature_vector, dtype=np.float32)
    feature_vector = feature_vector.reshape((1, 20, 63))

    return feature_vector

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def align_face(img, landmarks):
    left_eye_center = np.mean(np.array([(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y)]), axis=0).astype(int)
    right_eye_center = np.mean(np.array([(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y)]), axis=0).astype(int)

    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_img = cv2.warpAffine(img, matrix, (width, height))
    return aligned_img

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

# def detect(model, lm_list):
#     global label
#     image_input = landmarks_to_image(lm_list)  # Ensure this returns the correct shape
#     print("Input shape:", image_input.shape)  # Debugging: Check the shape
#     if image_input.shape != (1, 20, 132):  # Ensure the shape is correct
#         print(f"Unexpected input shape: {image_input.shape}")
#         return
#     result = model.predict(image_input)
#     if len(result) > 0 and len(result[0]) > 0 and result[0][0] > 0.5:
#         label = "violent"
#     else:
#         label = "neutral"
#     return str(label)

def detect(model, lm_list):
    global label
    try:
        # Convert landmarks to input format
        image_input = landmarks_to_image(lm_list)

        # Verify input shape
        print(f"Model input shape: {model.input_shape}")
        print(f"Current input shape: {image_input.shape}")

        # Predict
        result = model.predict(image_input)

        # Print prediction result for debugging
        print(f"Prediction result: {result}")

        # Process result
        if len(result) > 0 and len(result[0]) > 0 and result[0][0] > 0.5:
            label = "violent"
        else:
            label = "neutral"

        print(f"Detected label: {label}")
        return str(label)

    except Exception as e:
        print(f"Prediction error: {e}")
        # Print full traceback for more detailed error information
        import traceback
        traceback.print_exc()
        return "neutral"

def stream1():
    print("Face Recognition Called")
    global output_frame1, lock1, student_id, concentration_values
    last_attendance_check = time.time()
    attendance_times = {}
    while True:
        success, img = cap.read()
        if not success:
            continue

        # Resize for faster processing
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect face locations in the smaller image
        faceCurFrame = face_recognition.face_locations(imgS_rgb)
        encodeCurFrame = face_recognition.face_encodings(imgS_rgb, faceCurFrame)

        # Create a copy of the original image for drawing
        img_display = img.copy()

        # First run YOLO to detect faces (helps with spoofing detection)
        results = model_yolo(img, stream=True, verbose=False)

        # Default state
        name = "Unknown"
        box_color = (0, 0, 255)  # Red for unknown/fake faces
        student_id = None
        is_real_face = False

        # Process YOLO results first
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy()).astype(int)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Check face classification
                class_name = classNames[cls]

                if conf > confidence_threshold:
                    # Calculate face size to determine proximity
                    face_area = w * h
                    face_size_threshold = 50000  # Adjust this value based on your camera and setup

                    if face_area > face_size_threshold:
                        # Only do fake detection when face is very close to camera
                        if class_name == "fake":
                            # Draw red box and write FAKE
                            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img_display, "FAKE", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            is_real_face = True

                            # Compare with known faces
                            faceCurFrame = face_recognition.face_locations(imgS_rgb)
                            encodeCurFrame = face_recognition.face_encodings(imgS_rgb, faceCurFrame)
                            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                                encodeFace = encodeCurFrame[0]
                                faceLoc = faceCurFrame[0]
                                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                                # Convert face_location back to original image size
                                top, right, bottom, left = faceLoc
                                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                                if len(faceDis) > 0:
                                    matchIndex = np.argmin(faceDis)
                                    if matches[matchIndex]:
                                        student_id = studentIds[matchIndex]
                                        studentInfo = db.reference(f'Students/{student_id}').get()

                                        if studentInfo:
                                            name = studentInfo.get('name', 'Unknown')
                                            cv2.rectangle(img_display, (left, top), (right, bottom), box_color, 2)
                                            cv2.rectangle(img_display, (left, bottom - 35), (right, bottom), box_color,
                                                          cv2.FILLED)
                                            cv2.putText(img_display, name, (left + 6, bottom - 6),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        elif class_name == "real" and faceCurFrame:
                            # Process only real faces
                            is_real_face = True

                            # Take the first face
                            encodeFace = encodeCurFrame[0]
                            faceLoc = faceCurFrame[0]

                            # Convert face_location back to original image size
                            top, right, bottom, left = faceLoc
                            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                            # Compare with known faces
                            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                            # If we have a match with a known face
                            if len(faceDis) > 0:
                                matchIndex = np.argmin(faceDis)
                                if matches[matchIndex]:
                                    student_id = studentIds[matchIndex]
                                    studentInfo = db.reference(f'Students/{student_id}').get()

                                    if studentInfo:
                                        name = studentInfo.get('name', 'Unknown')
                                        box_color = (0, 255, 0)  # Green for known faces

                                        # Check if it's time to update attendance
                                        current_time = time.time()
                                        if student_id not in attendance_times or (
                                                current_time - attendance_times.get(student_id, 0)) > 30:
                                            # Update attendance in Firebase (once every 30 seconds per student)
                                            attendance_ref = db.reference(f'Students/{student_id}')
                                            attendance_info = {
                                                'last_attendance_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'total_attendance': studentInfo.get('total_attendance', 0) + 1
                                            }
                                            attendance_ref.update(attendance_info)
                                            attendance_times[student_id] = current_time
                                            print(f"Attendance updated for {name}")

                                            # If we have concentration values, store them
                                            if concentration_values:
                                                store_values_to_firebase(student_id, concentration_values)

                            # Draw rectangle and name for real faces
                            if is_real_face:
                                cv2.rectangle(img_display, (left, top), (right, bottom), box_color, 2)
                                cv2.rectangle(img_display, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                                cv2.putText(img_display, name, (left + 6, bottom - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update the output frame
        with lock1:
            output_frame1 = img_display.copy()

        # Display the result
        # cv2.imshow("Face Attendance", img_display)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

def stream2():
    print("Stream 2 Called ...")
    global output_frame2, lock2, lm_list  # Add lm_list to the global scope

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # Skip the frame if reading fails
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)  # This now works because lm_list is global

            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []  # Clear lm_list after detection

            x_coordinate = []
            y_coordinate = []
            for lm in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)

            cv2.rectangle(frame, (min(x_coordinate), max(y_coordinate)), (max(x_coordinate), min(y_coordinate) - 25), (0, 255, 0), 1)
            frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_class_on_image(label, frame)

        with lock2:
            output_frame2 = frame.copy()

        # cv2.imshow("Pose Detection", frame)

        # if cv2.waitKey(1) == ord('q'):
        #     break
    cap.release()
    cv2.destroyAllWindows()

def stream_gaze_detection():
    global output_frame3, lock3, current_concentration
    previous_gaze_vector = None
    print("Starting concentration monitoring...")
    print("Press 'q' to quit")
    print("-" * 40)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, current_gaze_vector, eye_landmarks = process_frame(frame, previous_gaze_vector)

        # Create a copy for displaying metrics
        display_frame = processed_frame.copy()

        if eye_landmarks and current_gaze_vector is not None:
            concentration_values_update, focus_duration, blink_rate, concentration = update_metrics(
                current_gaze_vector, previous_gaze_vector, eye_landmarks
            )

            # Display metrics on frame
            cv2.putText(display_frame, f"Focus Duration: {focus_duration:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Blink Rate: {blink_rate:.2f}/s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Concentration: {concentration:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            previous_gaze_vector = current_gaze_vector

        # Update the output frame with the frame that has metrics drawn on it
        with lock3:
            output_frame3 = display_frame.copy()

        # cv2.imshow("Concentration Monitoring", display_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()

def generate_video1():
    global output_frame1, lock1
    while True:
        with lock1:
            if output_frame1 is not None:
                # Ensure the frame is properly encoded
                try:
                    _, buffer = cv2.imencode('.jpg', output_frame1)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    # Add a small delay to prevent busy waiting
                    time.sleep(0.01)
            else:
                # Add a small delay to prevent busy waiting
                time.sleep(0.01)

# Apply the same fixes to the other generate functions
def generate_video2():
    global output_frame2, lock2
    while True:
        with lock2:
            if output_frame2 is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', output_frame2)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

def generate_video3():
    global output_frame3, lock3
    while True:
        with lock3:
            if output_frame3 is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', output_frame3)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('Gp/assets', filename)

@app.route('/')
def index1():
    print("Main Template called")
    return send_from_directory('Gp', "index.html")

@app.route('/gp/index1')
def index():
    return render_template('index1.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_video1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_video2(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed3')
def video_feed3():
    return Response(generate_video3(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Add these routes to your existing Flask app
@app.route('/video_feed4')
def video_feed4():
    return Response(generate_video4(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_user_data')
def get_user_data():
    global output_frame1, lock1
    with lock1:
        if output_frame1 is None:
            return jsonify({
                "detected": False,
                "message": "No face detected."
            })

        # Use the logic from stream1 to extract and return user data
        success, img = cap.read()
        if not success:
            return jsonify({
                "detected": False,
                "message": "Failed to capture frame."
            })

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    id = studentIds[matchIndex]
                    studentInfo = db.reference(f'Students/{id}').get()
                    return jsonify({
                        "detected": True,
                        "id": id,
                        "name": studentInfo.get('name', "Unknown"),
                        "status": "Active",
                        "total_attendance": studentInfo.get('total_attendance', 0),
                        "last_attendance_time": studentInfo.get('last_attendance_time', "Never")
                    })
        return jsonify({
            "detected": False,
            "message": "No known face detected."
        })

@app.route('/run_noter_script', methods=['POST'])
def run_noter_script():
    print("NOTER script executed!")
    try:
        # Your script logic here
        return render_template("SpeechToText.html")  # Make sure this matches your file name
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

# Add a new route to handle the speech to text page
@app.route('/gp/SpeechToText.html')
def speech_to_text():
    return render_template("SpeechToText.html")

@app.route('/activate', methods=['POST'])
def activate_system():
    try:
        # Your activation logic here
        return jsonify({
            "success": True,
            "message": "System activated successfully!"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

if __name__ == '__main__':
    t1 = threading.Thread(target=stream1)
    t2 = threading.Thread(target=stream2)
    t3 = threading.Thread(target=stream_gaze_detection)
    t4 = threading.Thread(target=stream_weapon_detection)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    app.run(host="0.0.0.0", port=500)