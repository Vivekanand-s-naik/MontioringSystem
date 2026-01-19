import asyncio
import cv2
import numpy as np
import face_recognition
import firebase_admin
import time
from datetime import datetime
from flask import Flask, Response, render_template, send_from_directory, jsonify
from quart import Quart, websocket
import pickle
import json
import h5py
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
import dlib
import concurrent.futures

# Replace Flask with Quart for async support
app = Quart(__name__)

# Create thread pools for CPU and I/O bound tasks
cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Frame buffers
latest_frame = None
face_recognition_frame = None
pose_detection_frame = None
gaze_detection_frame = None

# Locks for thread safety (still needed for executor communication)
frame_lock = asyncio.Lock()
face_lock = asyncio.Lock()
pose_lock = asyncio.Lock()
gaze_lock = asyncio.Lock()

# Configuration
FACE_PROCESSING_INTERVAL = 3  # Process every Nth frame
POSE_PROCESSING_INTERVAL = 2
GAZE_PROCESSING_INTERVAL = 2
RESOLUTION = (320, 240)  # Lower resolution for processing
FRAME_RATE = 15


# Camera initialization
async def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, RESOLUTION[0])
    cap.set(4, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    return cap.isOpened()


# Frame capture coroutine
async def capture_frames():
    global latest_frame, cap
    print("Starting asynchronous frame capture...")

    while True:
        # Run the actual frame capture in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        success, frame = await loop.run_in_executor(io_executor, lambda: cap.read())

        if success:
            async with frame_lock:
                latest_frame = frame.copy()

        # Small sleep to prevent CPU overuse
        await asyncio.sleep(1 / FRAME_RATE)


# Face recognition coroutine
async def process_face_recognition():
    global latest_frame, face_recognition_frame, student_id

    print("Starting asynchronous face recognition...")
    frame_counter = 0

    while True:
        frame_counter += 1
        # Get a copy of the current frame
        async with frame_lock:
            if latest_frame is None:
                await asyncio.sleep(0.01)
                continue
            current_frame = latest_frame.copy()

        # Only process every Nth frame
        if frame_counter % FACE_PROCESSING_INTERVAL == 0:
            # Run face recognition in the thread pool (CPU-bound)
            loop = asyncio.get_event_loop()

            # Resize image in the thread pool
            imgS = await loop.run_in_executor(
                cpu_executor,
                lambda: cv2.resize(current_frame, (0, 0), None, 0.25, 0.25)
            )

            imgS_rgb = await loop.run_in_executor(
                cpu_executor,
                lambda: cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            )

            # Face locations using HOG (faster)
            faceCurFrame = await loop.run_in_executor(
                cpu_executor,
                lambda: face_recognition.face_locations(imgS_rgb, model="hog")
            )

            # Process detected faces
            img_display = current_frame.copy()

            if faceCurFrame:
                # Get face encodings only if faces detected
                encodeCurFrame = await loop.run_in_executor(
                    cpu_executor,
                    lambda: face_recognition.face_encodings(imgS_rgb, faceCurFrame, num_jitters=1)
                )

                # Process the face recognition results asynchronously
                for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                    # Process face detection asynchronously
                    # Your face detection code here, similar to your original code
                    # but using await for any I/O operations
                    pass

            # Update the output frame
            async with face_lock:
                face_recognition_frame = img_display.copy()

        # Allow other coroutines to run
        await asyncio.sleep(0.01)


# Pose detection coroutine
async def process_pose_detection():
    global latest_frame, pose_detection_frame, lm_list, pose, mpDraw

    print("Starting asynchronous pose detection...")
    frame_counter = 0

    while True:
        frame_counter += 1
        # Get a copy of the current frame
        async with frame_lock:
            if latest_frame is None:
                await asyncio.sleep(0.01)
                continue
            current_frame = latest_frame.copy()

        # Only process every Nth frame
        if frame_counter % POSE_PROCESSING_INTERVAL == 0:
            # Run pose detection in the thread pool
            loop = asyncio.get_event_loop()

            # Convert image to RGB for pose detection
            frameRGB = await loop.run_in_executor(
                cpu_executor,
                lambda: cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            )

            # Run pose detection
            results = await loop.run_in_executor(
                cpu_executor,
                lambda: pose.process(frameRGB)
            )

            # Process the pose results
            if results.pose_landmarks:
                # Process landmarks
                # Your pose detection code here, similar to your original code
                # but using await for any I/O operations
                pass

            # Update the output frame
            async with pose_lock:
                pose_detection_frame = current_frame.copy()

        # Allow other coroutines to run
        await asyncio.sleep(0.01)


# Gaze detection coroutine
async def process_gaze_detection():
    global latest_frame, gaze_detection_frame, face_detector, landmark_predictor

    print("Starting asynchronous gaze detection...")
    frame_counter = 0
    previous_gaze_vector = None

    while True:
        frame_counter += 1
        # Get a copy of the current frame
        async with frame_lock:
            if latest_frame is None:
                await asyncio.sleep(0.01)
                continue
            current_frame = latest_frame.copy()

        # Only process every Nth frame
        if frame_counter % GAZE_PROCESSING_INTERVAL == 0:
            # Run gaze detection in the thread pool
            loop = asyncio.get_event_loop()

            # Convert image to grayscale for face detection
            gray = await loop.run_in_executor(
                cpu_executor,
                lambda: cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            )

            # Detect faces using dlib
            faces = await loop.run_in_executor(
                cpu_executor,
                lambda: face_detector(gray)
            )

            # Process the gaze detection results
            frame_center = np.array([current_frame.shape[1] / 2, current_frame.shape[0] / 2])

            for face in faces:
                # Process face for gaze detection
                # Your gaze detection code here, similar to your original code
                # but using await for any I/O operations
                pass

            # Update the output frame
            async with gaze_lock:
                gaze_detection_frame = current_frame.copy()

        # Allow other coroutines to run
        await asyncio.sleep(0.01)


# Generate video feed for flask response
async def generate_video_feed(frame_type):
    while True:
        if frame_type == 'face':
            async with face_lock:
                frame = face_recognition_frame
        elif frame_type == 'pose':
            async with pose_lock:
                frame = pose_detection_frame
        elif frame_type == 'gaze':
            async with gaze_lock:
                frame = gaze_detection_frame
        else:
            await asyncio.sleep(0.01)
            continue

        if frame is not None:
            # Encode frame to JPEG
            loop = asyncio.get_event_loop()
            success, buffer = await loop.run_in_executor(
                cpu_executor,
                lambda: cv2.imencode('.jpg', frame)
            )

            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(0.01)


# Flask routes (now using Quart for async support)
@app.route('/')
async def index():
    return await app.send_static_file('index.html')


@app.route('/assets/<path:filename>')
async def serve_assets(filename):
    return await send_from_directory('Gp/assets', filename)


@app.route('/video_feed1')
async def video_feed1():
    return Response(generate_video_feed('face'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
async def video_feed2():
    return Response(generate_video_feed('pose'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed3')
async def video_feed3():
    return Response(generate_video_feed('gaze'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_user_data')
async def get_user_data():
    # Get user data asynchronously
    # Your code to fetch user data, similar to your original code
    # but using await for any I/O operations
    return jsonify({
        "detected": True,
        "name": "Example User",
        "status": "Active"
    })


# WebSocket for real-time updates (optional enhancement)
@app.websocket('/ws')
async def ws():
    while True:
        # Send concentration metrics via WebSocket
        data = {
            'concentration': current_concentration,
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send(json.dumps(data))
        await asyncio.sleep(1)  # Update every second


# Main function to start everything
async def main():
    # Initialize camera
    if not await initialize_camera():
        print("Failed to open camera. Exiting.")
        return

    # Start all processing coroutines
    asyncio.create_task(capture_frames())
    asyncio.create_task(process_face_recognition())
    asyncio.create_task(process_pose_detection())
    asyncio.create_task(process_gaze_detection())

    # Keep the main coroutine running
    while True:
        await asyncio.sleep(1)


if __name__ == '__main__':
    # Start the asyncio event loop with all our coroutines
    loop = asyncio.get_event_loop()

    # Start the main coroutine
    asyncio.run(main())

    # Run the Quart app
    app.run(host="0.0.0.0", port=500)