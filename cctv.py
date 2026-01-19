import cv2
import datetime
import matplotlib.pyplot as plt

# Open camera (0 = default webcam, or replace with IP Camera URL)
camera_source = 0  # Change to "rtsp://username:password@ip:port" for CCTV
cap = cv2.VideoCapture(0)

# Video recording setup
recording = False
out = None

# Motion detection setup
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display Date & Time on video
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    frame_diff = cv2.absdiff(prev_frame_gray, gray)

    thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    motion_detected = cv2.countNonZero(thresh) > 1000  # Adjust threshold

    if motion_detected:
        cv2.putText(frame, "Motion Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    prev_frame_gray = gray

    # Start/Stop Recording
    if recording:
        out.write(frame)
        cv2.putText(frame, "Recording...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show video feed
    # cv2.imshow("CCTV Feed", frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('r'):  # Start/Stop recording
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('cctv_record.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
        else:
            recording = False
            out.release()

# Cleanup
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
