import cv2
import json
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response
from playsound import playsound
import threading

# Initialize YOLO model
model = YOLO('vehicle_dataset.pt')

# Initialize Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(1)

# Define two symmetrical and centered regions (x1, y1, x2, y2)
region1 = (10, 15, 320, 445)   # Left region
region2 = (320, 15, 630, 445)  # Right region

# Track last alert time (to avoid spam)
last_alert_time = 0

def play_audio():
    """Plays the alert sound."""
    playsound("roblox.mp3")  # Ensure the file exists in the same directory

def check_overlap(box, region):
    """Check if any part of the bounding box overlaps with the defined region."""
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = region
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

def generate_frames():
    """Captures video frames, runs YOLO detection, and streams the results."""
    global last_alert_time  # Access global variable

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame)

        vehicle_in_region1 = False
        vehicle_in_region2 = False

        # Process detection results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                class_id = int(box.cls[0])  # Class ID
                confidence = box.conf[0]  # Confidence score
                class_name = model.names[class_id]  # Get class name

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display class name and confidence
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if the vehicle is inside either region
                if check_overlap((x1, y1, x2, y2), region1):
                    vehicle_in_region1 = True
                if check_overlap((x1, y1, x2, y2), region2):
                    vehicle_in_region2 = True

        # Draw predefined regions
        cv2.rectangle(frame, (region1[0], region1[1]), (region1[2], region1[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (region2[0], region2[1]), (region2[2], region2[3]), (255, 0, 0), 2)

        # Determine double parking status
        double_parked = vehicle_in_region1 and vehicle_in_region2
        json_data = {"dpark": double_parked}

        # Write status to JSON
        with open("data.json", "w") as file:
            json.dump(json_data, file)

        # Play alert if double parking is detected
        now = time.time()
        if double_parked and (now - last_alert_time >= 5):  # 5-second cooldown
            threading.Thread(target=play_audio).start()  # Play sound in a separate thread
            last_alert_time = now  # Update alert time

        # Add alerts on the frame
        if double_parked:
            cv2.putText(frame, "Double Parking Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif vehicle_in_region1 or vehicle_in_region2:
            cv2.putText(frame, "Vehicle Detected in Region!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode and yield frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Renders the web page."""
    return render_template('index.html')

@app.route('/video')
def video():
    """Streams the processed video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
