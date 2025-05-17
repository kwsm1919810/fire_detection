import serial
import time
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov5s.pt")

# Initialize serial communication
# Adjust port and baud rate as needed
ser = serial.Serial('COM3', 9600, timeout=1)
# Give time for the serial port to initialize
time.sleep(2)

# Define objects of interest
target_objects = {"person", "fire"}

# Start video capture (webcam or video file)
# 0 for webcam, or path to video file
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Process detections
    detected = set()
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name in target_objects:
                detected.add(class_name)

    # Send detected objects via serial
    if detected:
        message = f"Detected: {', '.join(detected)}"
        print(message)
        # Send message
        ser.write((message + "\n").encode('utf-8'))
        # Prevent flooding the serial device
        time.sleep(5)

    # Optional: Display the frame with annotations
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()