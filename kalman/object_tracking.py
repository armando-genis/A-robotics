import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load the YOLOv8 model pre-trained on COCO dataset
model = YOLO('yolov8n.pt')
# Initialize SORT tracker
mot_tracker = Sort()
threshold = 0.65 # Confidence threshold for YOLOv8 detections
# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 detection on the frame
        results = model(frame, conf=threshold,)
        # Prepare detections for SORT tracker
        dets = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                if int(c) == 0:  # Check if the detected class is 'person' (class ID 0)
                    if b.size >= 5:  
                        conf = b[4]  
                        dets.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), conf])
                    else:
                        # If confidence is not available, append without it
                        dets.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), 1.0]) 

        # Convert detections to a numpy array for SORT
        if dets:
            dets = np.array(dets)
            trackers = mot_tracker.update(dets)
            # Draw tracked objects
            for d in trackers:
                x1, y1, x2, y2, track_id = map(int, d[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            trackers = mot_tracker.update(np.empty((0, 5)))
        # Display the frame with detections and tracking
        cv2.imshow('Frame', frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached or no frame is read
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
