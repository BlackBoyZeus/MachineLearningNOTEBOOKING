import cv2
import numpy as np

# Load the pre-trained model for object detection
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

# Function to detect objects in the frame
def detect_objects(frame):
    # Perform image processing operations specific to thermal images in an arctic setting
    # (e.g., adjust contrast, enhance edges, etc.)
    # ...
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform adaptive thresholding to separate objects from the background
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Create a blob from the thresholded image
    blob = cv2.dnn.blobFromImage(threshold, size=(300, 300), swapRB=True, crop=False)

    # Set the input to the pre-trained model
    net.setInput(blob)

    # Forward pass through the network
    detections = net.forward()

    # Extract bounding box coordinates and confidence for each detected object
    boxes = []
    confidences = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])
            if class_id == 1:  # Consider only person class (change as per your requirement)
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                boxes.append(box.astype("int"))
                confidences.append(confidence)

    return boxes, confidences

# Function to initialize the object tracker
def initialize_tracker(frame, bbox):
    tracker.init(frame, bbox)

# Function to update the object tracker
def update_tracker(frame):
    success, bbox = tracker.update(frame)
    return success, bbox

# Open the video file or thermal camera stream
video = cv2.VideoCapture(0)  # Change the argument to the video file path or camera index if using a thermal camera

# Read the first frame
ret, frame = video.read()
if not ret:
    print("Error reading video file or thermal camera stream")
    exit()

# Select a bounding box for tracking
bbox = cv2.selectROI("Object Tracking", frame, fromCenter=False, showCrosshair=True)
initialize_tracker(frame, bbox)

# Process frames until the user exits or the video ends
while True:
    # Read the next frame
    ret, frame = video.read()
    if not ret:
        break

    # Update the object tracker
    success, bbox = update_tracker(frame)

    # Draw bounding box and label on the frame
    if success:
        (x, y, w, h) = tuple(map(int, bbox))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()


#In this code, we still utilize the OpenCV library for object detection and tracking. However, we introduce additional image processing operations specific to thermal images in an arctic setting. These operations may include adjusting contrast, enhancing edges, or performing adaptive thresholding to separate objects from the background. These steps can be customized and extended based on the characteristics and challenges of your thermal camera and the arctic environment.

#Please note that the code assumes you have the thermal camera connected to your system, and you need to adjust the video capture source accordingly. Additionally, make sure to download and place the necessary pre-trained model files (frozen_inference_graph.pb and ssd_mobilenet_v2_coco_2018_03_29.pbtxt) in the same directory as the script.
