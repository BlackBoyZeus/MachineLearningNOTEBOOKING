import cv2
import numpy as np

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

def measure_object_speed(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize object position in the first frame
    object_pos = None

    # Initialize a list to store object speeds
    speeds = []

    while True:
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if object_pos is not None:
            # Calculate optical flow using Lucas-Kanade method
            new_object_pos, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, object_pos, None, **lk_params)

            # Select only the points with good optical flow
            good_new = new_object_pos[status == 1]
            good_old = object_pos[status == 1]

            # Calculate the object speed as the mean displacement of points
            speed = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            speeds.append(speed)

        # Update object position for the next frame
        object_pos = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)

        if object_pos is not None:
            object_pos = np.float32(object_pos).reshape(-1, 1, 2)

        # Display the frame with object speed information
        cv2.putText(
            frame,
            f"Object Speed: {round(speed, 2)} pixels/frame",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Object Speed Measurement", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Update the previous frame and its grayscale version
        prev_frame = frame
        prev_gray = gray

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate average object speed
    avg_speed = np.mean(speeds)
    print("Average Object Speed:", round(avg_speed, 2), "pixels/frame")

# Example usage
video_path = "video.mp4"
measure_object_speed(video_path)
#n this code, we utilize the Lucas-Kanade method for optical flow estimation. The measure_object_speed function takes the path to a video file as input and calculates the average object speed in pixels per frame.

#We read consecutive frames from the video file, convert them to grayscale, and use the Lucas-Kanade algorithm (cv2.calcOpticalFlowPyrLK) to track the movement of predefined points
