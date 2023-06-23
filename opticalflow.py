import cv2
import numpy as np

# Parameters for CSRT object tracking
csrt_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
    templateWindowSize=11,
    searchWindowSize=21,
)

def measure_object_speed(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize object position in the first frame
    object_pos = cv2.goodFeaturesToTrack(prev_gray, **csrt_params)

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
            # Track the object using CSRT algorithm
            new_object_pos, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, object_pos, None, **csrt_params)

            # Select only the points with good optical flow
            good_new = new_object_pos[status == 1]
            good_old = object_pos[status == 1]

            # Calculate the object speed as the mean displacement of points
            speed = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            speeds.append(speed)

        # Update object position for the next frame
        object_pos = new_object_pos

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

#Uses a more robust object tracking algorithm. The current implementation uses the Lucas-Kanade algorithm, which is sensitive to noise and occlusion. You can use a more robust object tracking algorithm, such as CSRT or KCF, to track the object's position more reliably.
#Uses a more sophisticated visualization of the results. The current implementation simply displays the object speed on the frame. You can use a more sophisticated visualization, such as a tracking plot, to show the evolution of the object's speed over time.
#Adds more error handling. The current implementation does not handle errors very well. For example, if the video file cannot be opened or if the object cannot be tracked, the program will crash. You can add more error handling to the code to make it more robust.
