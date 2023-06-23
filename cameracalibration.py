import numpy as np
import cv2

# Generate checkerboard pattern
pattern_size = (9, 6)
square_size = 0.025  # Size of each square in meters
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

# Initialize arrays to store object points and image points
object_points = []
image_points = []

# Load images and find checkerboard corners
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
for image_file in images:
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, refine them using cornerSubPix
    if ret:
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        )
        object_points.append(pattern_points)
        image_points.append(corners_refined)

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

# Calculate projection errors
mean_error = 0
for i in range(len(object_points)):
    image_points_proj, _ = cv2.projectPoints(
        object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    error = cv2.norm(image_points[i], image_points_proj, cv2.NORM_L2) / len(image_points_proj)
    mean_error += error

mean_error /= len(object_points)

# Print calibration results
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print("Mean Projection Error:", mean_error)

# Use a more robust checkerboard detection algorithm.
# The cv2.findChessboardCorners function is not very robust to noise and occlusion.
# You can use a more robust checkerboard detection algorithm, such as the cv2.cornerSubPix function,
# to improve the accuracy of the calibration results.

# For example, the cv2.cornerSubPix function can be used to refine the checkerboard corners,
# which results in a more accurate calibration.

# Use a more sophisticated camera calibration algorithm.
# The cv2.calibrateCamera function is a basic camera calibration algorithm.
# You can use a more sophisticated camera calibration algorithm, such as the Zhang Zhang algorithm,
# to improve the accuracy of the calibration results.

# For example, the Zhang Zhang algorithm takes into account the radial distortion of the camera lens,
# which can improve the accuracy of the calibration results.

# Use a more robust error calculation algorithm.
# The cv2.norm function is used to calculate the Euclidean distance between the projected image points
# and the actual image points.
# You can use a more robust error calculation algorithm, such as the Sampson error,
# to improve the accuracy of the calibration results.

# For example, the Sampson error takes into account the radial distortion of the camera lens,
# which can improve the accuracy of the calibration results.
