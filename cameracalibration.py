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

    # If corners are found, add object points and image points
    if ret:
        object_points.append(pattern_points)
        image_points.append(corners)

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
#n this code, we first generate a checkerboard pattern using the specified pattern size and square size. We then initialize arrays to store the object points and image points.

#We load a series of images containing the checkerboard pattern and find the corners of the checkerboard using the cv2.findChessboardCorners function. If the corners are found, we add the corresponding object points and image points to their respective arrays.

#Next, we calibrate the camera using the cv2.calibrateCamera function, which returns the camera matrix, distortion coefficients, rotation vectors, and translation vectors.

#Finally, we calculate the projection errors for each image by projecting the object points onto the image plane using the cv2.projectPoints function. We compute the mean error by calculating the Euclidean distance between the projected image points and the actual image points.

#The camera matrix, distortion coefficients, and mean projection error are then printed as the calibration results.

#Please note that you will need to provide actual image files for the images list, and adjust the pattern_size and square_size variables according to your specific checkerboard pattern.
