import numpy as np
import cv2

def convert_image_points_to_homogeneous_coordinates(image_points):
    """
    Convert image points to homogeneous coordinates.

    Args:
        image_points: The image points in 2D.

    Returns:
        The image points in homogeneous coordinates.
    """

    return np.concatenate((image_points, np.ones((len(image_points), 1))), axis=1)

def compute_object_coordinates_in_real_world_coordinates(image_points, camera_matrix, rotation_vector, translation_vector):
    """
    Compute the object coordinates in real-world coordinates.

    Args:
        image_points: The image points in homogeneous coordinates.
        camera_matrix: The camera matrix.
        rotation_vector: The rotation vector.
        translation_vector: The translation vector.

    Returns:
        The object coordinates in real-world coordinates.
    """

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Invert rotation matrix and translation vector
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    inv_translation_vector = np.dot(-inv_rotation_matrix, translation_vector)

    # Calculate camera matrix inverse
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Compute object points in camera coordinates
    object_points_camera_coords = np.dot(camera_matrix_inv, image_points.T).T

    # Compute object points in world coordinates
    object_points_world_coords = np.dot(inv_rotation_matrix, object_points_camera_coords.T).T + inv_translation_vector

    return object_points_world_coords

def main():
    # Load camera calibration parameters
    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    distortion_coeffs = np.array([-0.2, 0.1, 0, 0, 0])

    # Load image and detect object
    image = cv2.imread("image.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    object_points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)

    # Find object pose using solvePnP
    _, rotation_vector, translation_vector, _ = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coeffs)

    # Define image points of the object
    image_points = np.array([[200, 150], [250, 150], [250, 200], [200, 200]], dtype=np.float32)

    # Convert image points to homogeneous coordinates
    image_points_homogeneous = convert_image_points_to_homogeneous_coordinates(image_points)

    # Compute object coordinates in real-world coordinates
    object_coordinates_world_coords = compute_object_coordinates_in_real_world_coordinates(image_points_homogeneous, camera_matrix, rotation_vector, translation_vector)

    # Print the object coordinates
    for i, point in enumerate(object_coordinates_world_coords):
        print(f"Point {i+1}: x={point[0]:.2f}m, y={point[1]:.2f}m, z={point[2]:.2f}m")

if __name__ == "__main__":
    main()

    main()
#In this code, we define a function find_object_coordinates that takes image points, camera matrix, rotation vector, and translation vector as input. It calculates the x, y, z coordinates of the object in real-world coordinates using geometric transformations.

#The main function demonstrates the usage of the find_object_coordinates function. It loads the camera calibration parameters (camera matrix and distortion coefficients), loads an image, and detects the image points of the object.

#Using the solvePnP function, it estimates the rotation vector and translation vector of the object. Then, it passes the image points, camera matrix, rotation vector, and translation vector to the find_object_coordinates function to compute the object coordinates in real-world coordinates.

#Finally, the script prints the x, y, z coordinates of the object.
