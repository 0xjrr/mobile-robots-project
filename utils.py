import numpy as np
import math
import cv2

def detect_colored_dots_in_rgb(image, color_lower, color_upper):
    """
    Detects dots of a specific color in an image using RGB color space.

    :param image: The image to search in.
    :param color_lower: The lower bound of the color range in RGB.
    :param color_upper: The upper bound of the color range in RGB.
    :return: List of coordinates of detected dots.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(image, color_lower, color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            coordinates.append((cx, cy))

    return coordinates

def find_coordinates(matrix, value):
    """Find the coordinates of a given value in the matrix."""
    coordinates = np.argwhere(matrix == value)
    if coordinates.size > 0:
        return coordinates[0]  # Assuming only one occurrence
    return None

def calculate_robot_position_and_orientation(matrix, front_val, back_val):
    """
    Calculate the position (x, y) and orientation (theta) of a robot.
    :param matrix: 2D numpy array representing the environment.
    :param front_val: Integer representing the front of the robot in the matrix.
    :param back_val: Integer representing the back of the robot in the matrix.
    :return: Tuple (x, y, theta) where theta is in degrees.
    """
    # Find coordinates of the front and back
    front_coords = find_coordinates(matrix, front_val)
    back_coords = find_coordinates(matrix, back_val)

    if front_coords is not None and back_coords is not None:
        # Calculate (x, y)
        x = back_coords[0]
        y = back_coords[1]
        # Calculate orientation theta
        theta = np.arctan2(front_coords[1] - back_coords[1], front_coords[0] - back_coords[0])
        theta_degrees = np.degrees(theta)

        return x, y, theta_degrees
    else:
        return None, None, None
