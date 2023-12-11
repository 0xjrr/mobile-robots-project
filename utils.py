import numpy as np
import math
import cv2
from typing import Tuple, Union

def detect_colored_dots_in_rgb(image, color_lower, color_upper):
    """
    Detects dots of a specific color in an image using RGB color space.

    :param image: The image to search in.
    :param color_lower: The lower bound of the color range in RGB.
    :param color_upper: The upper bound of the color range in RGB.
    :return: List of coordinates of detected dots.
    """
    mask = cv2.inRange(image, color_lower, color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            coordinates.append((cy, cx))

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

def ball_shooting_point(ball_coords: tuple, goal_coords: tuple, extension_value: int, isRobotShooting: bool) -> Tuple[Union[int, None], Union[int, None]]:
    if ball_coords == () or goal_coords == ():
        return None, None
    m = (goal_coords[1] - ball_coords[1]) / (goal_coords[0] - ball_coords[0])
    b = ball_coords[1] - ( m * ball_coords[0] )  
    x =  ball_coords[0] - extension_value if not isRobotShooting else ball_coords[0] + extension_value
    y = int((m * x) + b)
    return x, y

def safe_get(lst, idx, default=None):
    return lst[idx] if 0 <= idx < len(lst) else default

def get_angle_of_rotation(robot_coords, shooting_point):
    delta_x = shooting_point[0] - robot_coords[0]
    delta_y = shooting_point[1] - robot_coords[1]

    # Calculate the angle to the target in radians, then convert to degrees
    desired_angle = math.degrees(math.atan2(delta_y, delta_x))

    # Calculate the difference between the desired angle and the robot's current angle
    angle_difference = desired_angle - robot_coords[2]

    # Normalize the angle to be within the range [-180, 180]
    angle_difference = (angle_difference + 180) % 360 - 180

    return angle_difference

def move_to_shooting_point(angle_of_rotation, distance_to_point, rotation_threshold, distance_threshold):
    if(not (angle_of_rotation < rotation_threshold) and not(angle_of_rotation >= 0)):
        if(angle_of_rotation < 0):
            send_event("rodar_direita")
        else:
            send_event("rodar_esquerda")
    else:
        if(not (distance_to_point < distance_threshold) and not(distance_to_point >= 0)):
            send_event("andar300")
        else:
            send_event("parar")

def get_distance_between_two_points(point1, point2):
    distance_x = abs(point2[0] - point1[0])
    distance_y = abs(point2[1] - point1[1])
    distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

    return distance