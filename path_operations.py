
import math
import matplotlib.pyplot as plt
from points import is_collinear, line_to_point, calculate_angle
import numpy as np
import random

def line_norm_coords(line, bbox):
    xmin, xmax, ymin, ymax = bbox
    x_range = xmax - xmin
    y_range = ymax - ymin
    start_coords = ((line.start.real - xmin) / x_range, (line.start.imag - ymin) / y_range)
    end_coords = ((line.end.real - xmin) / x_range, (line.end.imag - ymin) / y_range)
    return start_coords, end_coords

def reverse_normalize_coords(norm_coords, bbox):
    xmin, xmax, ymin, ymax = bbox
    x_range = xmax - xmin
    y_range = ymax - ymin
    orig_start_coords = (norm_coords[0][0] * x_range + xmin, norm_coords[0][1] * y_range + ymin)
    orig_end_coords = (norm_coords[1][0] * x_range + xmin, norm_coords[1][1] * y_range + ymin)
    return orig_start_coords, orig_end_coords

def get_paths_extent(paths):
    extent = [0, 0, 0, 0]
    for p in paths:
        bbox = p.bbox()
        extent[0] = min(extent[0], bbox[0])
        extent[1] = max(extent[1], bbox[1])
        extent[2] = min(extent[2], bbox[2])
        extent[3] = max(extent[3], bbox[3])
    return extent

def segments_to_path(segments):
    """
    Converts an array of line segments into an array of points.
    Each segment is a tuple of two points (start and end), and this function
    will return a list of points in the order they appear in the segments.
    """
    if not segments:
        return []
    path = [segments[0][0]]
    for segment in segments:
        path.append(segment[1])
    return path


def remove_collinear_points(path):
    if len(path) < 3:
        return path
    optimized_path = [path[0]]
    for i in range(1, len(path) - 1):
        if not is_collinear(path[i - 1], path[i], path[i + 1]):
            optimized_path.append(path[i])
    optimized_path.append(path[-1])
    return optimized_path

# custom algorithm based on angle and distance
def reduce_path_complexity(path, distance_threshold, angle_threshold):
    if len(path) < 3:
        return path  # Not enough points to simplify

    reduced_path = [path[0]]
    for i in range(1, len(path) - 1):
        prev_point = reduced_path[-1]
        current_point = path[i]
        next_point = path[i + 1]

        dist = math.sqrt((current_point[0] - prev_point[0])**2 + (current_point[1] - prev_point[1])**2)
        angle = calculate_angle(prev_point, current_point, next_point)

        if dist > distance_threshold or angle < angle_threshold:
            reduced_path.append(current_point)

    reduced_path.append(path[-1])  # Always include the last point
    return reduced_path

def optimize_path(path, dist_thresh = 0.01, angle_thresh = 0.01):
    path = remove_collinear_points(path)
    path = reduce_path_complexity(path, dist_thresh, angle_thresh)
    return path

def simplify_ramer_douglas_peucker(points, epsilon):
    """
    Reduce the number of points in a curve with the Ramer-Douglas-Peucker algorithm.

    :param points: List of points as tuples [(x1, y1), (x2, y2), ...]
    :param epsilon: Distance threshold for point reduction. Points farther than
                    epsilon from the line segment will be kept.
    :return: Reduced list of points.
    """
    # Find the point with the maximum distance from the line formed by the start and end points
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    # If the max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = simplify_ramer_douglas_peucker(points[:index+1], epsilon)
        rec_results2 = simplify_ramer_douglas_peucker(points[index:], epsilon)

        # Build the result list
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[-1]]

    return result

def perpendicular_distance(pt, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line.

    :param pt: The point (x, y)
    :param line_start: Start point of the line (x, y)
    :param line_end: End point of the line (x, y)
    :return: Perpendicular distance from pt to the line.
    """
    if line_start == line_end:
        return math.sqrt((pt[0] - line_start[0]) ** 2 + (pt[1] - line_start[1]) ** 2)

    else:
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        num = abs(dy * pt[0] - dx * pt[1] + line_end[0] * line_start[1] - line_end[1] * line_start[0])
        den = math.sqrt(dy ** 2 + dx ** 2)
        return num / den

def plot_simplification(original, simplified):
    percent = ((len(simplified) / len(original)) * 100)
    plt.figure(figsize=(5,5))
    plt.title(f"Path Simplification ({(100-percent):.2f}% reduction)")
    plt.scatter([p[0] for p in original], [p[1] for p in original], label="Original", c="r")
    plt.scatter([p[0] for p in simplified], [p[1] for p in simplified], label="Simplified", c="b")
    plt.legend()
    plt.show()


def is_point_in_path(point, path, tolerance=0.01):
    x, y = point
    num = len(path)
    j = num - 1
    inside = False
    for i in range(num):
        xi, yi = path[i]
        xj, yj = path[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def random_point_within_path(path, max_iters=10000):
    iters = 0
    while True:
        rand_point = (random.random(), random.random())
        if is_point_in_path(rand_point, path):
            return rand_point
        iters += 1
        if iters > max_iters:
            raise Exception("Random point max iterations reached")
        
def bbox_range(bbox):
    return (bbox[1] - bbox[0], bbox[3] - bbox[2])
        
def path_eccentricity(path):
    bbox = path.bbox()
    xrange, yrange = bbox_range(bbox)
    return yrange / xrange