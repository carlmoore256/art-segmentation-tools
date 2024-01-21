from path_operations import is_point_in_path, random_point_within_path
import random
from scipy.spatial import Voronoi
import numpy as np
import cv2
from image_processing import get_image_gradients
import math


def poisson_disk_sampling_within_path(path, min_dist=0.1):
    cell_size = min_dist / np.sqrt(2)
    # Grid dimensions
    grid_width = int(np.ceil(1 / cell_size))
    grid_height = int(np.ceil(1 / cell_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

    def grid_coords(point):
        return int(point[0] / cell_size), int(point[1] / cell_size)

    def point_valid(point):
        if not is_point_in_path(point, path):
            return False
        gx, gy = grid_coords(point)
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                grid_point = grid[x][y]
                if grid_point and np.linalg.norm(np.array(point) - np.array(grid_point)) < min_dist:
                    return False
        return True

    def random_point_around(point):
        r = random.uniform(min_dist, 2 * min_dist)
        theta = random.uniform(0, 2 * np.pi)
        return point[0] + r * np.cos(theta), point[1] + r * np.sin(theta)

    active_points = [random_point_within_path(path)]

    while active_points:
        idx = random.randint(0, len(active_points) - 1)
        point = active_points[idx]
        new_point_found = False
        for _ in range(30):  # k attempts
            new_point = random_point_around(point)
            if point_valid(new_point):
                active_points.append(new_point)
                grid[grid_coords(new_point)[0]][grid_coords(
                    new_point)[1]] = new_point
                new_point_found = True
                break
        if not new_point_found:
            active_points.pop(idx)

    # Extract points from the grid
    points = []
    for x in range(grid_width):
        for y in range(grid_height):
            if grid[x][y]:
                points.append(grid[x][y])
    return points


def get_average_density(x, y, density_map, neighborhood_size=3):
    # Calculate the bounds of the neighborhood
    half_size = neighborhood_size // 2
    x_min, x_max = max(x - half_size, 0), min(x +
                                              half_size + 1, density_map.shape[1])
    y_min, y_max = max(y - half_size, 0), min(y +
                                              half_size + 1, density_map.shape[0])

    # Extract the neighborhood and calculate the average density
    neighborhood = density_map[y_min:y_max, x_min:x_max]
    return np.mean(neighborhood)


def thin_points_based_on_density(points, density_map, power=0.5, neighborhood_size=3):
    thinned_points = []
    minx, maxx = min(p[0] for p in points), max(p[0] for p in points)
    miny, maxy = min(p[1] for p in points), max(p[1] for p in points)

    for point in points:
        # Normalize and scale point coordinates to match density map dimensions
        x = int((point[0] - minx) / (maxx - minx) * density_map.shape[1])
        y = int((point[1] - miny) / (maxy - miny) * density_map.shape[0])
        y = density_map.shape[0] - 1 - y
        x = min(max(x, 0), density_map.shape[1] - 1)
        y = min(max(y, 0), density_map.shape[0] - 1)

        # Get the average density value in the neighborhood
        avg_density_value = get_average_density(
            x, y, density_map, neighborhood_size)

        # Use the average density value to decide whether to keep the point
        if math.pow(random.random(), power) < avg_density_value:
            thinned_points.append(point)

    return thinned_points


def compute_centroids(points, path):
    vor = Voronoi(points)
    centroids = []
    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            # Ensure the polygon is within the path
            polygon = [p for p in polygon if is_point_in_path(p, path)]
            if polygon:
                x, y = zip(*polygon)
                centroid = (sum(x) / len(polygon), sum(y) / len(polygon))
                if is_point_in_path(centroid, path):
                    centroids.append(centroid)
    return centroids


def centroidal_voronoi_tessellation_within_path(path, num_points=100, iterations=10):
    # Initialize points randomly within the path
    points = [random_point_within_path(path) for _ in range(num_points)]
    for _ in range(iterations):
        centroids = compute_centroids(points, path)
        if len(centroids) < len(points):
            # Add new random points if some centroids are discarded
            centroids += [random_point_within_path(path)
                          for _ in range(len(points) - len(centroids))]
        points = centroids
    return points


def quad_points(shape_bbox, grid_size):
    xmin, xmax, ymin, ymax = shape_bbox
    xgrid = np.arange(xmin, xmax, grid_size)
    ygrid = np.arange(ymin, ymax, grid_size)
    grid_points = np.transpose(
        [np.tile(xgrid, len(ygrid)), np.repeat(ygrid, len(xgrid))])
    return grid_points


def quad_points_within_path(path, grid_size):
    bbox = [min(path, key=lambda x: x[0])[0], max(path, key=lambda x: x[0])[0],
            min(path, key=lambda x: x[1])[1], max(path, key=lambda x: x[1])[1]]
    grid_points = quad_points(bbox, grid_size)
    point_set = [point for point in grid_points if is_point_in_path(
        (point[0], point[1]), path)]
    point_set.extend(path)  # Adding boundary points
    return np.array(point_set)


# produce a map which informs tesselation process
# on the desired density of the mesh within a given region
def generate_mesh_density_map(mask, depth, bbox):
    # crop the depth to the bbox
    depth = depth.copy()[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])]
    masked_depth = mask.apply(depth)
    med_depth = np.median(masked_depth[masked_depth > 0])
    mask_gradients = get_image_gradients(mask.mask_data)
    mask_boundary = mask_gradients['magnitude'] > 0
    mask_boundary = cv2.dilate(mask_boundary.astype(
        np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=3)
    mask_boundary = cv2.GaussianBlur(mask_boundary, (15, 15), 0)
    masked_depth[mask_boundary > 0] = 0
    masked_depth -= med_depth
    masked_depth_norm = masked_depth / np.max(masked_depth)
    depth_gradients = get_image_gradients(masked_depth_norm)
    output_map = depth_gradients['magnitude']
    output_map[output_map > 1] = 0
    output_map = cv2.GaussianBlur(output_map, (15, 15), 0)
    output_map = output_map / np.max(output_map)
    return output_map

def generate_sobel_density_map(cropped_img):
    gradients = get_image_gradients(cropped_img)
    mag = gradients['magnitude']
    mag = mag / np.max(mag)
    mag = cv2.GaussianBlur(mag, (15, 15), 0)
    return mag