from points import is_collinear
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatchesea
from scipy.spatial import Delaunay
import numpy as np
import triangle as tr
import triangle.plot as tplot
from path_operations import is_point_in_path, random_point_within_path
import random
from scipy.spatial import Voronoi, voronoi_plot_2d

def is_ear(p1, p2, p3, polygon):
    if is_collinear(p1, p2, p3):
        return False

    triangle = [p1, p2, p3]
    for p in polygon:
        if p not in triangle and point_in_triangle(p, triangle):
            return False

    return True


def point_in_triangle(pt, triangle):
    # Barycentric coordinates method
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(pt, triangle[0], triangle[1]) < 0.0
    b2 = sign(pt, triangle[1], triangle[2]) < 0.0
    b3 = sign(pt, triangle[2], triangle[0]) < 0.0

    return ((b1 == b2) and (b2 == b3))


def ear_clip_triangulation(polygon):
    triangles = []
    remaining_polygon = polygon.copy()

    while len(remaining_polygon) > 3:
        for i in range(len(remaining_polygon)):
            p1 = remaining_polygon[i]
            p2 = remaining_polygon[(i + 1) % len(remaining_polygon)]
            p3 = remaining_polygon[(i + 2) % len(remaining_polygon)]

            if is_ear(p1, p2, p3, remaining_polygon):
                triangles.append([p1, p2, p3])
                del remaining_polygon[(i + 1) % len(remaining_polygon)]
                break

    triangles.append(
        [remaining_polygon[0], remaining_polygon[1], remaining_polygon[2]])
    return triangles


def delaunay_triangulation(points):
    points = np.array(points)
    delaunay_triangles = Delaunay(points).simplices
    triangles = points[delaunay_triangles]
    return triangles


def filter_triangles(triangles, path, filter_point_fn):
    filtered_triangles = []
    for triangle in triangles:
        if all(filter_point_fn((vertex[0], vertex[1]), path) for vertex in triangle):
            filtered_triangles.append(triangle)
    return filtered_triangles


def constrained_delaunay_triangulation(path, round_digits=6):
    # round the path to avoid floating point errors
    vertices = [(round(x, round_digits), round(y, round_digits))
                for x, y in path]
    segments = np.array(
        [[i, i+1] for i in range(len(vertices)-1)] + [[len(vertices)-1, 0]])
    triangulated = tr.triangulate({
        "vertices": vertices,
        "segments": segments,
    }, 'p')
    return triangulated


# absolute degenerate shit because I don't want to debug a C libary for random segfaults
def run_constrained_delaunay_safe(path, internal_points=None):
    import subprocess
    import json

    polygon_verts = path.copy()
    vertices = polygon_verts
    if internal_points is not None:
        vertices = np.vstack([polygon_verts, internal_points]).tolist()
    else:
        vertices = polygon_verts

    segments = ([[i, i+1] for i in range(len(polygon_verts)-1)] +
                [[len(polygon_verts)-1, 0]])


    triangulation_code = """
import sys
import json
import triangle
import numpy as np

def triangulate(data):
    vertices = np.array(data['vertices'])
    segments = np.array(data['segments'])
    return triangle.triangulate({'vertices': vertices, 'segments': segments}, 'p')

if __name__ == "__main__":
    data = json.load(sys.stdin)
    result = triangulate(data)
    for key in result.keys():
        result[key] = result[key].tolist()
    print(json.dumps(result))
"""


    data = {
        "vertices": vertices,
        "segments": segments
    }

    max_retries = 100
    retry = 0

    while True:
        try:
            proc = subprocess.Popen(['python', '-c', triangulation_code],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            output, errors = proc.communicate(json.dumps(data))

            if proc.returncode == 0:      
                return json.loads(output)
            else:
                retry += 1
                if retry > max_retries:
                    raise Exception("Max retries reached")
        except Exception as e:
            print(f"An error occurred, retrying")
            # Optional: Add a delay or a limit to retries


# plots results made from the package triangle
def plot_triangles_triangulation(triangulated):
    plt.figure(figsize=(8, 8))
    tplot(plt.gca(), **triangulated)
    plt.show()


def plot_triangulation(path, triangles):
    plt.figure()
    for triangle in triangles:
        # Adding the first point at the end to close the triangle when plotting
        polygon = np.vstack([triangle, triangle[0]])
        plt.plot(*zip(*polygon), color='black')

    # Plot the original path
    x, y = zip(*path)
    plt.title("Path triangulation")
    plt.plot(x, y, marker='o', color='red', ls='')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
