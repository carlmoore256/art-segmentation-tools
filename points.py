import math

def is_collinear(p1, p2, p3):
    # Calculate the cross product
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def line_to_point(line):
    return (line.start.real, line.start.imag), (line.end.real, line.end.imag)

def calculate_angle(p1, p2, p3):
    """Calculate the angle formed by three points (in degrees)."""
    a = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    b = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
    c = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
    if a * b == 0:  # Prevent division by zero
        return 0
    x = (a**2 + b**2 - c**2) / (2 * a * b)
    if x < -1:
        x = -1
    elif x > 1:
        x = 1
    angle = math.acos(x)
    return math.degrees(angle)


