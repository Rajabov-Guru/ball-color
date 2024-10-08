from math import sqrt
from settings import settings


# Function to calculate Euclidean distance between two RGB colors
def color_distance(color1, color2):
    return sqrt(sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(color1, color2)))


COLOR_MAP = settings.COLOR_MAP


def is_color(rgb, color_name, threshold=60):
    if color_name not in COLOR_MAP:
        raise ValueError(f"Color '{color_name}' is not defined in the color map.")

    target_rgb = COLOR_MAP[color_name]
    distance = color_distance(rgb, target_rgb)
    return color_distance(rgb, target_rgb) < threshold, distance


def which_color_is_it(color):
    min_distance = 1000
    matched = None

    for color_name in COLOR_MAP:
        result, distance = is_color(color, color_name)
        if result and distance <= min_distance:
            matched = color_name
            min_distance = distance
    return matched
