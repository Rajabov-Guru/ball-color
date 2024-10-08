from math import sqrt


# Function to calculate Euclidean distance between two RGB colors
def color_distance(color1, color2):
    return sqrt(sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(color1, color2)))


# Dictionary to store the RGB values for various colors
COLOR_MAP = {
    'red': (201, 74, 77),
    'pink': (230, 122, 126),
    'yellow': (234, 173, 78),
    'blue': (61, 96, 158),
    'orange': (229, 117, 66),
    'green': (71, 107, 93),
    'brown': (180, 117, 67),
    'black': (70, 57, 54),
    'white': (221,206,153)
}


# Universal function to check if a given color matches a target color
def is_color(rgb, color_name, threshold=50):
    if color_name not in COLOR_MAP:
        raise ValueError(f"Color '{color_name}' is not defined in the color map.")

    target_rgb = COLOR_MAP[color_name]
    return color_distance(rgb, target_rgb) < threshold


def which_color_is_it(color):
    matched_colors = []
    # Loop through the color map to check which color matches
    for color_name in COLOR_MAP:
        if is_color(color, color_name):
            matched_colors.append(color_name)
    return matched_colors