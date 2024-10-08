import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from termcolor import colored
from utils import which_color_is_it

color_palette = {
    "yellow": [
        [255, 255, 0],     # Yellow
        [255, 239, 102],   # Light Yellow
        [255, 223, 0],     # Gold Yellow
        [252, 232, 131],   # Pale Yellow
        [248, 222, 126],   # Flax Yellow
        [250, 250, 210],   # Lemon Chiffon
        [255, 255, 153],   # Light Golden Yellow
        [253, 253, 150],   # Pastel Yellow
        [255, 247, 0],     # Canary Yellow
        [255, 255, 102] ,   # Mellow Yellow
        [234, 173, 78]
    ],
    "blue": [
        [0, 0, 255],       # Blue
        [173, 216, 230],   # Light Blue
        [0, 191, 255],     # Deep Sky Blue
        [135, 206, 250],   # Light Sky Blue
        [70, 130, 180],    # Steel Blue
        [25, 25, 112],     # Midnight Blue
        [0, 0, 139],       # Dark Blue
        [30, 144, 255],    # Dodger Blue
        [100, 149, 237],   # Cornflower Blue
        [0, 0, 205],        # Medium Blue
        [61, 96, 158]
    ],
    "red": [
        [201, 74, 77],
        [255, 0, 0],       # Red
        [255, 99, 71],     # Tomato
        [255, 69, 0],      # Orange Red
        [255, 127, 80],    # Coral
        [220, 20, 60],     # Crimson
        [255, 160, 122],   # Light Salmon
        [255, 182, 193],   # Light Pink
        [250, 128, 114],   # Salmon
        [255, 36, 0],      # Scarlet
        [178, 34, 34]      # Firebrick
    ],
    "pink": [
        [230, 122, 126],
        [255, 182, 193],   # Light Pink
        [255, 105, 180],   # Hot Pink
        [255, 20, 147],    # Deep Pink
        [250, 218, 221],   # Pale Pink
        [255, 228, 225],   # Rose Pink
        [244, 194, 194],   # Baby Pink
        [255, 192, 203],   # Bubblegum Pink
        [255, 0, 255],     # Fuchsia
        [255, 145, 164],   # Salmon Pink
        [255, 174, 185]    # Blush Pink
    ],
    "orange": [
        [229, 117, 66],
        [255, 165, 0],     # Orange
        [255, 140, 0],     # Dark Orange
        [255, 159, 0],     # Amber
        [255, 99, 71],     # Tomato
        [255, 127, 80],    # Coral
        [255, 153, 51],    # Burnt Orange
        [255, 160, 122],   # Light Salmon
        [255, 222, 173],   # Navajo White
        [255, 204, 153],   # Peach Orange
        [255, 69, 0]       # Red Orange
    ],
    "green": [
        [71, 107, 93],
        [0, 255, 0],       # Green
        [144, 238, 144],   # Light Green
        [34, 139, 34],     # Forest Green
        [0, 128, 0],       # Dark Green
        [152, 251, 152],   # Pale Green
        [50, 205, 50],     # Lime Green
        [127, 255, 0],     # Chartreuse
        [124, 252, 0],     # Lawn Green
        [60, 179, 113],    # Medium Sea Green
        [0, 255, 127]      # Spring Green
    ],
    "brown": [
        [180, 117, 67],
        [165, 42, 42],     # Brown
        [160, 82, 45],     # Sienna
        [210, 105, 30],    # Chocolate
        [222, 184, 135],   # Burlywood
        [205, 133, 63],    # Peru
        [188, 143, 143]    # Rosy Brown
    ],
    "black": [
        [70, 57, 54],
        [0, 0, 0],         # Black
        [25, 25, 25],      # Dark Grayish Black
        [35, 35, 35],      # Jet Black
        [47, 79, 79],      # Dark Slate Gray
        [54, 69, 79],      # Charcoal Black
        [69, 69, 69],      # Granite Black
        [85, 85, 85],      # Smoky Black
        [105, 105, 105],   # Dim Black
        [112, 128, 144],   # Slate Gray Black
        [119, 136, 153]    # Light Slate Gray Black
    ],
    "white": [
        [221, 206, 153],
        [255, 255, 255],   # White
        [248, 248, 255],   # Ghost White
        [255, 250, 250],   # Snow White
        [240, 255, 255],   # Azure White
        [245, 245, 245],   # White Smoke
        [255, 255, 240],   # Ivory White
        [253, 245, 230],   # Antique White
        [255, 239, 213],   # Papaya Whip
        [255, 245, 238],   # Seashell White
        [250, 240, 230]    # Linen White
    ]
}

ball_colors = {
        'yellow': [234, 173, 78],
        'blue': [61, 96, 158],
        'red': [201, 74, 77],
        'pink': [230, 122, 126],
        'orange': [229, 117, 66],
        'green': [71, 107, 93],
        'brown': [180, 117, 67],
        'black': [70, 57, 54],
        'white': [221, 206, 153]
}


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img, size=(100, 100)):
    resized = cv2.resize(img, size)
    height, width, _ = resized.shape
    square_size = 55
    # Calculate the top-left corner of the square (center the square)
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    # Crop the square from the image
    cropped_image = resized[start_y:start_y + square_size, start_x:start_x + square_size]
    # Save or display the cropped image
    cv2.imwrite('cropped_billiard_ball.jpg', cropped_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped_image


def find_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    return colors


def plot_colors(colors):
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // len(colors)

    for i, color in enumerate(colors):
        rect[:, i * step:(i + 1) * step] = color.astype(int)

    plt.axis('off')
    plt.imshow(rect)
    # plt.show()


def map_color_to_ball(dominant_color):

    closest_ball = None
    min_distance = float('inf')

    for ball, color in ball_colors.items():
        distance = np.linalg.norm(np.array(color) - dominant_color)
        if distance < min_distance:
            min_distance = distance
            closest_ball = ball

    return closest_ball


def identify_ball_color(image_path: str):
    img = load_image(image_path)
    preprocessed_img = preprocess_image(img)

    dominant_colors = find_dominant_color(preprocessed_img, k=2)
    plot_colors(dominant_colors)
    array = []
    for dominant in dominant_colors:
        formatted = get_color_formatted_text(dominant.tolist())
        array.append(formatted)

    print(", ".join(array))
    # print(dominant_colors)

    dominant_color = dominant_colors[0]
    new_color_array = [
        str(round(dominant_color[0])),
        str(round(dominant_color[1])),
        str(round(dominant_color[2]))
    ]

    identified_ball = map_color_to_ball(dominant_color)
    return identified_ball, new_color_array


def get_color_formatted_text(color):
    new_array = map(str, map(int, color))
    return f"\033[48;2;{";".join(new_array)}m  \033[0m"


def test():
    crops_dir = "crops"
    crop_tests = [
        ["ultralytics_crop_1.jpg", "brown"],
        ["ultralytics_crop_2.jpg", "brown"],
        ["ultralytics_crop_3.jpg", "yellow"],
        ["ultralytics_crop_4.jpg", "red"],
        ["ultralytics_crop_5.jpg", "yellow"],
        ["ultralytics_crop_6.jpg", "blue"],
        ["ultralytics_crop_7.jpg", "blue"],
        ["ultralytics_crop_8.jpg", "white"],
        ["ultralytics_crop_9.jpg", "green"],
        ["ultralytics_crop_10.jpg", "orange"],
        ["ultralytics_crop_11.jpg", "pink"],
        ["ultralytics_crop_12.jpg", "green"],
        ["ultralytics_crop_13.jpg", "orange"],
        ["ultralytics_crop_14.jpg", "red"],
        ["ultralytics_crop_15.jpg", "black"],
    ]
    passed_tests_counter = 0
    i = 1
    for crop_test in crop_tests:
        expected = crop_test[1]
        expected_formatted = get_color_formatted_text(ball_colors[expected])
        crop_path = f"{crops_dir}/{crop_test[0]}"

        result, color = identify_ball_color(crop_path)
        formatted_color = get_color_formatted_text(color)
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))
        maybe = which_color_is_it(color_tuple)

        if result == expected:
            print(f"{i} ✅ {crop_test[0]} -> expected: {expected} {expected_formatted}  |  result: {result} {formatted_color}. Maybe: {maybe}")
            passed_tests_counter += 1
        else:
            print(f"{i} ❌ {crop_test[0]} -> expected: {expected} {expected_formatted}  |  result: {result} {formatted_color}. Maybe: {maybe}")
        i += 1
    print(f"Passed {passed_tests_counter} test from {len(crop_tests)}")


if __name__ == '__main__':
    test()
