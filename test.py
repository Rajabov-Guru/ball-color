import os
from pathlib import Path

from identify import identify_ball_color
from settings import settings
from utils import which_color_is_it


def get_color_formatted_text(color, text = "  "):
    new_array = map(str, map(int, color))
    return f"\033[48;2;{";".join(new_array)}m{text}\033[0m"


def test_crop(crop_path, expected, number = 1):
    expected_formatted = get_color_formatted_text(settings.COLOR_MAP[expected])
    result, color, dominant_colors, filtered_dominants = identify_ball_color(str(crop_path))

    formatted_color = get_color_formatted_text(color)
    maybe, _ = which_color_is_it(color)

    sign = "❌"
    file_link = f"| {str(crop_path)}"
    if result == expected:
        file_link = ""
        sign = "✅"
        return True
    else:
        if settings.output_dominant_colors:
            dominants_lists = [dominant_colors, filtered_dominants]

            for dominant_list in dominants_lists:
                array = []
                for dominant in dominant_list:
                    formatted = get_color_formatted_text(dominant.tolist())
                    array.append(formatted)

                print(", ".join(array))
        print(f"{number} {sign} {expected_formatted}  -> {formatted_color}. Maybe: {maybe} {file_link}\n")

def test_clusters():
    passed_tests_counter = 0
    i = 1
    if settings.custom_test is not None:
        crop_path = settings.custom_test["crop_path"]
        color_name = settings.custom_test["expected"]
        test_crop(crop_path, color_name, i)
        return
    target_colors = [settings.test_color]
    # target_colors = settings.COLOR_MAP
    for color_name in target_colors:
        dir_path = Path(settings.test_crops_dir, color_name)
        test_crops = os.listdir(dir_path)
        print(f"Testing {color_name.upper()}")
        for crop_test in test_crops:
            crop_path = Path(dir_path, crop_test)

            result = test_crop(crop_path, color_name, i)
            if result:
                passed_tests_counter += 1
            i += 1
    print(f"Passed {passed_tests_counter} test from {i-1}")


if __name__ == '__main__':
    test_clusters()
