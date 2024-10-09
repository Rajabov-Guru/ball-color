import os
from pathlib import Path

from identify import identify_ball_color, get_most_spreaded_color
from settings import settings
from utils import which_color_is_it


def get_color_formatted_text(color, text = "  "):
    new_array = map(str, map(int, color))
    return f"\033[48;2;{";".join(new_array)}m{text}\033[0m"


def test_clusters():
    passed_tests_counter = 0
    i = 1
    target_colors = [settings.test_color]
    for color_name in target_colors:
        dir_path = Path(settings.test_crops_dir, color_name)
        test_crops = os.listdir(dir_path)
        print(f"Testing {color_name.upper()}")
        for crop_test in test_crops:
            expected_formatted = get_color_formatted_text(settings.COLOR_MAP[color_name])
            crop_path = Path(dir_path, crop_test)

            result, color, dominant_colors, filtered_dominants = identify_ball_color(str(crop_path))

            if settings.output_dominant_colors:
                dominants_lists = [dominant_colors, filtered_dominants]

                for dominant_list in dominants_lists:
                    array = []
                    for dominant in dominant_list:
                        formatted = get_color_formatted_text(dominant.tolist())
                        array.append(formatted)

                    print(", ".join(array))

            formatted_color = get_color_formatted_text(color)
            maybe, _ = which_color_is_it(color)

            sign = "❌"
            file_link = f"| {str(crop_path)}"
            if result == color_name:
                file_link = ""
                sign = "✅"
                passed_tests_counter += 1
            else:
                print(f"{i} {sign} {expected_formatted}  -> {formatted_color}. Maybe: {maybe} {file_link}\n")
            i += 1
    print(f"Passed {passed_tests_counter} test from {i-1}")


if __name__ == '__main__':
    test_clusters()
