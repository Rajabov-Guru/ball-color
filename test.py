from identify import identify_ball_color, get_most_spreaded_color
from settings import settings
from utils import which_color_is_it
from colorthief import ColorThief


def get_color_formatted_text(color):
    new_array = map(str, map(int, color))
    return f"\033[48;2;{";".join(new_array)}m  \033[0m"

crops_dir = settings.test_crops_dir
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


def test_clusters():
    passed_tests_counter = 0
    i = 1
    for crop_test in crop_tests:
        expected = crop_test[1]
        expected_formatted = get_color_formatted_text(settings.COLOR_MAP[expected])
        crop_path = f"{crops_dir}/{crop_test[0]}"

        result, color, dominant_colors, filtered_dominants = identify_ball_color(crop_path)

        dominants_lists = [dominant_colors, filtered_dominants]

        for dominant_list in dominants_lists:
            array = []
            for dominant in dominant_list:
                formatted = get_color_formatted_text(dominant.tolist())
                array.append(formatted)

            print(", ".join(array))

        formatted_color = get_color_formatted_text(color)
        maybe = which_color_is_it(color)

        sign = "❌"
        if result == expected:
            sign = "✅"
            passed_tests_counter += 1
        print(f"\n{i} {sign} {crop_test[0]} -> expected: {expected} {expected_formatted}  |  result: {result} {formatted_color}. Maybe: {maybe}\n")
        i += 1
    print(f"Passed {passed_tests_counter} test from {len(crop_tests)}")


def test_most_spreaded():
    for crop_test in crop_tests:
        crop_path = f"{crops_dir}/{crop_test[0]}"
        most = get_most_spreaded_color(crop_path, 2)
        print(f"{" - ".join(map(get_color_formatted_text, most))} {crop_test[1]}\n")


if __name__ == '__main__':
    test_most_spreaded()