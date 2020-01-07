import re

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start

def get_tile_coordinates(window_name):
    underscore_count = len(re.findall('_', window_name))
    first_underscore = find_nth(window_name, '_', underscore_count - 1)
    second_underscore = find_nth(window_name, '_', underscore_count)

    x_coord = int(window_name[first_underscore + 1: second_underscore])
    y_coord = int(window_name[second_underscore + 1:])
    coordinates = [x_coord, y_coord]
    return coordinates

def get_top_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord - 1) + "_" + str(y_coord)

def get_left_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord) + "_" + str(y_coord - 1)

def get_right_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord) + "_" + str(y_coord + 1)

def get_topleft_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord - 1) + "_" + str(y_coord - 1)

def get_topright_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord - 1) + "_" + str(y_coord + 1)

def get_bottomleft_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord + 1) + "_" + str(y_coord - 1)

def get_bottom_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord + 1) + "_" + str(y_coord)

def get_bottomright_neighbor(window_name):
    x_coord = get_tile_coordinates(window_name)[0]
    y_coord = get_tile_coordinates(window_name)[1]
    return "window_" + str(x_coord + 1) + "_" + str(y_coord + 1)

def get_neighbors(window):
    neighbors = [
        get_topleft_neighbor(window),
        get_top_neighbor(window),
        get_topright_neighbor(window),
        get_left_neighbor(window),
        get_right_neighbor(window),
        get_bottomleft_neighbor(window),
        get_bottom_neighbor(window),
        get_bottomright_neighbor(window)
    ]

    return neighbors