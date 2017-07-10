# coding=utf-8
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from kdtree import ImplicitKdTree


# RGB-absolute to XYZ conversion matrix (dot mul.)
rgb_to_xyz = np.array((
    (0.412424, 0.357579, 0.180464),
    (0.212656, 0.715158, 0.0721856),
    (0.0193324, 0.119193, 0.950444)
), dtype=np.float64)
CIE_E = 216.0 / 24389.0
xyz_to_lab = np.array((
    (0, 500, 0),
    (116, -500, 200),
    (0, 0, -200),
), dtype=np.float64)


neighbor_sets = {
    '8': np.array((
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    ), dtype=np.int32),
    '4': np.array((
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    ), dtype=np.int32),
    '3': np.array((
        (-1, -1),
        (1, 0),
        (0, 1),
    ), dtype=np.int32),
    'T': np.array((
        (-1, 0),
        (1, 0),
        (0, 1),
    ), dtype=np.int32),
}


def log(*s):
    print(*s)


def get_color_values(colors, colorspace):
    """Convert RGB integer colors to LAB triple-float32"""
    # get [r, g, b] for each color, reorient to have colors along the 0-axis
    values = np.array((colors & 0xff, (colors & 0xff00) >> 8, (colors & 0xff0000) >> 16), dtype=np.uint8).transpose()

    if colorspace == 'srgb':
        return values.astype(np.float64)

    # convert [r, g, b] from sRGB to linear scaled values
    srgb_linear = np.arange(256, dtype=np.float64) / 256
    srgb_linear = np.where(srgb_linear <= 0.04045, srgb_linear / 12.92, np.power((srgb_linear + 0.055) / 1.055, 2.4))
    values = srgb_linear[values]
    # convert linear RGB to XYZ
    values = np.dot(values, rgb_to_xyz)

    if colorspace == 'xyz':
        return values

    # convert XYZ to LAB
    # scale all the values
    values = np.where(values > CIE_E, np.power(values, 1.0 / 3.0), values * 7.787 + (16.0 / 116.0))
    # final conversion to LAB
    values = np.dot(values, xyz_to_lab) - (16, 0, 0)

    if colorspace == 'lab':
        return values


def main(size=None, origin=None, colorspace='lab', sortaxis=None, fuzz=0.0, neighbor_count='8'):

    def store_frontier_value(new_value, point):
        kdtree.set(tuple(point), new_value)

    def get_best_frontier(value):
        point, closest_value, sqdist = kdtree.nearest(value)
        return point

    def delete_frontier_value(point):
        kdtree.remove(tuple(point))

    def place_color(index, point):
        """Place a color on the image"""
        canvas[tuple(point)] = index
        delete_frontier_value(point)
        # get neighboring points inbounds
        surround = point + neighbors
        inbound = np.alltrue((surround >= 0) & (surround < size), axis=1)
        inbound_points = surround[inbound]
        # get points that are still inbounds and update them
        frontier = inbound_points[canvas[list(inbound_points.T)] == -1]
        for affected in frontier:
            update_frontier(affected)

    def update_frontier(point):
        """Update a frontier point on the map"""
        surround = point - neighbors
        inbound = np.alltrue((surround >= 0) & (surround < size), axis=1)
        # remove out of bounds points
        inbound_points = surround[inbound]
        wt = weights[inbound]
        # get value indices for inbound points
        surround = canvas[list(inbound_points.T)]
        # remove not-present indices
        present = surround != -1
        surround = surround[present]
        wt = wt[present]
        # calculate weighted average
        new_value = np.average(values[surround], axis=0, weights=wt)
        # store
        store_frontier_value(new_value, point)

    # init params
    if size is None:
        size = (4096, 4096)
    else:
        size = tuple(map(int, size.split(',')))
    w, h = size
    if origin is None:
        origin = (w // 2, h // 2)
    else:
        origin = tuple(map(int, origin.split(',')))
    colorspace = colorspace.lower()
    if sortaxis:
        sortaxis = sortaxis.lower()

    # prepare
    log('allocating')
    canvas = np.full(size, -1, dtype=np.int32)  # indices to colors placed on the image
    # create every color
    colors = np.arange(1 << 24, dtype=np.uint32) | 0xff000000  # mask max alpha channel
    log('shuffling')
    np.random.shuffle(colors)
    # truncate colors to our image size
    colors = colors[:w * h]
    log('calculating color values')
    values = get_color_values(colors, colorspace)

    if sortaxis and sortaxis != '*':
        log('sorting')
        if sortaxis.startswith('-'):
            sort_dir = -1
            sortaxis = sortaxis[1:]
        else:
            sort_dir = 1
            if sortaxis.startswith('+'):
                sortaxis = sortaxis[1:]

        lookup = {
            ch: (space, idx)
            for space, chs in {
                'srgb': 'rgb',
                'xyz': 'xyz',
                'lab': 'lab',
            }.items()
            for idx, ch in enumerate(chs)
        }
        if sortaxis not in lookup:
            log(f'Error: Invalid sort-axis specified: "{sortaxis}"')
            return
        sort_space_name, idx = lookup[sortaxis]
        if sort_space_name == colorspace:
            sort_space = values
        else:
            sort_space = get_color_values(colors, sort_space_name)
        # get the axis we want
        sort_space = sort_space[:, idx]
        # fuzz sorting
        try:
            fuzz = float(fuzz)
        except:
            fuzz = 0
        if fuzz:
            sort_space += np.random.normal(scale=fuzz, size=sort_space.shape)
        log(
            f'sorting {"reversed " if sort_dir == -1 else ""}by axis "{sortaxis}" from {sort_space_name} '
            f'with {fuzz} normal fuzzing'
        )
        sort_space = np.argsort(sort_space, kind='mergesort')  # use stable sort

        # re-order colors by the sort we determined
        colors = colors[sort_space[::sort_dir]]
        values = values[sort_space[::sort_dir]]

    envelope = tuple(zip(np.amin(values, axis=0), np.amax(values, axis=0)))
    log(f'envelope: {envelope}')

    kdtree = ImplicitKdTree(3, envelope)

    # choose neighbor set
    log(f'frontier will be calculated with {neighbor_count} neighbors')
    neighbors = neighbor_sets[neighbor_count]
    weights = np.sum(neighbors ** 2, axis=1) ** -0.5

    log('placing colors')
    # place first color
    place_color(0, origin)
    for i in tqdm(range(1, w * h)):
        best = get_best_frontier(values[i])
        if not best:
            colors[-1] = 0  # use transparent black for unfilled pixels
            break
        place_color(i, best)

    log('writing image')
    img = Image.fromarray(colors[canvas], 'RGBA')

    log('saving output file')
    img.save('output.png', 'PNG')
    log('done!')


if __name__ == '__main__':
    main(*sys.argv[1:])
