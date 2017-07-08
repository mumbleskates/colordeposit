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
), dtype=np.float32)
CIE_E = 216.0 / 24389.0
xyz_to_lab = np.array((
    (0, 500, 0),
    (116, -500, -200),
    (0, 0, 200),
), dtype=np.float32)


neighbors = np.array((
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
), dtype=np.int32)
weights = np.sum(neighbors**2, axis=1)**-0.5


def log(*s):
    print(*s)


def get_lab_values(colors):
    """Convert RGB integer colors to LAB triple-float32"""
    # get [r, g, b] for each color
    values = np.array((colors & 0xff, (colors & 0xff00) >> 8, (colors & 0xff0000) >> 16), dtype=np.int8).transpose()
    # convert [r, g, b] from sRGB to linear scaled values
    srgb_linear = np.arange(256, dtype=np.float32) / 256
    srgb_linear = np.where(srgb_linear <= 0.04045, srgb_linear / 12.92, np.power((srgb_linear + 0.055) / 1.055, 2.4))
    values = srgb_linear[values]
    # convert linear RGB to XYZ
    values = np.dot(values, rgb_to_xyz)
    # convert XYZ to LAB
    # scale all the values and reshape to have colors on axis 0
    values = np.where(values > CIE_E, np.power(values, 1.0 / 3.0), values * 7.787 + (16.0 / 116.0))
    # final conversion to LAB
    return np.dot(values, xyz_to_lab) - (16, 0, 0)


def main(size=None, origin=None):

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
        inbound_points = surround.compress(inbound, axis=0)
        # get points that are still inbounds and update them
        frontier = inbound_points.compress(canvas[list(inbound_points.T)] == -1, axis=0)
        for affected in frontier:
            update_frontier(affected)

    def update_frontier(point):
        """Update a frontier point on the map"""
        surround = point + neighbors
        inbound = np.alltrue((surround >= 0) & (surround < size), axis=1)
        # remove out of bounds points
        inbound_points = surround.compress(inbound, axis=0)
        wt = weights.compress(inbound)
        # get value indices for inbound points
        surround = canvas[list(inbound_points.T)]
        # remove not-present indices
        present = surround != -1
        surround = surround.compress(present)
        wt = wt.compress(present)
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

    # prepare
    log('allocating')
    canvas = np.full(size, -1, dtype=np.int32)  # indices to colors placed on the image
    colors = np.arange(1 << 24, dtype=np.int32)  # every color, ready to be placed
    log('shuffling')
    np.random.shuffle(colors)
    log('calculating color values')
    values = get_lab_values(colors)
    envelope = tuple(zip(np.amin(values, axis=0), np.amax(values, axis=0)))
    log(f'envelope: {envelope}')

    kdtree = ImplicitKdTree(3, envelope)

    log('placing colors')
    # place first color
    store_frontier_value((0, 0, 0), origin)
    place_color(0, origin)
    for i in tqdm(range(1, w * h)):
        place_color(i, get_best_frontier(values[i]))

    log('writing image')
    img = Image.fromarray(colors[canvas], 'RGBX').convert('RGBA')

    log('saving output file')
    img.save('output.png', 'PNG')
    log('done!')


if __name__ == '__main__':
    main(*sys.argv[1:])
