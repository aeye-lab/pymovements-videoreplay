# Copyright (c) 2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""OCR utilities for identifying and locating text regions in images.

This module uses Tesseract OCR via the pytesseract library to detect text
regions in an image. It identifies bounding boxes for recognized text and
computes their center points. Additionally, it provides utilities for finding
the closest bounding boxes in each cardinal direction
(top, left, bottom, right) relative to a given point.
This module is utilized by the fixaioncorrection module
to move fixations directly to the closest word.

Classes
-------
OCR_Reader
    Reads an image and extracts the center positions of text boxes using OCR.

Functions
---------
distance(p1, p2)
    Computes Euclidean distance between two points.

find_closest_top_box(px, py, list_of_centers)
    Finds the nearest text box center
    located above the point (px, py).

find_closest_left_box(px, py, list_of_centers)
    Finds the nearest text box center
    located to the left of the point (px, py).

find_closest_bottom_box(px, py, list_of_centers)
    Finds the nearest text box center
    located below the point (px, py).

find_closest_right_box(px, py, list_of_centers)
    Finds the nearest text box center
    located to the right of the point (px, py).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Return the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_closest_top_box(px: int, py: int, list_of_centers: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the closest word to the top."""
    closest_top_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cy < py and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_top_box = center
    if closest_distance == float('inf'):
        closest_top_box = (px, py)
    return closest_top_box


def find_closest_left_box(px: int, py: int, list_of_centers: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the closest word to the left."""
    closest_left_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cx < px and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_left_box = center
    if closest_distance == float('inf'):
        closest_left_box = (px, py)
    return closest_left_box


def find_closest_bottom_box(px: int, py: int, list_of_centers: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the closest word to the bottom."""
    closest_bottom_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cy > py and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_bottom_box = center
    if closest_distance == float('inf'):
        closest_bottom_box = (px, py)
    return closest_bottom_box


def find_closest_right_box(px: int, py: int, list_of_centers: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the closest word to the right."""
    closest_right_box = None
    closest_distance = float('inf')
    for center in list_of_centers:
        cx, cy = center
        if cx > px and np.hypot(px-cx, py-cy) < closest_distance:
            closest_distance = np.hypot(px-cx, py-cy)
            closest_right_box = center
    if closest_distance == float('inf'):
        closest_right_box = (px, py)
    return closest_right_box


class OCR_Reader:
    """Read an image and extract the center positions of text boxes using OCR.

    Parameters
    ----------
    path_to_image : str
        File path to the image.

    """

    def __init__(self, path_to_image: str):
        self.path_to_image: str = path_to_image
        self.list_of_centers: list[tuple[int, int]] = []

    def get_list_of_centers(self) -> dict:
        """Extract and store the centers of valid OCR-detected text boxes.

        Returns
        -------
        dict
            The full OCR result dictionary from pytesseract.
        """
        img = cv2.imread(self.path_to_image)
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if (
                int(d['conf'][i]) > 60
                and d['width'][i] / d['height'][i] > 0.3
                and d['height'][i] > 2
            ):
                x, y, w, h = (
                    d['left'][i], d['top'][i],
                    d['width'][i], d['height'][i],
                )
                center = (int(x + w / 2), int(y + h / 2))
                self.list_of_centers.append(center)
        return d

    def read_image(self) -> dict:
        """Visualize OCR results by drawing bounding boxes on the image.

        This method is not used by the fixationcorrection module.

        Returns
        -------
        dict
            The full OCR result dictionary from pytesseract.
        """
        img = cv2.imread(self.path_to_image)
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if (
                int(d['conf'][i]) > 60
                and d['width'][i] / d['height'][i] > 0.3
                and d['height'][i] > 2
            ):
                x, y, w, h = (
                    d['left'][i], d['top'][i],
                    d['width'][i], d['height'][i],
                )
                img = cv2.rectangle(
                    img, (x, y), (x + w, y + h), (0, 255, 0), 2,
                )

        cv2.imshow('image', img)
        cv2.waitKey(0)
        return d
